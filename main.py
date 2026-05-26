"""
RAG搭載NPC 最小実装
====================
このスクリプトは「ゲームNPCに知識をRAG(検索拡張生成)で与える」ための最小構成です。

RAG(Retrieval-Augmented Generation)とは:
  LLM(大規模言語モデル)が持っていない特定の知識を、外部データベースから
  「検索して取り出し」、それをLLMへのプロンプトに追加することで、
  より正確・文脈に沿った回答を生成させる手法。

【処理の流れ】
1. ingest: data/lore.md を細かく分割し、各断片をベクトル化してFAISSに保存
2. chat:   ユーザーの発言をベクトル化 → FAISSから関連する断片を検索
           → NPCの設定 + 関連断片 + 会話履歴 を Gemini に渡して応答生成
3. evaluate: eval_questions.yaml の質問を順に投げて結果をMarkdownに保存

【使い方】
  python main.py ingest                          # 初回または知識更新時
  python main.py chat --session test01           # NPCと対話
  python main.py evaluate                        # 評価バッチ実行
"""

# ====================================================================
# セクション1: インポートと定数
# ====================================================================
# このセクションでは必要なライブラリを読み込み、プロジェクト全体で使う
# 定数を定義する。定数は1か所にまとめることで、設定変更が容易になる。
# ====================================================================

import argparse
import json
import os
import pickle
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import faiss        # Facebook AI製のベクトル検索ライブラリ
import numpy as np  # 数値計算ライブラリ(ベクトル演算に使用)
import yaml         # YAML形式の設定ファイルを読み書きするライブラリ
from dotenv import load_dotenv  # .envファイルから環境変数を読み込むライブラリ
from sentence_transformers import SentenceTransformer  # テキストをベクトルに変換するライブラリ

# .envファイルからAPIキー等の環境変数を読み込む
# .envファイルはGitにコミットしないので、秘密情報を安全に管理できる
load_dotenv()

# --- パス定数 ---
# Pathオブジェクトを使うと、OSによるパス区切り文字の違い(/ vs \)を気にしなくて済む
BASE_DIR = Path(__file__).parent                    # main.pyがあるディレクトリ
DATA_DIR = BASE_DIR / "data"                        # データの保存場所
VECTOR_STORE_PATH = DATA_DIR / "vector_store"       # FAISSインデックスの保存先
CONVERSATIONS_DIR = DATA_DIR / "conversations"      # 会話履歴の保存先
EVAL_RESULTS_DIR = BASE_DIR / "eval_results"        # 評価結果の保存先
LORE_FILE = DATA_DIR / "lore.md"                    # NPCに与える世界観テキスト
NPC_CONFIG_FILE = BASE_DIR / "npc_config.yaml"      # NPCのキャラ設定
EVAL_QUESTIONS_FILE = BASE_DIR / "eval_questions.yaml"  # 評価用の質問リスト

# FAISSインデックスとメタデータのファイル名
FAISS_INDEX_FILE = VECTOR_STORE_PATH / "index.faiss"
FAISS_META_FILE = VECTOR_STORE_PATH / "meta.pkl"

# --- チャンキング設定 ---
# 日本語1チャンク400文字: Geminiのコンテキスト圧迫を避けつつ、
# 意味のある文脈が1つのチャンクに収まる程度の長さ
CHUNK_SIZE = 400

# オーバーラップとは「隣り合うチャンク同士で重複させる文字数」。
# チャンクの境界で文章が切れても、次のチャンクの先頭で重複部分があるため
# 文脈が失われにくくなる。例: チャンクAの末尾50文字がチャンクBの先頭に含まれる
CHUNK_OVERLAP = 50

# --- 検索設定 ---
# RAG検索で取得する関連チャンクの上限数。
# 多いほど豊富な文脈を与えられるが、プロンプトが長くなりLLMのコスト・遅延が増す
TOP_K = 4

# --- モデル設定 ---
# multilingual-e5-smallを選んだ理由:
#   - 日本語に対応(Multilingual)
#   - 軽量(small)でローカル動作が速い
#   - e5シリーズは検索タスクで高い精度を誇る
EMBEDDING_MODEL = "intfloat/multilingual-e5-small"

# Gemini 2.0 flash: Googleの最新世代モデル。高速かつ低コストで、
# 短〜中長程度の応答生成に適している
GEMINI_MODEL = "gemini-2.0-flash-exp"

# グローバルにモデルをキャッシュする変数
# 毎回ロードするとメモリ・時間を無駄遣いするため、1度だけロードして再利用する
_embedding_model: Optional[SentenceTransformer] = None


# ====================================================================
# セクション2: 埋め込み(Embedding)関連
# ====================================================================
# Embedding(埋め込み)とは、テキストを数値のベクトル(多次元の座標)に変換する処理。
# 意味が近いテキストは、ベクトル空間でも近い位置に配置される性質を持つ。
# これにより「意味的な検索」が可能になる。
# ====================================================================

def load_embedding_model() -> SentenceTransformer:
    """
    Embeddingモデルをロードする(初回のみ。2回目以降はキャッシュを返す)。

    Returns:
        SentenceTransformer: ロード済みのEmbeddingモデル
    """
    global _embedding_model
    if _embedding_model is None:
        print(f"Embeddingモデル '{EMBEDDING_MODEL}' をロード中... (初回は数十秒かかります)")
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL)
        print("Embeddingモデルのロード完了。")
    return _embedding_model


def embed_texts(texts: list[str], is_query: bool = False) -> np.ndarray:
    """
    テキストのリストをベクトル(numpy配列)に変換する。

    multilingual-e5-small の仕様上、入力テキストに以下のプレフィックスを
    付けないと性能が大きく低下する:
      - 検索対象(DB側): "passage: " を先頭に付ける
      - 検索クエリ(質問側): "query: " を先頭に付ける
    この関数が内部で自動的に付与するため、呼び出し側は意識しなくてよい。

    Args:
        texts: ベクトル化したいテキストのリスト
        is_query: Trueなら "query: " プレフィックスを付ける(検索クエリ用)
                  Falseなら "passage: " プレフィックスを付ける(保存するデータ用)

    Returns:
        np.ndarray: shape=(len(texts), 埋め込み次元数) のfloat32配列
    """
    model = load_embedding_model()

    # プレフィックスを付加してモデルに渡す
    prefix = "query: " if is_query else "passage: "
    prefixed = [prefix + t for t in texts]

    # encode()でベクトルに変換。normalize_embeddings=Trueで正規化する。
    # 正規化(ベクトルの長さを1に揃える)をする理由:
    #   FAISSのIndexFlatIPは「内積(inner product)」で類似度を計算する。
    #   ベクトルを正規化しておくと「内積 = コサイン類似度」になり、
    #   テキストの「意味の近さ」を正しく計算できる。
    vectors = model.encode(prefixed, normalize_embeddings=True, show_progress_bar=False)
    return vectors.astype(np.float32)


# ====================================================================
# セクション3: ベクトルストア(FAISS)関連
# ====================================================================
# FAISSはFacebook AI Research製の高速ベクトル検索ライブラリ。
# 数百万件のベクトルから最も近いものを瞬時に見つけることができる。
# ====================================================================

def build_vector_store(chunks: list[dict]) -> faiss.IndexFlatIP:
    """
    チャンクのリストからFAISSインデックスを構築する。

    Args:
        chunks: チャンク情報の辞書リスト。各辞書は {"text": str, ...} を含む

    Returns:
        faiss.IndexFlatIP: 構築済みのFAISSインデックス
    """
    texts = [c["text"] for c in chunks]
    vectors = embed_texts(texts, is_query=False)

    # IndexFlatIPを使う理由:
    #   IP = Inner Product(内積)。ベクトルを正規化済みであれば
    #   内積はコサイン類似度と等価になる(値が大きいほど意味が近い)。
    #   "Flat"は全件総当たり検索を意味し、インデックスサイズが小さい研究用途では
    #   最も精度が高い方法。
    dim = vectors.shape[1]  # ベクトルの次元数(モデルにより決まる)
    index = faiss.IndexFlatIP(dim)
    index.add(vectors)  # すべてのベクトルをインデックスに追加
    return index


def save_vector_store(index: faiss.IndexFlatIP, metadatas: list[dict]) -> None:
    """
    FAISSインデックスとメタデータをファイルに保存する。

    FAISSインデックス自体にはテキスト情報が含まれないため、
    「どの番号がどのチャンクか」を対応づけるメタデータを
    別ファイル(pickleファイル)に保存する必要がある。

    Args:
        index: 保存するFAISSインデックス
        metadatas: 各ベクトルに対応するメタデータのリスト
    """
    VECTOR_STORE_PATH.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(FAISS_INDEX_FILE))
    with open(FAISS_META_FILE, "wb") as f:
        pickle.dump(metadatas, f)
    print(f"ベクトルストアを保存しました: {VECTOR_STORE_PATH}")


def load_vector_store() -> tuple[faiss.IndexFlatIP, list[dict]]:
    """
    保存済みのFAISSインデックスとメタデータを読み込む。

    Returns:
        tuple: (FAISSインデックス, メタデータリスト)

    Raises:
        SystemExit: インデックスファイルが見つからない場合
    """
    if not FAISS_INDEX_FILE.exists():
        print("エラー: ベクトルストアが見つかりません。")
        print("  まず 'python main.py ingest' を実行してください。")
        sys.exit(1)

    index = faiss.read_index(str(FAISS_INDEX_FILE))
    with open(FAISS_META_FILE, "rb") as f:
        metadatas = pickle.load(f)
    return index, metadatas


def search_vector_store(query: str, k: int = TOP_K) -> list[dict]:
    """
    クエリテキストに意味的に近いチャンクをベクトルストアから検索する。

    Args:
        query: 検索クエリ(ユーザーの発言など)
        k: 取得する上位件数

    Returns:
        list[dict]: 類似度スコア付きのメタデータリスト(スコア降順)
    """
    index, metadatas = load_vector_store()

    # クエリをベクトルに変換(is_query=Trueで "query: " プレフィックスが付く)
    query_vector = embed_texts([query], is_query=True)

    # FAISSで最近傍検索: distances=類似度スコア, indices=チャンク番号
    distances, indices = index.search(query_vector, k)

    results = []
    for score, idx in zip(distances[0], indices[0]):
        if idx == -1:  # FAISSが結果なしを示す場合はスキップ
            continue
        result = metadatas[idx].copy()
        result["score"] = float(score)  # 類似度スコアを付与
        results.append(result)

    return results


# ====================================================================
# セクション4: テキスト分割(チャンキング)
# ====================================================================
# 長いテキストをそのままベクトル化すると「どの部分を聞いているか」が
# 曖昧になり検索精度が落ちる。適切なサイズに分割することで、
# 質問に対して本当に関連する部分だけを取り出せるようになる。
# ====================================================================

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """
    長いテキストを指定サイズのチャンク(断片)に分割する。

    日本語向けの簡易チャンキング戦略:
      1. まず空行(段落)で分割する
      2. 各段落を句点(。)で文単位に分割する
      3. chunk_size文字以内に収まるよう文を束ねる
      4. 隣り合うチャンク間でoverlapぶん重複させる(文脈の切れ目対策)

    オーバーラップを設ける理由:
      チャンクの境界で重要な情報が「前後に分かれて」しまうことがある。
      先頭と末尾をわずかに重複させることで、境界付近の情報が落ちにくくなる。

    Args:
        text: 分割したいテキスト全文
        chunk_size: 1チャンクの最大文字数
        overlap: 隣り合うチャンク間で重複させる文字数

    Returns:
        list[str]: 分割されたテキストチャンクのリスト
    """
    # まず空行で段落に分割する
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    # 段落をさらに句点で文に分割してフラットなリストにする
    sentences: list[str] = []
    for para in paragraphs:
        # 句点で分割し、句点自体を文末に残す
        parts = para.replace("。", "。\n").split("\n")
        for part in parts:
            part = part.strip()
            if part:
                sentences.append(part)

    # 文を束ねてチャンクを作る
    chunks: list[str] = []
    current_chunk = ""

    for sentence in sentences:
        # 現在のチャンクに文を追加してもchunk_size以内なら追加
        if len(current_chunk) + len(sentence) <= chunk_size:
            current_chunk += sentence
        else:
            # チャンクが埋まったら確定させて新しいチャンクを開始
            if current_chunk:
                chunks.append(current_chunk)
            # オーバーラップ: 前のチャンクの末尾overlap文字を次のチャンクの先頭に含める
            overlap_text = current_chunk[-overlap:] if len(current_chunk) > overlap else current_chunk
            current_chunk = overlap_text + sentence

    # 最後のチャンクを追加
    if current_chunk:
        chunks.append(current_chunk)

    return chunks


# ====================================================================
# セクション5: 会話履歴管理
# ====================================================================
# 【研究メモ】
# 会話が長くなるとここで返すメッセージ数が多くなり、LLMのコンテキストを圧迫する。
# 将来的に「要約」「重要発話の抽出」「ベクトル検索による関連発話のみ取得」などに差し替える予定。
# 現在は最新N件を素直に返すだけのナイーブ実装。
#
# この部分が研究のコア対象になる。chat_with_npc()から呼ぶ箇所を変えなければ
# 内部の実装は自由に差し替えられる設計にしている。
# ====================================================================

def save_message(session_id: str, role: str, content: str) -> None:
    """
    1件の発言を会話履歴ファイルに追記する。

    JSONL(JSON Lines)形式: 1行が1つのJSONオブジェクト。
    追記が簡単で、巨大なファイルでも1行ずつ読めるため大きな会話履歴に向いている。

    Args:
        session_id: セッションの識別子(例: "test01")
        role: 発言者の役割 ("user" or "model")
        content: 発言内容
    """
    CONVERSATIONS_DIR.mkdir(parents=True, exist_ok=True)
    filepath = CONVERSATIONS_DIR / f"{session_id}.jsonl"

    record = {
        "timestamp": datetime.now().isoformat(),
        "role": role,
        "content": content,
    }

    with open(filepath, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_recent_messages(session_id: str, n: int = 10) -> list[dict]:
    """
    会話履歴から直近N件のメッセージを読み込む。

    【現在の実装】最新N件をそのまま返す(ナイーブ実装)。
    【将来の差し替え候補】
      - 要約: 古い発言をLLMで要約してから渡す
      - 抽出: 現在の質問と意味的に近い発言のみをRAGで取得する
      - 圧縮: 重複・冗長な発言を除去してから渡す

    Args:
        session_id: セッションの識別子
        n: 取得する直近の発言件数

    Returns:
        list[dict]: {"role": str, "content": str} の辞書リスト(古い順)
    """
    filepath = CONVERSATIONS_DIR / f"{session_id}.jsonl"
    if not filepath.exists():
        return []

    lines = filepath.read_text(encoding="utf-8").strip().split("\n")
    # 直近N件だけ取り出す(リストの末尾N件)
    recent_lines = lines[-n:] if len(lines) > n else lines

    messages = []
    for line in recent_lines:
        if line.strip():
            record = json.loads(line)
            messages.append({"role": record["role"], "content": record["content"]})

    return messages


# ====================================================================
# セクション6: LLM呼び出し
# ====================================================================
# 【重要】この関数を差し替えるだけで、別のLLMに移行できる設計になっている。
#
# 現在: Google Gemini API を使用
# 将来の差し替え候補:
#   - ローカルLLM (Ollama, llama.cpp など): APIコールをローカルHTTPリクエストに変更
#   - ファインチューニング済みモデル: モデル名を変更するだけで対応可能
#   - OpenAI API: メッセージ形式が異なるため変換処理を追加
#
# call_llm()の引数と返り値の型を変えなければ、呼び出し側のコードは一切変更不要。
# ====================================================================

def call_llm(system_prompt: str, messages: list[dict]) -> str:
    """
    LLM(Gemini)にシステムプロンプトと会話履歴を渡して応答を生成する。

    Args:
        system_prompt: NPCのペルソナや世界観の文脈を含むシステムプロンプト
        messages: 会話履歴。[{"role": "user"/"model", "content": str}, ...] の形式

    Returns:
        str: LLMが生成したテキスト応答

    Raises:
        SystemExit: APIキーが設定されていない場合
    """
    import google.generativeai as genai  # Gemini SDKは使う直前にインポート(差し替え時の影響範囲を限定)

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("エラー: GEMINI_API_KEY が設定されていません。")
        print("  .env ファイルを作成して GEMINI_API_KEY=<your_key> を設定してください。")
        sys.exit(1)

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(
        model_name=GEMINI_MODEL,
        system_instruction=system_prompt,  # システムプロンプトはモデル初期化時に設定
    )

    # Gemini SDKのメッセージ形式に変換して渡す
    history = []
    # 最後のユーザー発言は history に含めず、send_message()で渡す
    for msg in messages[:-1]:
        history.append({"role": msg["role"], "parts": [msg["content"]]})

    chat = model.start_chat(history=history)

    # 最後のメッセージ(=最新のユーザー発言)を送信して応答を受け取る
    last_user_message = messages[-1]["content"] if messages else ""
    response = chat.send_message(last_user_message)

    return response.text


# ====================================================================
# セクション7: NPCエージェント(本体)
# ====================================================================
# このセクションがシステムの中核。NPCのペルソナとRAG検索結果を組み合わせて
# LLMへのプロンプトを組み立て、応答を生成する。
# ====================================================================

def load_npc_config() -> dict:
    """
    npc_config.yaml からNPCの設定を読み込む。

    Returns:
        dict: NPCの設定情報

    Raises:
        SystemExit: 設定ファイルが見つからない場合
    """
    if not NPC_CONFIG_FILE.exists():
        print(f"エラー: NPCの設定ファイルが見つかりません: {NPC_CONFIG_FILE}")
        sys.exit(1)

    with open(NPC_CONFIG_FILE, encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_system_prompt(npc_config: dict, retrieved_chunks: list[dict]) -> str:
    """
    NPCのペルソナ + RAGで検索した世界観断片 + 行動制約 を組み合わせた
    システムプロンプトを組み立てる。

    プロンプトの構造を明確に分けることで、各部分の効果を実験しやすくなる:
      [NPC設定] → キャラクターとして振る舞うための指示
      [参照知識] → RAGで取得した世界観テキスト(根拠となる情報)
      [制約]    → 守るべきルール(現代語を使わない など)

    Args:
        npc_config: npc_config.yaml から読み込んだNPC設定
        retrieved_chunks: FAISSの検索で得られた関連チャンクのリスト

    Returns:
        str: LLMに渡すシステムプロンプト文字列
    """
    persona = npc_config.get("persona", {})
    constraints = npc_config.get("constraints", [])
    background = npc_config.get("background", "")

    # RAGで取得したチャンクをまとめて「参照知識」セクションにする
    knowledge_section = ""
    if retrieved_chunks:
        knowledge_pieces = []
        for i, chunk in enumerate(retrieved_chunks, 1):
            knowledge_pieces.append(f"【参照{i}】{chunk['text']}")
        knowledge_section = "\n\n".join(knowledge_pieces)
    else:
        knowledge_section = "（関連する記録が見つかりませんでした）"

    # 制約リストを箇条書きに変換
    constraints_text = "\n".join(f"- {c}" for c in constraints)

    # システムプロンプトのテンプレート
    # プロンプトを構造化すると、LLMが各情報の役割を理解しやすくなる
    prompt = f"""あなたは以下のNPCとして振る舞ってください。

=== キャラクター設定 ===
名前: {npc_config.get('name', '不明')}
年齢: {persona.get('age', '不明')}歳
役割: {persona.get('role', '不明')}
性格: {persona.get('personality', '')}
話し方: {persona.get('speaking_style', '')}

背景:
{background}

=== 参照知識(図書館の記録より) ===
以下は、あなたが知っている世界の記録です。回答の根拠として使ってください。

{knowledge_section}

=== 行動制約 ===
{constraints_text}

ユーザーとの会話に、上記のキャラクター設定・参照知識・行動制約を忠実に反映して応答してください。
参照知識に記載のない内容について聞かれた場合も、キャラクターとして自然に応答してください。"""

    return prompt


def chat_with_npc(session_id: str, user_message: str, debug: bool = False) -> str:
    """
    ユーザーの発言を受け取り、NPCとしての応答を生成して返す。
    RAGパイプライン全体がこの関数1つで完結する。

    Args:
        session_id: セッションの識別子(会話履歴の保存先を特定するために使う)
        user_message: ユーザーの発言テキスト
        debug: Trueにすると検索ヒットしたチャンクの情報を表示する

    Returns:
        str: NPCの応答テキスト
    """
    # ステップ1: ユーザー発言を埋め込みベクトルに変換
    # ベクトル化することで「意味的な類似検索」が可能になる
    # (例: 「戦争はいつ?」と「魔法戦争の時期は?」が同じチャンクにヒットする)

    # ステップ2: ベクトルストアから関連する世界観チャンクをTop-K件取得
    retrieved_chunks = search_vector_store(user_message, k=TOP_K)

    if debug:
        print("\n--- [DEBUG] 検索ヒットしたチャンク ---")
        for i, chunk in enumerate(retrieved_chunks, 1):
            print(f"[{i}] スコア={chunk['score']:.4f} | {chunk['text'][:80]}...")
        print("--------------------------------------\n")

    # ステップ3: NPC設定 + 検索結果 + 直近の会話履歴 でプロンプトを組み立て
    npc_config = load_npc_config()
    system_prompt = build_system_prompt(npc_config, retrieved_chunks)

    # 会話履歴を読み込み、最新ユーザー発言を末尾に追加
    history = load_recent_messages(session_id)
    messages = history + [{"role": "user", "content": user_message}]

    # ステップ4: Geminiに送って応答を取得
    npc_response = call_llm(system_prompt, messages)

    # ステップ5: ユーザー発言とNPC応答の両方を会話履歴に保存
    # 両方を保存することで、次回以降の会話に文脈が引き継がれる
    save_message(session_id, "user", user_message)
    save_message(session_id, "model", npc_response)

    return npc_response


# ====================================================================
# セクション8: 各コマンド実装
# ====================================================================
# CLIの各サブコマンド(ingest/chat/evaluate)の処理を実装する。
# メイン処理とユーティリティ関数を分離することで、
# 将来的にWebサーバーやGUIに差し替えても再利用しやすくなる。
# ====================================================================

def cmd_ingest() -> None:
    """
    data/lore.md を読み込み、チャンク分割してFAISSに保存する。

    このコマンドは「知識をNPCに与える」処理。
    lore.md の内容を更新したら再実行が必要。
    複数のLLMを使って自動生成したテキストをlore.mdに追記するだけで
    RAGデータの実験ができる設計にしている(研究方向1への対応)。
    """
    print("=" * 50)
    print("Ingest(知識の取り込み)を開始します")
    print("=" * 50)

    # lore.md の存在確認
    if not LORE_FILE.exists():
        print(f"エラー: 世界観ファイルが見つかりません: {LORE_FILE}")
        sys.exit(1)

    print("[1/4] 世界観テキストを読み込み中...")
    text = LORE_FILE.read_text(encoding="utf-8")
    print(f"  読み込んだ文字数: {len(text)} 文字")

    print("[2/4] テキストをチャンクに分割中...")
    chunks_text = chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP)
    print(f"  分割されたチャンク数: {len(chunks_text)} チャンク")

    # 各チャンクにメタデータを付与する
    # メタデータはFAISSには保存できないため別ファイルで管理する
    chunks = []
    for i, chunk_str in enumerate(chunks_text):
        chunks.append({
            "chunk_id": i,
            "text": chunk_str,
            "source": str(LORE_FILE.name),  # どのファイルから来たかを記録
        })

    print("[3/4] チャンクをベクトル化してFAISSインデックスを構築中...")
    index = build_vector_store(chunks)
    print(f"  インデックスに登録したベクトル数: {index.ntotal}")

    print("[4/4] ベクトルストアを保存中...")
    save_vector_store(index, chunks)

    print()
    print("Ingest完了! 次は 'python main.py chat --session <セッション名>' でNPCと会話できます。")


def cmd_chat(session_id: str, debug: bool = False) -> None:
    """
    NPCとの対話ループを起動する。

    コマンド:
      /exit  - 会話を終了する
      /reset - このセッションの会話履歴を削除してリセットする
      /debug - 検索ヒット情報の表示をON/OFFで切り替える

    Args:
        session_id: セッションの識別子(例: "test01")
        debug: デバッグ表示の初期状態
    """
    npc_config = load_npc_config()
    npc_name = npc_config.get("name", "NPC")

    print("=" * 50)
    print(f"{npc_name} との会話を開始します (セッション: {session_id})")
    print("終了: /exit | 履歴リセット: /reset | デバッグ切替: /debug")
    print("=" * 50)

    # セッション開始時に既存の履歴件数を表示
    history = load_recent_messages(session_id)
    if history:
        print(f"(前回の会話履歴 {len(history)} 件を引き継いでいます)\n")

    while True:
        try:
            user_input = input("あなた: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n会話を終了します。")
            break

        if not user_input:
            continue

        # 特殊コマンドの処理
        if user_input == "/exit":
            print("会話を終了します。")
            break

        if user_input == "/reset":
            session_file = CONVERSATIONS_DIR / f"{session_id}.jsonl"
            if session_file.exists():
                session_file.unlink()  # ファイルを削除
                print(f"セッション '{session_id}' の会話履歴を削除しました。\n")
            else:
                print("削除する会話履歴がありません。\n")
            continue

        if user_input == "/debug":
            debug = not debug
            state = "ON" if debug else "OFF"
            print(f"デバッグ表示を {state} にしました。\n")
            continue

        # NPCの応答を生成
        print(f"{npc_name}: ", end="", flush=True)
        try:
            response = chat_with_npc(session_id, user_input, debug=debug)
            print(response)
        except Exception as e:
            print(f"\nエラーが発生しました: {e}")
            print("APIキーや通信状況を確認してください。")
        print()


def cmd_evaluate() -> None:
    """
    eval_questions.yaml の質問を一括で実行し、結果をMarkdownファイルに保存する。

    RAGが機能しているかを客観的に評価するためのバッチ処理。
    「lore.mdに答えがある質問」と「ない質問」を混ぜることで、
    RAGの有無による回答の違いを確認できる。
    """
    print("=" * 50)
    print("評価バッチを開始します")
    print("=" * 50)

    # 評価用質問ファイルの読み込み
    if not EVAL_QUESTIONS_FILE.exists():
        print(f"エラー: 評価質問ファイルが見つかりません: {EVAL_QUESTIONS_FILE}")
        sys.exit(1)

    with open(EVAL_QUESTIONS_FILE, encoding="utf-8") as f:
        eval_data = yaml.safe_load(f)

    questions = eval_data.get("questions", [])
    if not questions:
        print("エラー: 評価質問が見つかりません。eval_questions.yaml を確認してください。")
        sys.exit(1)

    npc_config = load_npc_config()
    npc_name = npc_config.get("name", "NPC")

    # 結果保存先の準備
    EVAL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = EVAL_RESULTS_DIR / f"eval_{timestamp}.md"

    # Markdownレポートのヘッダーを書き込む
    lines = [
        f"# 評価レポート",
        f"",
        f"- 実行日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- NPC: {npc_name}",
        f"- モデル: {GEMINI_MODEL}",
        f"- Embeddingモデル: {EMBEDDING_MODEL}",
        f"- TOP_K: {TOP_K}",
        f"",
        f"---",
        f"",
    ]

    # 各質問を処理する
    for i, q in enumerate(questions, 1):
        q_id = q.get("id", f"q{i:02d}")
        q_text = q.get("text", "")
        q_topic = q.get("expected_topic", "")

        print(f"[{i}/{len(questions)}] {q_id}: {q_text}")

        # RAG検索を実行して関連チャンクを取得
        retrieved = search_vector_store(q_text, k=TOP_K)

        # チャットとは別の評価専用セッションで応答を生成
        # セッションIDに "eval_" + timestamp を使うことで本番の会話と混ざらない
        eval_session_id = f"eval_{timestamp}"
        try:
            response = chat_with_npc(eval_session_id, q_text, debug=False)
        except Exception as e:
            response = f"[エラー: {e}]"

        # 検索ヒットしたチャンクのソース情報をまとめる
        sources_text = ""
        if retrieved:
            source_lines = []
            for r in retrieved:
                score = r.get("score", 0.0)
                text_preview = r.get("text", "")[:60].replace("\n", " ")
                source_lines.append(f"  - スコア={score:.4f}: {text_preview}...")
            sources_text = "\n".join(source_lines)
        else:
            sources_text = "  (ヒットなし)"

        # Markdownの各質問セクションを追記
        lines.extend([
            f"## {q_id}: {q_text}",
            f"",
            f"**期待トピック**: {q_topic}",
            f"",
            f"**検索ヒットしたチャンク**:",
            f"",
            f"{sources_text}",
            f"",
            f"**NPCの応答**:",
            f"",
            f"> {response.replace(chr(10), chr(10) + '> ')}",
            f"",
            f"---",
            f"",
        ])

        print(f"  → 応答を記録しました")

    # 結果をファイルに書き出す
    result_file.write_text("\n".join(lines), encoding="utf-8")
    print()
    print(f"評価完了! 結果を保存しました: {result_file}")


# ====================================================================
# セクション9: エントリポイント
# ====================================================================
# CLIのサブコマンドを解析して、対応する関数を呼び出す。
# argparseを使うことで --help でコマンド説明が自動生成される。
# ====================================================================

def main() -> None:
    """
    コマンドライン引数を解析してサブコマンドを実行する。
    """
    parser = argparse.ArgumentParser(
        description="RAG搭載NPC 最小実装 — ゲームNPCに知識をRAGで与える",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  python main.py ingest                       # 知識の取り込み(初回必須)
  python main.py chat --session test01        # NPCと対話
  python main.py chat --session test01 --debug # デバッグ表示あり
  python main.py evaluate                     # 評価バッチ実行
        """
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # ingest サブコマンド
    subparsers.add_parser(
        "ingest",
        help="data/lore.md を読み込んでFAISSに保存する(初回または知識更新時に実行)"
    )

    # chat サブコマンド
    chat_parser = subparsers.add_parser(
        "chat",
        help="NPCと対話する"
    )
    chat_parser.add_argument(
        "--session",
        required=True,
        help="セッションID(例: test01)。会話履歴の保存先ファイル名になる"
    )
    chat_parser.add_argument(
        "--debug",
        action="store_true",
        help="RAG検索のヒット情報をリアルタイムで表示する"
    )

    # evaluate サブコマンド
    subparsers.add_parser(
        "evaluate",
        help="eval_questions.yaml の質問を一括実行して評価レポートを生成する"
    )

    args = parser.parse_args()

    # サブコマンドに応じた関数を呼び出す
    if args.command == "ingest":
        cmd_ingest()
    elif args.command == "chat":
        cmd_chat(session_id=args.session, debug=args.debug)
    elif args.command == "evaluate":
        cmd_evaluate()


if __name__ == "__main__":
    main()
