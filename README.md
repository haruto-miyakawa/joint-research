# RAG搭載NPC 研究用プロジェクト

ゲームNPCに RAG(検索拡張生成)で知識を持たせる仕組みの最小実装です。
共同研究の土台として使います。

---

## このプロジェクトでできること

- `data/lore.md` に書いた世界観をNPCに「知識」として持たせる
- NPCと対話できる(キャラ設定は `npc_config.yaml` で管理)
- テスト質問を一括で投げて応答品質を確認できる

---

## 動作環境

- **Windows + WSL2(Ubuntu) + VS Code Remote-WSL** を想定
- Python 3.11 以上
- Gemini API キー(無料枠あり)

WSL接続済みの VS Code のターミナル(Ubuntu)で作業する前提で書きます。

---

## セットアップ手順(初回のみ)

### ステップ1: リポジトリをクローン

VS Code で WSL に接続した状態でターミナルを開き、作業したいフォルダに移動してから:

```bash
git clone <このリポジトリのURL>
cd <クローンされたフォルダ名>
```

リポジトリのURLは GitHub のページの緑色の `<> Code` ボタンから `HTTPS` のURLをコピーしてください。

### ステップ2: Python と必要なパッケージの確認

WSL の Ubuntu に Python 3.11+ が入っていることを確認:

```bash
python3 --version
```

`Python 3.11.x` 以上が表示されればOK。古いか入っていない場合は:

```bash
sudo apt update
sudo apt install python3.11 python3.11-venv python3-pip
```

### ステップ3: 仮想環境を作って入る

プロジェクトごとにライブラリを分けるためのおまじないです:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

成功するとターミナルの行頭に `(.venv)` が付きます。これが付いている間は仮想環境の中にいます。

> **メモ:** 次回以降ターミナルを開き直したときは `source .venv/bin/activate` だけ実行すれば仮想環境に戻れます。

### ステップ4: 必要なライブラリをインストール

```bash
pip install -r requirements.txt
```

初回は埋め込みモデル関連のダウンロードもあるので数分かかります。

### ステップ5: Gemini API キーを取得して設定

1. [Google AI Studio](https://aistudio.google.com/apikey) にアクセス(Googleアカウントでログイン)
2. `Create API key` をクリックしてキーを発行
3. 表示されたキー(`AIza...` で始まる文字列)をコピー

そのあとプロジェクト内で:

```bash
cp .env.example .env
```

VS Code の左側のファイル一覧から `.env` を開いて、`your_api_key_here` の部分を実際のキーに置き換えて保存します。

```
GEMINI_API_KEY=AIza...(コピーしたキー)
```

> **⚠️ 注意:** `.env` は `.gitignore` に入っているのでGitHubには上がりませんが、間違っても他人に共有しないでください。無料枠を使い切られたり、有料プランの場合は課金が発生します。

---

## 動かしてみる(セットアップ後の通常作業)

このプロジェクトは3つのコマンドで動きます。

### 1. ingest(知識をNPCに覚えさせる)

```bash
python main.py ingest
```

`data/lore.md` の世界観テキストを細かく分割して、検索できる形(FAISSインデックス)で `data/vector_store/` に保存します。

**いつ実行するか:**
- 一番最初(必須)
- `data/lore.md` を編集したあと

NPC設定(`npc_config.yaml`)を変えただけのときは再実行不要です。

---

### 2. chat(NPCと対話する)

```bash
python main.py chat --session test01
```

`--session test01` の `test01` は会話セッションの名前です。好きな名前でOK。同じ名前を指定すれば前回の続きから話せます。

**対話中のコマンド:**
| 入力 | 動作 |
|---|---|
| 普通に文章を入力 | NPCに話しかける |
| `/exit` | 対話を終了 |
| `/reset` | このセッションの会話履歴を削除 |
| `/debug` | 検索でヒットしたチャンクの表示ON/OFF |

**デバッグモードで起動する場合:**

```bash
python main.py chat --session test01 --debug
```

NPCがどの世界観チャンクを参照して答えているかが見えるので、最初はこちらがおすすめ。

---

### 3. evaluate(テスト質問を一括で投げる)

```bash
python main.py evaluate
```

`eval_questions.yaml` に書いた質問リストを順番にNPCに投げて、結果を `eval_results/eval_YYYYMMDD_HHMMSS.md` に保存します。

研究の比較・進捗確認に使います。

---

## ファイル構成

```
.
├── README.md                  # このファイル
├── requirements.txt           # 必要なライブラリ一覧
├── .env.example               # APIキー設定のテンプレート
├── .env                       # ★ 自分で作る(APIキーを書く)
├── .gitignore                 # GitHubに上げないファイル一覧
├── main.py                    # ★ 本体。全機能ここに入っている
├── npc_config.yaml            # NPCのキャラ設定
├── eval_questions.yaml        # 評価用の質問リスト
└── data/
    ├── lore.md                # NPCに与える世界観テキスト
    ├── vector_store/          # FAISSインデックス(ingestで自動生成)
    └── conversations/         # 会話履歴(chatで自動生成)
```

---

## 仕組み(処理の流れ)

```
[ユーザー発言]
     ↓
[Embeddingモデルでベクトル化]
     ↓
[FAISSで関連する世界観チャンクを検索 (Top-K件)]
     ↓
[NPC設定 + 検索結果 + 会話履歴 をGeminiに送信]
     ↓
[NPCの応答が返ってくる]
     ↓
[ユーザー発言とNPC応答を会話履歴に保存]
```

`main.py` を上から読むと、この流れがそのままセクションごとに書かれています。

---

## よくあるトラブル

| 症状 | 原因と対処 |
|---|---|
| `ModuleNotFoundError: No module named 'faiss'` など | 仮想環境に入っていない。`source .venv/bin/activate` を実行 |
| `GEMINI_API_KEY が見つかりません` | `.env` ファイルが未作成、またはキーが書かれていない |
| `chat` で「インデックスが見つかりません」 | 先に `python main.py ingest` を実行する必要がある |
| `python3: command not found` | Python未インストール。セットアップ ステップ2 を参照 |
| 文字化けする | WSL のターミナルが UTF-8 か確認。VS Code のターミナルなら通常問題なし |
| `Permission denied` 系 | `sudo` を使うか、`.venv` フォルダの権限を確認 |

解決しないときは、エラーメッセージ全文をコピーして共有してください。

---

## 自分の研究範囲を編集するときに触る場所

| やりたいこと | 編集するファイル |
|---|---|
| NPCのキャラ設定を変える | `npc_config.yaml` |
| NPCに与える知識を変える | `data/lore.md` → `python main.py ingest` で反映 |
| 評価用の質問を増やす | `eval_questions.yaml` |
| 検索や生成のロジックを変える | `main.py`(セクション番号付きで分かれている) |

`main.py` の各セクションが何の役割か:

| セクション | 役割 | 研究で触る可能性 |
|---|---|---|
| 1. インポートと定数 | チャンクサイズなどの設定 | チャンク戦略の研究で触る |
| 2. 埋め込み(Embedding) | テキスト → ベクトル変換 | 別の埋め込みモデルを試すとき |
| 3. ベクトルストア(FAISS) | ベクトルの保存・検索 | 検索手法の改良 |
| 4. テキスト分割(チャンキング) | テキストを断片に分ける | **RAGデータの作り方の研究で重点** |
| 5. 会話履歴管理 | 過去発言の保存・読み込み | **会話履歴蓄積の研究で重点** |
| 6. LLM呼び出し | Gemini API呼び出し | ローカルLLMやFTモデルに差し替えるとき |
| 7. NPCエージェント | プロンプト組み立て・対話処理 | キャラ表現の改善 |
| 8. 各コマンド実装 | ingest/chat/evaluate の中身 | 通常は触らない |
| 9. エントリポイント | コマンドライン引数の処理 | 通常は触らない |

---

## 開発の流れ(自分の変更を共有するとき)

研究中の変更をチームに共有するときの基本手順:

```bash
# 自分用のブランチを作って作業
git checkout -b feature/自分の名前-変更内容

# ファイルを編集...

# 変更を確認
git status
git diff

# コミット
git add .
git commit -m "変更内容の説明"

# GitHubにプッシュ
git push origin feature/自分の名前-変更内容
```

そのあと GitHub のページで Pull Request を作ってチームでレビューします。

> **メモ:** `main` ブランチに直接 push しないこと。必ず自分のブランチを切ってからコミットしてください。

---

## 参考

- [Google AI Studio](https://aistudio.google.com/apikey) - Gemini API キー取得
- [FAISS](https://github.com/facebookresearch/faiss) - ベクトル検索ライブラリ
- [sentence-transformers](https://www.sbert.net/) - 埋め込みモデル

---

## 困ったら

エラーメッセージと「何をしようとして」「何が起きたか」をセットでチームに共有してください。