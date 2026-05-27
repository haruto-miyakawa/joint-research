# プロジェクト処理フローチャート

このドキュメントは `joint-research` プロジェクトの主要な処理フローを図示したものです。
Mermaid 記法で書かれているため、GitHub 上でそのまま表示されます。

-----

## 1. 全体像:3つのコマンドの関係

```mermaid
flowchart TB
    Start([プロジェクト開始])

    Setup[セットアップ<br/>venv作成・pip install・.env設定]

    Ingest[python main.py ingest<br/>知識の取り込み]
    Chat[python main.py chat<br/>NPCと対話]
    Eval[python main.py evaluate<br/>一括評価]

    VS[(data/vector_store/<br/>FAISSインデックス)]
    Conv[(data/conversations/<br/>会話履歴)]
    Results[(eval_results/<br/>評価結果)]

    Start --> Setup
    Setup --> Ingest
    Ingest --> VS
    VS --> Chat
    VS --> Eval
    Chat --> Conv
    Eval --> Results

    style Ingest fill:#FFE4B5
    style Chat fill:#B5E4FF
    style Eval fill:#B5FFB5
```

-----

## 2. ingest コマンドの詳細フロー

```mermaid
flowchart TB
    Start([python main.py ingest])

    Read[data/lore.md を読み込み]
    Chunk[chunk_text&#40;&#41;<br/>句点優先で400文字ずつに分割<br/>50文字オーバーラップ]
    LoadModel[load_embedding_model&#40;&#41;<br/>multilingual-e5-small を<br/>HuggingFaceからロード]
    Embed["embed_texts(passage)<br/>各チャンクを384次元ベクトル化<br/>L2正規化"]
    Build[build_vector_store&#40;&#41;<br/>FAISS IndexFlatIP 構築]
    Save[save_vector_store&#40;&#41;<br/>index.faiss と meta.pkl を保存]

    End([完了])

    Lore[("data/lore.md")]
    Index[(data/vector_store/<br/>index.faiss)]
    Meta[(data/vector_store/<br/>meta.pkl)]

    Start --> Read
    Lore -.->|読み込み| Read
    Read --> Chunk
    Chunk --> LoadModel
    LoadModel --> Embed
    Embed --> Build
    Build --> Save
    Save -.->|保存| Index
    Save -.->|保存| Meta
    Save --> End

    style Lore fill:#FFE4B5
    style Index fill:#E4E4FF
    style Meta fill:#E4E4FF
```

-----

## 3. chat コマンドの詳細フロー

```mermaid
flowchart TB
    Start([python main.py chat<br/>--session test01])

    LoadCfg[load_npc_config&#40;&#41;<br/>NPC設定読み込み]
    LoadVS[load_vector_store&#40;&#41;<br/>FAISSインデックス読み込み]

    Loop{ユーザー入力待ち}

    CheckCmd{特殊コマンド?}
    Exit([終了])
    Reset[セッションファイル削除]
    Debug[デバッグモードON/OFF]

    Process[chat_with_npc&#40;&#41; 実行]

    Q["embed_texts(query)<br/>ユーザー発言をベクトル化"]
    Search[search_vector_store&#40;&#41;<br/>Top-K件の関連チャンク取得]
    History[load_recent_messages&#40;&#41;<br/>過去N件の会話履歴取得]
    BuildPrompt[build_system_prompt&#40;&#41;<br/>NPC設定+検索結果+履歴を統合]
    CallLLM[call_llm&#40;&#41;<br/>Gemini APIに送信]
    SaveMsg[save_message&#40;&#41;<br/>ユーザー発言と応答を保存]
    Display[応答を画面表示]

    NPCFile[("npc_config.yaml")]
    VSFile[("data/vector_store/")]
    ConvFile[(data/conversations/<br/>test01.jsonl)]

    Start --> LoadCfg
    NPCFile -.-> LoadCfg
    LoadCfg --> LoadVS
    VSFile -.-> LoadVS
    LoadVS --> Loop

    Loop --> CheckCmd
    CheckCmd -->|/exit| Exit
    CheckCmd -->|/reset| Reset
    CheckCmd -->|/debug| Debug
    CheckCmd -->|通常入力| Process

    Reset --> Loop
    Debug --> Loop

    Process --> Q
    Q --> Search
    Search --> History
    ConvFile -.-> History
    History --> BuildPrompt
    BuildPrompt --> CallLLM
    CallLLM --> SaveMsg
    SaveMsg -.->|追記| ConvFile
    SaveMsg --> Display
    Display --> Loop

    style Process fill:#B5E4FF
    style CallLLM fill:#FFB5B5
```

-----

## 4. evaluate コマンドの詳細フロー

```mermaid
flowchart TB
    Start([python main.py evaluate])

    LoadQ[eval_questions.yaml を読み込み]
    LoadCfg[NPC設定・ベクトルストアをロード]

    Loop{質問リストをループ}

    Process[chat_with_npc&#40;&#41; 実行<br/>※セッションは1問ごとに独立]
    Record[質問・検索チャンク・応答を記録]

    More{次の質問あり?}

    Format[Markdown形式に整形]
    Save[eval_YYYYMMDD_HHMMSS.md に保存]
    End([完了])

    QFile[("eval_questions.yaml")]
    Result[(eval_results/<br/>eval_*.md)]

    Start --> LoadQ
    QFile -.-> LoadQ
    LoadQ --> LoadCfg
    LoadCfg --> Loop

    Loop --> Process
    Process --> Record
    Record --> More
    More -->|Yes| Loop
    More -->|No| Format

    Format --> Save
    Save -.->|出力| Result
    Save --> End

    style Process fill:#B5FFB5
    style Result fill:#E4E4FF
```

-----

## 5. RAG の仕組み(コア処理)

```mermaid
flowchart LR
    User[ユーザー発言<br/>「魔法戦争について教えて」]

    Q[クエリベクトル化<br/>384次元]

    DB[(FAISSインデックス<br/>世界観チャンク多数)]

    Search[コサイン類似度検索<br/>Top-K取得]

    Context["関連チャンク<br/>1. 魔法戦争は約300年前...<br/>2. 戦争の発端は王の暗殺...<br/>3. ..."]

    Persona[NPCキャラ設定<br/>賢者マルクス]

    History[会話履歴]

    Prompt[統合プロンプト]

    LLM[(Gemini API)]

    Response[NPC応答<br/>「うむ、わしの知る限りでは...」]

    User --> Q
    Q --> Search
    DB --> Search
    Search --> Context

    Context --> Prompt
    Persona --> Prompt
    History --> Prompt

    Prompt --> LLM
    LLM --> Response

    style User fill:#FFE4B5
    style Response fill:#B5FFB5
    style LLM fill:#FFB5B5
    style DB fill:#E4E4FF
```

-----

## 6. ファイル間の関係図

```mermaid
flowchart TB
    subgraph Source[編集対象ファイル]
        Main[main.py<br/>プログラム本体]
        Config[npc_config.yaml<br/>キャラ設定]
        Eval[eval_questions.yaml<br/>評価質問]
        Lore[data/lore.md<br/>世界観テキスト]
        Env[.env<br/>APIキー]
    end

    subgraph Generated[自動生成ファイル]
        FAISS[(index.faiss)]
        MetaPkl[(meta.pkl)]
        JSONL[(*.jsonl<br/>会話履歴)]
        EvalMD[(eval_*.md<br/>評価結果)]
    end

    subgraph Commands[実行コマンド]
        CmdIngest[ingest]
        CmdChat[chat]
        CmdEval[evaluate]
    end

    Main --> CmdIngest
    Main --> CmdChat
    Main --> CmdEval

    Lore -->|入力| CmdIngest
    CmdIngest -->|生成| FAISS
    CmdIngest -->|生成| MetaPkl

    Config -->|入力| CmdChat
    FAISS -->|入力| CmdChat
    MetaPkl -->|入力| CmdChat
    Env -->|APIキー| CmdChat
    CmdChat -->|追記| JSONL

    Config -->|入力| CmdEval
    Eval -->|入力| CmdEval
    FAISS -->|入力| CmdEval
    MetaPkl -->|入力| CmdEval
    Env -->|APIキー| CmdEval
    CmdEval -->|生成| EvalMD

    style Source fill:#FFF4D4
    style Generated fill:#E4E4FF
    style Commands fill:#D4F4D4
```

-----

## 7. 研究の発展フロー(将来の拡張イメージ)

```mermaid
flowchart TB
    Current[現状<br/>人間 ↔ NPC の1対1対話]

    Phase1[Phase 1<br/>多様性指標の実装<br/>単調化現象の観測]

    Phase2[Phase 2<br/>複数LLM自動対話の実装<br/>多様性維持手法の提案]

    Phase3[Phase 3<br/>大規模実験<br/>論文化・ベストペーパー狙い]

    Future[Future Work<br/>世界観適合LLMの<br/>ファインチューニング]

    Current --> Phase1
    Phase1 --> Phase2
    Phase2 --> Phase3
    Phase3 --> Future

    Phase1 -.-> M1[新規モジュール:<br/>diversity_metrics.py]
    Phase2 -.-> M2[新規モジュール:<br/>multi_agent_dialogue.py]
    Phase3 -.-> M3[評価データセット<br/>論文・スライド]

    style Current fill:#E4E4FF
    style Phase1 fill:#B5FFB5
    style Phase2 fill:#FFE4B5
    style Phase3 fill:#FFB5B5
    style Future fill:#D4D4D4
```

-----

## 補足:Mermaid とは

このドキュメント内のフローチャートは **Mermaid** という記法で書かれています。
GitHub は標準で Mermaid をレンダリングするので、このファイルを GitHub にプッシュすれば自動的に図として表示されます。

ローカルで確認したい場合は、VS Code に「Markdown Preview Mermaid Support」拡張機能を入れると、プレビュー画面で図が見られます。

図を編集したい場合は [Mermaid Live Editor](https://mermaid.live/) でビジュアルに編集できます。