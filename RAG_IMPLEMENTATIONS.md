# RAG実装ガイド：RAPTOR & ColBERT

このプロジェクトには、以下の3つのRAG実装が含まれています：

1. **標準FAISS MultiVector** (`rag_multivector_faiss.py`)
2. **RAPTOR（階層的要約）** (`rag_raptor.py`)
3. **ColBERT（トークンレベル検索）** (`rag_colbert.py`)

## 📋 目次

- [実装の比較](#実装の比較)
- [環境セットアップ](#環境セットアップ)
- [各実装の使い方](#各実装の使い方)
- [パフォーマンス比較](#パフォーマンス比較)
- [使い分けのガイドライン](#使い分けのガイドライン)

---

## 実装の比較

| 実装 | 特徴 | 長所 | 短所 | 適用ケース |
|------|------|------|------|-----------|
| **標準FAISS MultiVector** | 要約で検索、親ドキュメントを返す | シンプル、高速 | 単一レベルの要約のみ | 一般的なRAG |
| **RAPTOR** | 階層的要約（L1, L2, L3） | 抽象度が異なるクエリに対応 | 初期化に時間がかかる | 多様な粒度の質問 |
| **ColBERT** | トークンレベルのMaxSim検索 | 高精度、意味的理解が優れる | インデックスサイズが大きい | 精度重視の検索 |

---

## 環境セットアップ

### 1. 必要なパッケージのインストール

```bash
# 基本パッケージ
pip install langchain langchain-anthropic langchain-huggingface
pip install faiss-cpu  # GPUを使う場合は faiss-gpu
pip install python-dotenv

# ColBERTのための追加パッケージ
pip install ragatouille
```

### 2. 環境変数の設定

`.env`ファイルを作成し、以下を設定：

```env
ANTHROPIC_API_KEY=your_api_key_here
```

### 3. ツアーデータの準備

`tour_data.json`が存在することを確認してください。

---

## 各実装の使い方

### 1. 標準FAISS MultiVector RAG

#### 初期化

```bash
# 要約モードで初期化
python rag_multivector_faiss.py --init

# 仮想質問モードで初期化
python rag_multivector_faiss.py --init --mode hypothetical
```

#### クエリ実行

```bash
# 基本クエリ
python rag_multivector_faiss.py "紅葉が見たいです"

# 仮想質問モードでクエリ
python rag_multivector_faiss.py --mode hypothetical "アクティビティが楽しいツアーは?"
```

#### 特徴
- 要約または仮想質問で検索
- 親ドキュメント（完全な情報）を返す
- シンプルで高速

---

### 2. RAPTOR（階層的要約）RAG

#### 初期化

```bash
# 3階層で初期化（デフォルト）
python rag_raptor.py --init

# カスタム階層数で初期化
python rag_raptor.py --init --levels 4
```

#### クエリ実行

```bash
# 全レベルを検索
python rag_raptor.py "紅葉が見たいです"

# 特定レベルのみ検索
python rag_raptor.py --level 1 "詳細な情報が欲しい"
python rag_raptor.py --level 3 "概要だけ知りたい"
```

#### 階層構造

```
レベル0: 親ドキュメント（最も詳細）
  ├─ レベル1: 第1階層の要約（中程度の詳細）
  ├─ レベル2: 第2階層の要約（簡潔）
  └─ レベル3: 第3階層の要約（最も抽象的）
```

#### 特徴
- 複数の抽象度レベルから検索
- クエリの粒度に応じて最適なレベルを選択可能
- 多様な質問タイプに対応

---

### 3. ColBERT RAG

#### 初期化

```bash
# 通常モードで初期化
python rag_colbert.py --init

# MultiVectorモードで初期化（要約で検索、親を返す）
python rag_colbert.py --init --multi-vector
```

#### クエリ実行

```bash
# 通常モード
python rag_colbert.py "紅葉が見たいです"

# MultiVectorモード
python rag_colbert.py --multi-vector "アクティビティが楽しいツアーは?"
```

#### 特徴
- トークンレベルのMaxSim計算
- 密ベクトルより高精度な検索
- 意味的理解が優れる

---

## パフォーマンス比較

### テストスクリプトの使用

```bash
# 全てのインデックスを初期化
python test_rag_implementations.py --init-all

# 全実装を比較
python test_rag_implementations.py --compare "紅葉が見たいです"

# 個別テスト
python test_rag_implementations.py --test raptor "紅葉が見たいです"
python test_rag_implementations.py --test colbert "紅葉が見たいです"
python test_rag_implementations.py --test standard "紅葉が見たいです"
```

### 比較観点

1. **処理速度**
   - 標準FAISS: 最速
   - RAPTOR: 中程度（複数レベル検索のため）
   - ColBERT: やや遅い（MaxSim計算のため）

2. **検索精度**
   - ColBERT: 最高（トークンレベルのマッチング）
   - RAPTOR: 高い（階層的検索）
   - 標準FAISS: 良好

3. **インデックスサイズ**
   - 標準FAISS: 小
   - RAPTOR: 中（階層数×ドキュメント数）
   - ColBERT: 大（トークンごとの表現）

---

## 使い分けのガイドライン

### 標準FAISS MultiVectorを使うべき場合

- ✅ シンプルなRAGが必要
- ✅ 高速な応答が重要
- ✅ ストレージ容量が限られている
- ✅ 一般的な検索精度で十分

### RAPTORを使うべき場合

- ✅ 質問の粒度が多様
- ✅ 概要から詳細まで柔軟に対応したい
- ✅ 抽象的な質問と具体的な質問の両方がある
- ✅ ドキュメントが長文で階層的な理解が必要

### ColBERTを使うべき場合

- ✅ 検索精度が最優先
- ✅ 意味的な理解が重要
- ✅ 密ベクトルでは不十分な複雑なクエリ
- ✅ ストレージとレイテンシーは許容範囲

---

## 実装の詳細

### 標準FAISS MultiVector

**ファイル**: `rag_multivector_faiss.py`

**アーキテクチャ**:
```
ツアーデータ
  ├─ 親ドキュメント → docstore
  └─ 要約/仮想質問 → FAISS
```

**検索フロー**:
1. クエリ → FAISSで要約を検索
2. 要約のdoc_idを取得
3. docstoreから親ドキュメントを取得

---

### RAPTOR

**ファイル**: `rag_raptor.py`

**アーキテクチャ**:
```
ツアーデータ
  ├─ レベル0（親） → docstore
  ├─ レベル1要約 → FAISS_L1
  ├─ レベル2要約 → FAISS_L2
  └─ レベル3要約 → FAISS_L3
```

**検索フロー**:
1. クエリ → 各レベルのFAISSで検索
2. 全レベルの結果をマージ
3. doc_idで親ドキュメントを取得

---

### ColBERT

**ファイル**: `rag_colbert.py`

**アーキテクチャ**:
```
通常モード:
  ツアーデータ → ColBERTインデックス

MultiVectorモード:
  ツアーデータ
    ├─ 親ドキュメント → docstore
    └─ 要約 → ColBERTインデックス
```

**検索フロー**:
1. クエリ → ColBERTでMaxSim検索
2. (MultiVectorの場合) doc_idで親を取得
3. 結果を返す

---

## トラブルシューティング

### ragatouille のインストールエラー

```bash
# システムの依存関係を確認
pip install --upgrade pip
pip install ragatouille --no-cache-dir
```

### FAISSのロードエラー

```bash
# 先に --init で初期化してください
python rag_multivector_faiss.py --init
python rag_raptor.py --init
python rag_colbert.py --init
```

### メモリ不足エラー

- ColBERTの`max_document_length`を小さくする
- RAPTORの階層数を減らす
- バッチサイズを調整する

---

## 参考文献

- **RAPTOR**: [Recursive Abstractive Processing for Tree-Organized Retrieval](https://arxiv.org/abs/2401.18059)
- **ColBERT**: [Contextualized Late Interaction over BERT](https://arxiv.org/abs/2004.12832)
- **RAGatouille**: [GitHub Repository](https://github.com/bclavie/RAGatouille)

---

## ライセンス

このプロジェクトは、LangChainおよび関連ライブラリのライセンスに従います。
