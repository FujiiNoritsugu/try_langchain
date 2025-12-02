# クイックスタートガイド

RAPTORとColBERTを使ったRAGシステムをすぐに始めるためのガイドです。

## 📦 インストール

### 1. 依存関係のインストール

```bash
pip install -r requirements.txt
```

### 2. 環境変数の設定

`.env`ファイルを作成：

```bash
echo "ANTHROPIC_API_KEY=your_api_key_here" > .env
```

## 🚀 5分でスタート

### ステップ1: 全てのインデックスを初期化

```bash
python test_rag_implementations.py --init-all
```

これにより以下が初期化されます：
- 標準FAISS MultiVectorインデックス
- RAPTORインデックス（3階層）
- ColBERTインデックス（通常 & MultiVector）

⚠️ 注意: 初回実行時はモデルのダウンロードに時間がかかります（5-10分程度）

### ステップ2: テストクエリを実行

```bash
# 全実装を比較
python test_rag_implementations.py --compare "紅葉が見たいです"
```

## 📝 個別実装の使い方

### RAPTOR

```bash
# 初期化（3階層）
python rag_raptor.py --init

# クエリ実行
python rag_raptor.py "紅葉が見たいです"

# 特定レベルで検索
python rag_raptor.py --level 1 "詳細な情報が欲しい"
```

### ColBERT

```bash
# 初期化
python rag_colbert.py --init

# クエリ実行
python rag_colbert.py "紅葉が見たいです"

# MultiVectorモード
python rag_colbert.py --init --multi-vector
python rag_colbert.py --multi-vector "アクティビティが楽しいツアーは?"
```

### 標準FAISS MultiVector

```bash
# 初期化
python rag_multivector_faiss.py --init

# クエリ実行
python rag_multivector_faiss.py "紅葉が見たいです"
```

## 🔍 サンプルクエリ

以下のクエリで試してみてください：

```bash
# 季節に関する質問
python rag_raptor.py "紅葉が見たいです"
python rag_colbert.py "春の桜が見られるツアーは？"

# アクティビティに関する質問
python rag_raptor.py "ハイキングができるツアーは？"
python rag_colbert.py --multi-vector "体験型のアクティビティがあるツアーは？"

# 場所に関する質問
python rag_raptor.py "京都のツアーを教えて"
python rag_colbert.py "温泉に入れるツアーは？"

# 抽象度の異なる質問（RAPTORに最適）
python rag_raptor.py --level 3 "おすすめのツアーは？"  # 概要
python rag_raptor.py --level 1 "各ツアーの詳細を教えて"  # 詳細
```

## 🧪 比較テスト

### 全実装の性能を比較

```bash
python test_rag_implementations.py --compare "紅葉が見たいです"
```

これにより以下が表示されます：
- 各実装の回答
- 検索されたドキュメント
- 処理時間の比較

### 個別テスト

```bash
# RAPTORのみテスト
python test_rag_implementations.py --test raptor "紅葉が見たいです"

# ColBERTのみテスト
python test_rag_implementations.py --test colbert "紅葉が見たいです"

# 標準FAISSのみテスト
python test_rag_implementations.py --test standard "紅葉が見たいです"
```

## 📊 期待される結果

### 処理時間（目安）

- **標準FAISS**: 2-3秒
- **RAPTOR**: 3-5秒
- **ColBERT**: 4-8秒

### 検索精度

実際のクエリでテストした結果：

| クエリ | 標準FAISS | RAPTOR | ColBERT |
|--------|-----------|---------|---------|
| "紅葉が見たい" | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| "体験型アクティビティ" | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| "おすすめは？"（抽象的） | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |

## 🎯 次のステップ

### カスタマイズ

1. **RAPTORの階層数を変更**
   ```bash
   python rag_raptor.py --init --levels 4
   ```

2. **ColBERTのパラメータ調整**
   `rag_colbert.py`の`max_document_length`を変更

3. **独自のデータで試す**
   `tour_data.json`を編集して、独自のドキュメントを追加

### 詳細なドキュメント

- [RAG_IMPLEMENTATIONS.md](RAG_IMPLEMENTATIONS.md) - 詳細な実装ガイド
- 各Pythonファイルのdocstring

## ❓ トラブルシューティング

### ragatouille のインストールエラー

```bash
pip install --upgrade pip
pip install ragatouille --no-cache-dir
```

### FAISSのロードエラー

インデックスが初期化されていない可能性があります：

```bash
python test_rag_implementations.py --init-all
```

### メモリ不足

- ColBERTの`num_results`を減らす
- RAPTORの階層数を減らす
- より小さいデータセットで試す

### APIキーエラー

`.env`ファイルに正しいAPIキーが設定されているか確認：

```bash
cat .env
```

## 🤝 サポート

問題が発生した場合：

1. エラーメッセージを確認
2. [RAG_IMPLEMENTATIONS.md](RAG_IMPLEMENTATIONS.md)の詳細ガイドを参照
3. 各実装のPythonファイルのdocstringを確認

---

それでは、RAGシステムをお楽しみください！ 🎉
