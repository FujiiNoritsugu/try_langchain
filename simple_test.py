"""
簡易テストスクリプト - 既存のインデックスを使用してRAGをテスト
"""
import sys
import os

# 環境変数を直接設定（dotenvなしで動作）
if not os.getenv("ANTHROPIC_API_KEY"):
    print("警告: ANTHROPIC_API_KEYが設定されていません")
    print("export ANTHROPIC_API_KEY=your_key_here を実行してください")
    sys.exit(1)

def test_standard_faiss():
    """標準FAISS MultiVectorをテスト"""
    print("\n" + "="*60)
    print("標準FAISS MultiVectorのテスト")
    print("="*60)

    try:
        from rag_multivector_faiss import load_multi_vector_retriever, create_rag_chain

        # Retrieverを読み込み
        print("インデックスを読み込み中...")
        retriever = load_multi_vector_retriever()

        # RAGチェーンを作成
        print("RAGチェーンを作成中...")
        rag_chain = create_rag_chain(retriever)

        # クエリ実行
        query = sys.argv[1] if len(sys.argv) > 1 else "紅葉が見たいです"
        print(f"クエリ: {query}")

        print("\n検索中...")
        source_docs = retriever.invoke(query)

        print(f"✓ {len(source_docs)}件のドキュメントを検索しました")

        print("\n回答を生成中...")
        result = rag_chain.invoke(query)

        print("\n【回答】")
        print(result)

        print("\n【検索されたソース】")
        for idx, doc in enumerate(source_docs, 1):
            print(f"  {idx}. {doc.metadata.get('tour_name', '不明')}")

        return True

    except Exception as e:
        print(f"✗ エラー: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_raptor():
    """RAPTORをテスト"""
    print("\n" + "="*60)
    print("RAPTORのテスト")
    print("="*60)

    try:
        from rag_raptor import load_raptor_retriever, create_rag_chain

        # Retrieverを読み込み
        print("インデックスを読み込み中...")
        retriever = load_raptor_retriever()

        # RAGチェーンを作成
        print("RAGチェーンを作成中...")
        rag_chain = create_rag_chain(retriever)

        # クエリ実行
        query = sys.argv[1] if len(sys.argv) > 1 else "紅葉が見たいです"
        print(f"クエリ: {query}")

        print("\n検索中...")
        source_docs = retriever.invoke(query)

        print(f"✓ {len(source_docs)}件のドキュメントを検索しました")

        print("\n回答を生成中...")
        result = rag_chain.invoke(query)

        print("\n【回答】")
        print(result)

        print("\n【検索されたソース】")
        for idx, doc in enumerate(source_docs, 1):
            print(f"  {idx}. {doc.metadata.get('tour_name', '不明')}")

        return True

    except Exception as e:
        print(f"✗ エラー: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--raptor":
        query = sys.argv[2] if len(sys.argv) > 2 else "紅葉が見たいです"
        sys.argv = ["simple_test.py", query]  # クエリを引数として設定
        test_raptor()
    elif len(sys.argv) > 1 and sys.argv[1] == "--standard":
        query = sys.argv[2] if len(sys.argv) > 2 else "紅葉が見たいです"
        sys.argv = ["simple_test.py", query]
        test_standard_faiss()
    elif len(sys.argv) > 1 and sys.argv[1] == "--both":
        query = sys.argv[2] if len(sys.argv) > 2 else "紅葉が見たいです"
        sys.argv = ["simple_test.py", query]

        result1 = test_standard_faiss()
        result2 = test_raptor()

        print("\n" + "="*60)
        print("テスト結果")
        print("="*60)
        print(f"標準FAISS: {'✓ 成功' if result1 else '✗ 失敗'}")
        print(f"RAPTOR: {'✓ 成功' if result2 else '✗ 失敗'}")
    else:
        print("使用方法:")
        print("  python simple_test.py --standard 'クエリ'")
        print("  python simple_test.py --raptor 'クエリ'")
        print("  python simple_test.py --both 'クエリ'")
        print("\n例:")
        print("  python simple_test.py --both '紅葉が見たいです'")
