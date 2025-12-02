"""
RAPTOR、ColBERT、標準RAGの比較テストスクリプト

各実装の検索精度と応答を比較します。

使用方法:
1. 各インデックスを初期化:
   python test_rag_implementations.py --init-all

2. 比較テストを実行:
   python test_rag_implementations.py --compare "紅葉が見たいです"

3. 個別のテストを実行:
   python test_rag_implementations.py --test raptor "紅葉が見たいです"
   python test_rag_implementations.py --test colbert "紅葉が見たいです"
   python test_rag_implementations.py --test standard "紅葉が見たいです"
"""

import sys
import os
import time
from typing import List, Dict
from dotenv import load_dotenv

load_dotenv()


def test_raptor(query: str) -> Dict:
    """RAPTOR実装をテスト"""
    from rag_raptor import load_raptor_retriever, create_rag_chain

    print(f"\n{'='*60}")
    print("RAPTOR実装のテスト")
    print(f"{'='*60}")

    start_time = time.time()

    try:
        # Retrieverを読み込み
        retriever = load_raptor_retriever()

        # RAGチェーンを作成
        rag_chain = create_rag_chain(retriever)

        # 検索とクエリ実行
        source_docs = retriever.invoke(query)
        result = rag_chain.invoke(query)

        elapsed_time = time.time() - start_time

        return {
            "method": "RAPTOR",
            "success": True,
            "result": result,
            "source_docs": source_docs,
            "num_docs": len(source_docs),
            "elapsed_time": elapsed_time,
        }

    except Exception as e:
        elapsed_time = time.time() - start_time
        return {
            "method": "RAPTOR",
            "success": False,
            "error": str(e),
            "elapsed_time": elapsed_time,
        }


def test_colbert(query: str, use_multi_vector: bool = False) -> Dict:
    """ColBERT実装をテスト"""
    from rag_colbert import load_colbert_retriever, load_multi_vector_colbert_retriever, create_rag_chain

    print(f"\n{'='*60}")
    print(f"ColBERT実装のテスト ({'MultiVector' if use_multi_vector else '通常'})")
    print(f"{'='*60}")

    start_time = time.time()

    try:
        # Retrieverを読み込み
        if use_multi_vector:
            retriever = load_multi_vector_colbert_retriever("tour_colbert_index_mv")
        else:
            retriever = load_colbert_retriever()

        # RAGチェーンを作成
        rag_chain = create_rag_chain(retriever)

        # 検索とクエリ実行
        source_docs = retriever.invoke(query)
        result = rag_chain.invoke(query)

        elapsed_time = time.time() - start_time

        return {
            "method": f"ColBERT {'MultiVector' if use_multi_vector else ''}",
            "success": True,
            "result": result,
            "source_docs": source_docs,
            "num_docs": len(source_docs),
            "elapsed_time": elapsed_time,
        }

    except Exception as e:
        elapsed_time = time.time() - start_time
        return {
            "method": f"ColBERT {'MultiVector' if use_multi_vector else ''}",
            "success": False,
            "error": str(e),
            "elapsed_time": elapsed_time,
        }


def test_standard(query: str) -> Dict:
    """標準FAISS MultiVector実装をテスト"""
    from rag_multivector_faiss import load_multi_vector_retriever, create_rag_chain

    print(f"\n{'='*60}")
    print("標準FAISS MultiVector実装のテスト")
    print(f"{'='*60}")

    start_time = time.time()

    try:
        # Retrieverを読み込み
        retriever = load_multi_vector_retriever()

        # RAGチェーンを作成
        rag_chain = create_rag_chain(retriever)

        # 検索とクエリ実行
        source_docs = retriever.invoke(query)
        result = rag_chain.invoke(query)

        elapsed_time = time.time() - start_time

        return {
            "method": "Standard FAISS MultiVector",
            "success": True,
            "result": result,
            "source_docs": source_docs,
            "num_docs": len(source_docs),
            "elapsed_time": elapsed_time,
        }

    except Exception as e:
        elapsed_time = time.time() - start_time
        return {
            "method": "Standard FAISS MultiVector",
            "success": False,
            "error": str(e),
            "elapsed_time": elapsed_time,
        }


def print_test_result(result: Dict):
    """テスト結果を表示"""
    print(f"\n{'='*60}")
    print(f"実装: {result['method']}")
    print(f"{'='*60}")

    if result['success']:
        print(f"✓ 成功")
        print(f"処理時間: {result['elapsed_time']:.2f}秒")
        print(f"検索されたドキュメント数: {result['num_docs']}")

        print(f"\n【回答】")
        print(result['result'])

        print(f"\n【検索されたソース】")
        for idx, doc in enumerate(result['source_docs'], 1):
            tour_name = doc.metadata.get('tour_name', '不明')
            print(f"  {idx}. {tour_name}")

    else:
        print(f"✗ 失敗")
        print(f"エラー: {result['error']}")
        print(f"処理時間: {result['elapsed_time']:.2f}秒")


def compare_all(query: str):
    """全実装を比較"""
    print(f"\n{'='*60}")
    print(f"質問: {query}")
    print(f"{'='*60}")

    # 各実装をテスト
    results = []

    # 1. 標準FAISS MultiVector
    results.append(test_standard(query))

    # 2. RAPTOR
    results.append(test_raptor(query))

    # 3. ColBERT (通常)
    results.append(test_colbert(query, use_multi_vector=False))

    # 4. ColBERT (MultiVector)
    results.append(test_colbert(query, use_multi_vector=True))

    # 結果を表示
    print(f"\n\n{'='*60}")
    print("比較結果サマリー")
    print(f"{'='*60}\n")

    for result in results:
        print_test_result(result)

    # 処理時間の比較
    print(f"\n{'='*60}")
    print("処理時間の比較")
    print(f"{'='*60}")

    successful_results = [r for r in results if r['success']]
    if successful_results:
        sorted_results = sorted(successful_results, key=lambda x: x['elapsed_time'])

        for idx, result in enumerate(sorted_results, 1):
            print(f"{idx}. {result['method']}: {result['elapsed_time']:.2f}秒")


def initialize_all():
    """全てのインデックスを初期化"""
    print("全てのインデックスを初期化します...\n")

    # 1. 標準FAISS MultiVector
    print("1. 標準FAISS MultiVectorインデックスを初期化中...")
    try:
        from rag_multivector_faiss import initialize_multi_vector_retriever
        initialize_multi_vector_retriever(mode="summary")
        print("✓ 完了\n")
    except Exception as e:
        print(f"✗ エラー: {e}\n")

    # 2. RAPTOR
    print("2. RAPTORインデックスを初期化中...")
    try:
        from rag_raptor import initialize_raptor_retriever
        initialize_raptor_retriever(num_levels=3)
        print("✓ 完了\n")
    except Exception as e:
        print(f"✗ エラー: {e}\n")

    # 3. ColBERT (通常)
    print("3. ColBERTインデックス（通常モード）を初期化中...")
    try:
        from rag_colbert import initialize_colbert_index
        initialize_colbert_index(index_name="tour_colbert_index", use_multi_vector=False)
        print("✓ 完了\n")
    except Exception as e:
        print(f"✗ エラー: {e}\n")

    # 4. ColBERT (MultiVector)
    print("4. ColBERTインデックス（MultiVectorモード）を初期化中...")
    try:
        from rag_colbert import initialize_colbert_index
        initialize_colbert_index(index_name="tour_colbert_index_mv", use_multi_vector=True)
        print("✓ 完了\n")
    except Exception as e:
        print(f"✗ エラー: {e}\n")

    print("全ての初期化が完了しました！")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--init-all":
        # 全てのインデックスを初期化
        initialize_all()

    elif len(sys.argv) > 1 and sys.argv[1] == "--compare":
        # 比較テスト
        query = sys.argv[2] if len(sys.argv) > 2 else "紅葉が見たいです"
        compare_all(query)

    elif len(sys.argv) > 1 and sys.argv[1] == "--test":
        # 個別テスト
        method = sys.argv[2] if len(sys.argv) > 2 else "raptor"
        query = sys.argv[3] if len(sys.argv) > 3 else "紅葉が見たいです"

        if method == "raptor":
            result = test_raptor(query)
        elif method == "colbert":
            result = test_colbert(query, use_multi_vector=False)
        elif method == "colbert-mv":
            result = test_colbert(query, use_multi_vector=True)
        elif method == "standard":
            result = test_standard(query)
        else:
            print(f"不明なメソッド: {method}")
            print("利用可能なメソッド: raptor, colbert, colbert-mv, standard")
            sys.exit(1)

        print_test_result(result)

    else:
        # ヘルプを表示
        print(__doc__)
