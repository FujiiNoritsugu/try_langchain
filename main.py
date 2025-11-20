import os
from google.cloud import aiplatform
from google.cloud.aiplatform import MatchingEngineIndex, MatchingEngineIndexEndpoint
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
import json

load_dotenv()


def upload_to_vector_search():
    # GCP設定（環境変数から取得）
    project_id = os.getenv("GCP_PROJECT_ID")
    location = os.getenv("GCP_LOCATION", "us-central1")
    index_id = os.getenv("VECTOR_SEARCH_INDEX_ID")

    if not project_id:
        raise ValueError("GCP_PROJECT_ID環境変数を設定してください")
    if not index_id:
        raise ValueError("VECTOR_SEARCH_INDEX_ID環境変数を設定してください")

    # Vertex AI初期化
    aiplatform.init(project=project_id, location=location)

    # 日本語対応のembeddingsモデルを使用
    embeddings = HuggingFaceEmbeddings(
        model_name="intfloat/multilingual-e5-large"
    )

    # 旅行テーマ名
    travel_themes = [
        {"id": "tour_001", "text": "秋の美濃の滝見学ツアー"},
        {"id": "tour_002", "text": "春の京都桜巡り"},
        {"id": "tour_003", "text": "夏の沖縄ビーチリゾート"}
    ]

    # ベクトル化とデータポイントの準備
    datapoints = []
    for theme in travel_themes:
        vector = embeddings.embed_query(theme["text"])
        print(f"\nテーマ: {theme['text']}")
        print(f"ベクトル次元数: {len(vector)}")
        print(f"ベクトルの最初の5要素: {vector[:5]}")

        # Vector Search用のデータポイント形式
        datapoint = {
            "datapoint_id": theme["id"],
            "feature_vector": vector,
        }
        datapoints.append(datapoint)

    # インデックスの取得
    index = MatchingEngineIndex(index_name=index_id)

    # データポイントをアップロード
    print("\n\nVector Searchにデータをアップロード中...")
    response = index.upsert_datapoints(datapoints=datapoints)
    print(f"アップロード完了: {len(datapoints)}件のベクトルを登録しました")

    return response


def read_vector_data():
    """Vector Searchに格納されたベクトルデータを確認"""
    # GCP設定（環境変数から取得）
    project_id = os.getenv("GCP_PROJECT_ID")
    location = os.getenv("GCP_LOCATION", "us-central1")
    index_id = os.getenv("VECTOR_SEARCH_INDEX_ID")

    if not project_id:
        raise ValueError("GCP_PROJECT_ID環境変数を設定してください")
    if not index_id:
        raise ValueError("VECTOR_SEARCH_INDEX_ID環境変数を設定してください")

    # Vertex AI初期化
    aiplatform.init(project=project_id, location=location)

    # インデックスの取得
    index = MatchingEngineIndex(index_name=index_id)

    # インデックスの基本情報を表示
    print("=== インデックス情報 ===")
    print(f"インデックス名: {index.display_name}")
    print(f"インデックスID: {index.resource_name}")
    print(f"説明: {index.description}")

    # 利用可能な属性を確認
    if hasattr(index, 'metadata') and index.metadata:
        if hasattr(index.metadata, 'config') and hasattr(index.metadata.config, 'dimensions'):
            print(f"次元数: {index.metadata.config.dimensions}")

    # インデックスの詳細情報を取得
    try:
        # indexオブジェクトのGCAリソースから情報を取得
        gca_resource = index.gca_resource
        if hasattr(gca_resource, 'index_stats'):
            print(f"登録されているベクトル数: {gca_resource.index_stats.vectors_count}")
    except Exception as e:
        print(f"統計情報の取得エラー: {e}")

    # 特定のデータポイントを読み取る
    print("\n=== データポイントの読み取り ===")
    datapoint_ids = ["tour_001", "tour_002", "tour_003"]

    # Vector Searchでのデータ確認に関する情報
    print("\nVector Searchに格納されたデータを確認する方法:")
    print("\n1. GCPコンソールで確認:")
    print("   https://console.cloud.google.com/vertex-ai/matching-engine/indexes")
    print(f"   プロジェクト: {project_id}")
    print(f"   ロケーション: {location}")
    print(f"   インデックス: {index.display_name}")

    print("\n2. エンドポイントをデプロイして検索で確認:")
    print("   python main.py search \"クエリテキスト\"")
    print("   ※エンドポイントのデプロイが必要です")

    print("\n3. 直接的なデータ読み取りについて:")
    print("   Vector Search APIでは、データポイントの直接読み取りは")
    print("   サポートされていません。検索クエリを通じてのみ")
    print("   データにアクセスできます。")

    print("\n補足情報:")
    print("   - データのアップロード後、インデックスの更新に数分〜数十分かかります")
    print("   - 更新が完了するまで、vectors_countは0と表示されることがあります")
    print(f"   - 現在のベクトル数: {gca_resource.index_stats.vectors_count if hasattr(gca_resource, 'index_stats') else 'N/A'}")


def search_similar_vectors(query_text: str, neighbor_count: int = 3):
    """類似ベクトルを検索（エンドポイントが必要）"""
    # GCP設定
    project_id = os.getenv("GCP_PROJECT_ID")
    location = os.getenv("GCP_LOCATION", "us-central1")
    index_endpoint_id = os.getenv("VECTOR_SEARCH_ENDPOINT_ID")
    deployed_index_id = os.getenv("DEPLOYED_INDEX_ID")

    if not all([project_id, index_endpoint_id, deployed_index_id]):
        print("警告: VECTOR_SEARCH_ENDPOINT_IDとDEPLOYED_INDEX_IDの環境変数が必要です")
        return

    # Vertex AI初期化
    aiplatform.init(project=project_id, location=location)

    # Embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="intfloat/multilingual-e5-large"
    )

    # クエリをベクトル化
    query_vector = embeddings.embed_query(query_text)

    # エンドポイントから検索
    index_endpoint = MatchingEngineIndexEndpoint(index_endpoint_name=index_endpoint_id)

    response = index_endpoint.find_neighbors(
        deployed_index_id=deployed_index_id,
        queries=[query_vector],
        num_neighbors=neighbor_count
    )

    print(f"\n=== 類似検索結果: '{query_text}' ===")
    for idx, neighbor in enumerate(response[0]):
        print(f"\n{idx + 1}. ID: {neighbor.id}")
        print(f"   距離: {neighbor.distance}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        command = sys.argv[1]

        if command == "upload":
            upload_to_vector_search()
        elif command == "read":
            read_vector_data()
        elif command == "search":
            query = sys.argv[2] if len(sys.argv) > 2 else "紅葉の名所"
            search_similar_vectors(query)
        else:
            print("使用方法:")
            print("  python main.py upload   - ベクトルデータをアップロード")
            print("  python main.py read     - 格納されたデータを確認")
            print("  python main.py search [クエリ] - 類似検索")
    else:
        # デフォルトはアップロード
        upload_to_vector_search()
