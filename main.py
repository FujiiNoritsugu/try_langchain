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


if __name__ == "__main__":
    upload_to_vector_search()
