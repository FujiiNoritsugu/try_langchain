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
    embeddings = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large")

    # 旅行ツアー情報（詳細データ付き）
    travel_themes = [
        {
            "id": "tour_001",
            "text": "秋の美濃の滝見学ツアー",
            "name": "秋の美濃の滝見学ツアー",
            "location": "岐阜県美濃市",
            "season": "10月下旬～11月中旬",
            "description": "紅葉シーズンの美濃の滝を訪れる2日間のツアー。鮮やかな紅葉に囲まれた滝の絶景を楽しめます。ハイキングコース付き。",
            "highlights": ["紅葉の名所", "滝の絶景", "温泉宿泊"],
        },
        {
            "id": "tour_002",
            "text": "春の京都桜巡り",
            "name": "春の京都桜巡り",
            "location": "京都府京都市",
            "season": "3月下旬～4月上旬",
            "description": "京都の桜の名所を巡る3日間のツアー。哲学の道、嵐山、清水寺など人気スポットを訪問。",
            "highlights": ["桜の名所", "寺社仏閣", "京料理"],
        },
        {
            "id": "tour_003",
            "text": "夏の沖縄ビーチリゾート",
            "name": "夏の沖縄ビーチリゾート",
            "location": "沖縄県",
            "season": "7月～9月",
            "description": "透き通る海と白い砂浜を満喫する4日間のリゾートツアー。シュノーケリングやマリンスポーツも楽しめます。",
            "highlights": ["ビーチリゾート", "マリンスポーツ", "沖縄料理"],
        },
    ]

    # ベクトル化とデータポイントの準備
    datapoints = []
    for theme in travel_themes:
        vector = embeddings.embed_query(theme["text"])
        print(f"\nテーマ: {theme['text']}")
        print(f"ベクトル次元数: {len(vector)}")
        print(f"ベクトルの最初の5要素: {vector[:5]}")

        # Vector Search用のデータポイント形式（restrictsにJSON形式で詳細情報を保存）
        metadata = {
            "name": theme["name"],
            "location": theme["location"],
            "season": theme["season"],
            "description": theme["description"],
            "highlights": theme["highlights"],
        }

        datapoint = {
            "datapoint_id": theme["id"],
            "feature_vector": vector,
            "restricts": [
                {"namespace": "tour_data", "allow_list": [json.dumps(metadata, ensure_ascii=False)]}
            ],
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
    if hasattr(index, "metadata") and index.metadata:
        if hasattr(index.metadata, "config") and hasattr(
            index.metadata.config, "dimensions"
        ):
            print(f"次元数: {index.metadata.config.dimensions}")

    # インデックスの詳細情報を取得
    try:
        # indexオブジェクトのGCAリソースから情報を取得
        gca_resource = index.gca_resource
        if hasattr(gca_resource, "index_stats"):
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
    print('   python main.py search "クエリテキスト"')
    print("   ※エンドポイントのデプロイが必要です")

    print("\n3. 直接的なデータ読み取りについて:")
    print("   Vector Search APIでは、データポイントの直接読み取りは")
    print("   サポートされていません。検索クエリを通じてのみ")
    print("   データにアクセスできます。")

    print("\n補足情報:")
    print("   - データのアップロード後、インデックスの更新に数分〜数十分かかります")
    print("   - 更新が完了するまで、vectors_countは0と表示されることがあります")
    print(
        f"   - 現在のベクトル数: {gca_resource.index_stats.vectors_count if hasattr(gca_resource, 'index_stats') else 'N/A'}"
    )


def register_new_theme(theme_text: str, embeddings, index_id: str) -> str:
    """新しいテーマをVector Searchに登録する"""
    import uuid

    # 新しいIDを生成
    new_id = f"tour_{uuid.uuid4().hex[:8]}"

    # ベクトル化
    vector = embeddings.embed_query(theme_text)

    # インデックスの取得
    index = MatchingEngineIndex(index_name=index_id)

    # データポイントを作成（restrictsにJSON形式で詳細情報を保存）
    metadata = {
        "name": theme_text,
        "location": "未定",
        "season": "未定",
        "description": f"{theme_text}に関するツアー（自動登録）",
        "highlights": [],
    }

    datapoint = {
        "datapoint_id": new_id,
        "feature_vector": vector,
        "restricts": [{"namespace": "tour_data", "allow_list": [json.dumps(metadata, ensure_ascii=False)]}],
    }

    # アップロード
    print("\n新しいテーマをVector Searchに登録中...")
    index.upsert_datapoints(datapoints=[datapoint])
    print(f"登録完了: ID={new_id}, テーマ='{theme_text}'")

    return new_id


def import_tours_from_json(json_file_path: str):
    """
    JSONファイルからツアーデータを読み込んでVector Searchに登録

    Args:
        json_file_path: ツアーデータが記載されたJSONファイルのパス

    JSONファイルの形式:
    [
        {
            "id": "tour_xxx",
            "text": "ツアー名",
            "name": "ツアー名",
            "location": "場所",
            "season": "時期",
            "description": "説明",
            "highlights": ["見どころ1", "見どころ2"]
        },
        ...
    ]
    """
    # GCP設定
    project_id = os.getenv("GCP_PROJECT_ID")
    location = os.getenv("GCP_LOCATION", "us-central1")
    index_id = os.getenv("VECTOR_SEARCH_INDEX_ID")

    if not all([project_id, index_id]):
        raise ValueError("GCP_PROJECT_ID, VECTOR_SEARCH_INDEX_IDを設定してください")

    # JSONファイルを読み込み
    if not os.path.exists(json_file_path):
        raise FileNotFoundError(f"JSONファイルが見つかりません: {json_file_path}")

    with open(json_file_path, "r", encoding="utf-8") as f:
        travel_themes = json.load(f)

    print(f"読み込んだツアー数: {len(travel_themes)}")

    # Vertex AI初期化
    aiplatform.init(project=project_id, location=location)

    # 日本語対応のembeddingsモデルを使用
    embeddings = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large")

    # ベクトル化とデータポイントの準備
    datapoints = []
    for theme in travel_themes:
        # 必須フィールドのチェック
        required_fields = ["id", "text", "name", "location", "season", "description"]
        missing_fields = [field for field in required_fields if field not in theme]
        if missing_fields:
            print(f"警告: ツアー {theme.get('id', '???')} に必須フィールドが不足しています: {missing_fields}")
            continue

        vector = embeddings.embed_query(theme["text"])
        print(f"\nツアー: {theme['name']}")
        print(f"  ベクトル次元数: {len(vector)}")

        # Vector Search用のデータポイント形式（restrictsにJSON形式で詳細情報を保存）
        metadata = {
            "name": theme["name"],
            "location": theme["location"],
            "season": theme["season"],
            "description": theme["description"],
            "highlights": theme.get("highlights", []),
        }

        datapoint = {
            "datapoint_id": theme["id"],
            "feature_vector": vector,
            "restricts": [
                {"namespace": "tour_data", "allow_list": [json.dumps(metadata, ensure_ascii=False)]}
            ],
        }
        datapoints.append(datapoint)

    if not datapoints:
        print("エラー: 登録可能なツアーデータがありません")
        return

    # インデックスの取得
    index = MatchingEngineIndex(index_name=index_id)

    # データポイントをアップロード
    print(f"\n\nVector Searchにデータをアップロード中...")
    response = index.upsert_datapoints(datapoints=datapoints)
    print(f"アップロード完了: {len(datapoints)}件のベクトルを登録しました")
    print("\n※ インデックスの更新には数分かかる場合があります。")

    return response


def remove_datapoints(datapoint_ids: list[str]):
    """
    Vector Searchから指定されたデータポイントを削除する

    Args:
        datapoint_ids: 削除するデータポイントのIDリスト
    """
    # GCP設定
    project_id = os.getenv("GCP_PROJECT_ID")
    location = os.getenv("GCP_LOCATION", "us-central1")
    index_id = os.getenv("VECTOR_SEARCH_INDEX_ID")

    if not all([project_id, index_id]):
        print("警告: GCP_PROJECT_ID, VECTOR_SEARCH_INDEX_IDの環境変数が必要です")
        return

    # Vertex AI初期化
    aiplatform.init(project=project_id, location=location)

    # インデックスの取得
    index = MatchingEngineIndex(index_name=index_id)

    # データポイントを削除
    print(f"\n削除するデータポイント: {datapoint_ids}")
    print("Vector Searchからデータポイントを削除中...")

    index.remove_datapoints(datapoint_ids=datapoint_ids)

    print(f"削除完了: {len(datapoint_ids)}件のデータポイントを削除しました")
    print("\n※ インデックスの更新には数分かかる場合があります。")


def search_similar_vectors(
    query_text: str, neighbor_count: int = 3, similarity_threshold: float = 0.9
):
    """
    類似ベクトルを検索（エンドポイントが必要）

    Args:
        query_text: 検索クエリ
        neighbor_count: 取得する類似結果の数
        similarity_threshold: 類似度の閾値（これより小さい場合は新規登録）
    """
    # GCP設定
    project_id = os.getenv("GCP_PROJECT_ID")
    location = os.getenv("GCP_LOCATION", "us-central1")
    index_id = os.getenv("VECTOR_SEARCH_INDEX_ID")
    index_endpoint_id = os.getenv("VECTOR_SEARCH_ENDPOINT_ID")
    deployed_index_id = os.getenv("DEPLOYED_INDEX_ID")

    if not all([project_id, index_id, index_endpoint_id, deployed_index_id]):
        print(
            "警告: VECTOR_SEARCH_INDEX_ID, VECTOR_SEARCH_ENDPOINT_ID, DEPLOYED_INDEX_IDの環境変数が必要です"
        )
        return

    # 型チェック（静的解析ツール用）
    assert index_id is not None

    # Vertex AI初期化
    aiplatform.init(project=project_id, location=location)

    # Embeddings
    embeddings = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large")

    # クエリをベクトル化
    query_vector = embeddings.embed_query(query_text)

    # エンドポイントから検索（return_full_datapoint=Trueでrestrictsも取得）
    index_endpoint = MatchingEngineIndexEndpoint(index_endpoint_name=index_endpoint_id)

    response = index_endpoint.find_neighbors(
        deployed_index_id=deployed_index_id,
        queries=[query_vector],
        num_neighbors=neighbor_count,
        return_full_datapoint=True,
    )

    print(f"\n=== 類似検索結果: '{query_text}' ===")
    print(f"(類似度閾値: {similarity_threshold})")

    # 結果を表示
    neighbors = response[0]
    for idx, neighbor in enumerate(neighbors):
        # restrictsからツアー情報（JSON）を取得
        tour_data = None
        tour_name = "不明なツアー"
        if hasattr(neighbor, "restricts") and neighbor.restricts:
            for restrict in neighbor.restricts:
                # Namespaceオブジェクトの属性は'name'と'allow_tokens'
                if hasattr(restrict, "name") and restrict.name == "tour_data":
                    if hasattr(restrict, "allow_tokens") and restrict.allow_tokens:
                        try:
                            tour_data = json.loads(restrict.allow_tokens[0])
                            tour_name = tour_data.get("name", "不明なツアー")
                        except json.JSONDecodeError:
                            pass
                        break

        print(f"\n{idx + 1}. {tour_name}")
        print(f"   ツアーID: {neighbor.id}")
        print(f"   類似度スコア: {neighbor.distance:.4f}")
        if tour_data:
            print(f"   場所: {tour_data.get('location', 'N/A')}")
            print(f"   時期: {tour_data.get('season', 'N/A')}")
            print(f"   説明: {tour_data.get('description', 'N/A')}")

    # 最大類似度を確認（最初の結果が最も類似度が高い）
    if neighbors:
        max_similarity = neighbors[0].distance

        if max_similarity < similarity_threshold:
            print(f"\n{'='*50}")
            print(
                f"注意: 最大類似度 ({max_similarity:.4f}) が閾値 ({similarity_threshold}) を下回りました。"
            )
            print(f"類似するツアーが見つかりませんでした。")
            print(f"{'='*50}")

            # 新しいテーマとして登録
            new_id = register_new_theme(query_text, embeddings, index_id)
            print(f"\n新しいツアーテーマとして登録しました:")
            print(f"   テーマ名: {query_text}")
            print(f"   ツアーID: {new_id}")
            print(f"\n※ インデックスの更新には数分かかる場合があります。")
        else:
            print(
                f"\n類似するツアーが見つかりました（最大類似度: {max_similarity:.4f} >= 閾値: {similarity_threshold}）"
            )


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
        elif command == "remove":
            if len(sys.argv) < 3:
                print("エラー: 削除するデータポイントのIDを指定してください")
                print("使用例: python main.py remove tour_fc6d9538 tour_baf2e76e")
            else:
                ids_to_remove = sys.argv[2:]
                remove_datapoints(ids_to_remove)
        elif command == "import":
            if len(sys.argv) < 3:
                print("エラー: インポートするJSONファイルのパスを指定してください")
                print("使用例: python main.py import tours.json")
            else:
                json_path = sys.argv[2]
                import_tours_from_json(json_path)
        else:
            print("使用方法:")
            print("  python main.py upload        - ベクトルデータをアップロード")
            print("  python main.py read          - 格納されたデータを確認")
            print("  python main.py search [クエリ] - 類似検索")
            print("  python main.py remove [ID...] - データポイントを削除")
            print("  python main.py import [JSON]  - JSONファイルからツアーを一括登録")
    else:
        # デフォルトはアップロード
        upload_to_vector_search()
