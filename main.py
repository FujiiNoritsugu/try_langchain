from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()


def vectorize_travel_theme():
    # 日本語対応のembeddingsモデルを使用
    embeddings = HuggingFaceEmbeddings(
        model_name="intfloat/multilingual-e5-large"
    )

    # 旅行テーマ名
    travel_themes = [
        "秋の美濃の滝見学ツアー",
        "春の京都桜巡り",
        "夏の沖縄ビーチリゾート"
    ]

    # テキストをベクトル化
    for theme in travel_themes:
        vector = embeddings.embed_query(theme)
        print(f"\nテーマ: {theme}")
        print(f"ベクトル次元数: {len(vector)}")
        print(f"ベクトルの最初の5要素: {vector[:5]}")


if __name__ == "__main__":
    vectorize_travel_theme()
