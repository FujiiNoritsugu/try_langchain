"""
LangGraphを使用してデータベースに登録されたテーマ名と類似しないテーマ名を生成するプログラム
Anthropic Claude APIを使用
"""
import sqlite3
import uuid
from typing import TypedDict, List, Annotated
import operator
import os
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_voyageai import VoyageAIEmbeddings
from langgraph.graph import StateGraph, END
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# .envファイルから環境変数を読み込む
load_dotenv()


# データベースの初期化
def init_database(db_path: str = "themes.db"):
    """SQLiteデータベースを初期化"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS themes (
            id TEXT PRIMARY KEY,
            theme_name TEXT NOT NULL
        )
    """)

    conn.commit()
    conn.close()


def add_theme_to_db(theme_name: str, db_path: str = "themes.db"):
    """テーマをデータベースに追加"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    theme_id = str(uuid.uuid4())
    cursor.execute("INSERT INTO themes (id, theme_name) VALUES (?, ?)", (theme_id, theme_name))

    conn.commit()
    conn.close()
    return theme_id


def get_all_themes(db_path: str = "themes.db") -> List[str]:
    """データベースから全てのテーマ名を取得"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("SELECT theme_name FROM themes")
    themes = [row[0] for row in cursor.fetchall()]

    conn.close()
    return themes


# LangGraphの状態定義
class ThemeGenerationState(TypedDict):
    """テーマ生成プロセスの状態"""
    existing_themes: List[str]  # 既存のテーマ一覧
    candidate_theme: str  # 生成候補のテーマ
    is_unique: bool  # ユニークかどうか
    attempt_count: int  # 試行回数
    max_attempts: int  # 最大試行回数
    similarity_threshold: float  # 類似度の閾値
    max_similarity: float  # 既存テーマとの最大類似度
    db_path: str  # データベースパス
    category: str  # テーマのカテゴリ（ユーザー指定）


# ノード関数の定義
def fetch_existing_themes(state: ThemeGenerationState) -> ThemeGenerationState:
    """既存のテーマをデータベースから取得"""
    themes = get_all_themes(state["db_path"])
    state["existing_themes"] = themes
    print(f"📚 既存テーマ数: {len(themes)}")
    return state


def generate_theme(state: ThemeGenerationState) -> ThemeGenerationState:
    """LLMを使って新しいテーマを生成"""
    state["attempt_count"] += 1
    print(f"\n🎲 テーマ生成試行: {state['attempt_count']}/{state['max_attempts']}")

    llm = ChatAnthropic(model="claude-3-haiku-20240307", temperature=0.9)

    existing_themes_str = "\n".join([f"- {theme}" for theme in state["existing_themes"]])

    prompt = f"""以下の既存のテーマとは異なる、ユニークで創造的なテーマを1つ生成してください。

カテゴリ: {state['category']}

既存のテーマ:
{existing_themes_str if existing_themes_str else "（まだテーマは登録されていません）"}

要件:
- 既存のテーマと内容や言葉遣いが大きく異なること
- 簡潔で魅力的なテーマ名であること
- テーマ名のみを出力すること（説明は不要）
"""

    response = llm.invoke(prompt)
    candidate = response.content.strip()
    state["candidate_theme"] = candidate
    print(f"💡 生成されたテーマ: {candidate}")

    return state


def check_similarity(state: ThemeGenerationState) -> ThemeGenerationState:
    """既存テーマとの類似度をチェック"""
    if not state["existing_themes"]:
        # 既存テーマがない場合はユニーク
        state["is_unique"] = True
        state["max_similarity"] = 0.0
        print("✅ 既存テーマがないため、ユニークと判定")
        return state

    embeddings = VoyageAIEmbeddings(model="voyage-3-lite")

    # 候補テーマと既存テーマの埋め込みを取得
    candidate_embedding = embeddings.embed_query(state["candidate_theme"])
    existing_embeddings = embeddings.embed_documents(state["existing_themes"])

    # コサイン類似度を計算
    candidate_vector = np.array(candidate_embedding).reshape(1, -1)
    existing_vectors = np.array(existing_embeddings)

    similarities = cosine_similarity(candidate_vector, existing_vectors)[0]
    max_similarity = float(np.max(similarities))

    state["max_similarity"] = max_similarity
    state["is_unique"] = max_similarity < state["similarity_threshold"]

    print(f"📊 最大類似度: {max_similarity:.4f} (閾値: {state['similarity_threshold']})")

    if state["is_unique"]:
        print("✅ ユニークなテーマと判定")
    else:
        most_similar_idx = int(np.argmax(similarities))
        print(f"⚠️  類似テーマ検出: '{state['existing_themes'][most_similar_idx]}' (類似度: {max_similarity:.4f})")

    return state


def should_regenerate(state: ThemeGenerationState) -> str:
    """再生成が必要かどうかを判定"""
    if state["is_unique"]:
        return "unique"
    elif state["attempt_count"] >= state["max_attempts"]:
        print(f"⚠️  最大試行回数 ({state['max_attempts']}) に達しました")
        return "max_attempts"
    else:
        return "regenerate"


def finalize(state: ThemeGenerationState) -> ThemeGenerationState:
    """最終処理"""
    if state["is_unique"]:
        print(f"\n🎉 ユニークなテーマが生成されました: '{state['candidate_theme']}'")
    else:
        print(f"\n⚠️  完全にユニークなテーマは生成できませんでしたが、最善の候補: '{state['candidate_theme']}'")
    return state


# LangGraphの構築
def create_theme_generator_graph():
    """テーマ生成グラフを作成"""
    workflow = StateGraph(ThemeGenerationState)

    # ノードの追加
    workflow.add_node("fetch_themes", fetch_existing_themes)
    workflow.add_node("generate", generate_theme)
    workflow.add_node("check_similarity", check_similarity)
    workflow.add_node("finalize", finalize)

    # エッジの定義
    workflow.set_entry_point("fetch_themes")
    workflow.add_edge("fetch_themes", "generate")
    workflow.add_edge("generate", "check_similarity")

    # 条件付きエッジ
    workflow.add_conditional_edges(
        "check_similarity",
        should_regenerate,
        {
            "unique": "finalize",
            "max_attempts": "finalize",
            "regenerate": "generate"
        }
    )

    workflow.add_edge("finalize", END)

    return workflow.compile()


def generate_unique_theme(
    category: str = "一般",
    similarity_threshold: float = 0.7,
    max_attempts: int = 5,
    db_path: str = "themes.db",
    save_to_db: bool = False
) -> dict:
    """
    ユニークなテーマを生成

    Args:
        category: テーマのカテゴリ
        similarity_threshold: 類似度の閾値（この値未満ならユニーク）
        max_attempts: 最大試行回数
        db_path: データベースのパス
        save_to_db: 生成したテーマをデータベースに保存するか

    Returns:
        生成結果の辞書
    """
    # データベースの初期化
    init_database(db_path)

    # グラフの作成
    app = create_theme_generator_graph()

    # 初期状態
    initial_state = {
        "existing_themes": [],
        "candidate_theme": "",
        "is_unique": False,
        "attempt_count": 0,
        "max_attempts": max_attempts,
        "similarity_threshold": similarity_threshold,
        "max_similarity": 0.0,
        "db_path": db_path,
        "category": category
    }

    # グラフの実行
    result = app.invoke(initial_state)

    # データベースに保存
    if save_to_db and result["candidate_theme"]:
        theme_id = add_theme_to_db(result["candidate_theme"], db_path)
        print(f"\n💾 データベースに保存されました (ID: {theme_id})")

    return {
        "theme": result["candidate_theme"],
        "is_unique": result["is_unique"],
        "max_similarity": result["max_similarity"],
        "attempts": result["attempt_count"]
    }


if __name__ == "__main__":
    # 使用例
    print("=" * 60)
    print("LangGraph テーマ生成システム")
    print("=" * 60)

    # サンプルテーマをデータベースに追加（初回のみ）
    init_database()
    existing_themes = get_all_themes()

    if len(existing_themes) == 0:
        print("\n📝 サンプルテーマを追加します...")
        sample_themes = [
            "未来の都市生活",
            "宇宙探検の冒険",
            "AI と人間の共生",
            "持続可能な社会",
            "デジタルアートの革新"
        ]
        for theme in sample_themes:
            add_theme_to_db(theme)
        print(f"✅ {len(sample_themes)}件のサンプルテーマを追加しました")

    # 新しいテーマを生成
    print("\n" + "=" * 60)
    result = generate_unique_theme(
        category="テクノロジーと社会",
        similarity_threshold=0.7,
        max_attempts=5,
        save_to_db=True
    )

    print("\n" + "=" * 60)
    print("📋 生成結果:")
    print(f"  テーマ: {result['theme']}")
    print(f"  ユニーク: {result['is_unique']}")
    print(f"  最大類似度: {result['max_similarity']:.4f}")
    print(f"  試行回数: {result['attempts']}")
    print("=" * 60)
