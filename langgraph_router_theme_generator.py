"""
LangGraphのルーターアーキテクチャを使用したテーマ生成プログラム
ユーザー入力に基づいて系列を判定し、系列別データベースで類似度チェックを行う
"""

import sqlite3
import uuid
from typing import TypedDict, List
import os
import json
import asyncio
import sys
import time
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langgraph.graph import StateGraph, END
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# .envファイルから環境変数を読み込む
load_dotenv()


# テーマ系列の定義
THEME_CATEGORIES = {
    "technology": {
        "name": "テクノロジー",
        "db_path": "themes_technology.db",
        "description": "AI、ソフトウェア、ハードウェア、デジタル技術など"
    },
    "art": {
        "name": "芸術・文化",
        "db_path": "themes_art.db",
        "description": "美術、音楽、文学、デザイン、エンターテインメントなど"
    },
    "business": {
        "name": "ビジネス",
        "db_path": "themes_business.db",
        "description": "経営、マーケティング、起業、経済、金融など"
    },
    "nature": {
        "name": "自然・環境",
        "db_path": "themes_nature.db",
        "description": "環境保護、気候変動、生態系、持続可能性など"
    },
    "lifestyle": {
        "name": "ライフスタイル",
        "db_path": "themes_lifestyle.db",
        "description": "健康、教育、趣味、旅行、日常生活など"
    }
}


# データベース関連関数
def init_database(db_path: str):
    """SQLiteデータベースを初期化"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS themes (
            id TEXT PRIMARY KEY,
            theme_name TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """
    )

    conn.commit()
    conn.close()


def add_theme_to_db(theme_name: str, db_path: str):
    """テーマをデータベースに追加"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    theme_id = str(uuid.uuid4())
    cursor.execute(
        "INSERT INTO themes (id, theme_name) VALUES (?, ?)", (theme_id, theme_name)
    )

    conn.commit()
    conn.close()
    return theme_id


def get_all_themes(db_path: str) -> List[str]:
    """データベースから全てのテーマ名を取得"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("SELECT theme_name FROM themes")
    themes = [row[0] for row in cursor.fetchall()]

    conn.close()
    return themes


# LangGraphの状態定義
class RouterThemeState(TypedDict):
    """ルーター型テーマ生成プロセスの状態"""

    user_input: str  # ユーザーの入力文字列
    detected_category: str  # 検出されたテーマ系列
    category_name: str  # 系列の日本語名
    db_path: str  # 使用するデータベースパス
    existing_themes: List[str]  # 既存のテーマ一覧
    candidate_theme: str  # 生成候補のテーマ
    is_unique: bool  # ユニークかどうか
    attempt_count: int  # 試行回数
    max_attempts: int  # 最大試行回数
    similarity_threshold: float  # 類似度の閾値
    max_similarity: float  # 既存テーマとの最大類似度
    save_to_db: bool  # DBに保存するかどうか
    final_message: str  # 最終メッセージ


# ノード関数の定義

def route_category(state: RouterThemeState) -> RouterThemeState:
    """ユーザー入力からテーマ系列を判定するルーターノード"""
    print(f"\n🔍 ユーザー入力: '{state['user_input']}'")
    print("🤖 テーマ系列を判定中...")

    llm = ChatAnthropic(model="claude-3-haiku-20240307", temperature=0.3)

    categories_desc = "\n".join(
        [f"- {key}: {info['name']} ({info['description']})"
         for key, info in THEME_CATEGORIES.items()]
    )

    prompt = f"""以下のユーザー入力が、どのテーマ系列に該当するか判定してください。

ユーザー入力: {state['user_input']}

利用可能なテーマ系列:
{categories_desc}

以下のいずれかのキーのみを出力してください（説明は不要）:
{', '.join(THEME_CATEGORIES.keys())}

最も適切な系列を1つだけ選択してください。
"""

    response = llm.invoke(prompt)
    detected = response.content.strip().lower()

    # 有効な系列かチェック
    if detected not in THEME_CATEGORIES:
        # デフォルトはlifestyle
        detected = "lifestyle"
        print(f"⚠️  不明な系列のためデフォルト設定: {detected}")

    category_info = THEME_CATEGORIES[detected]
    state["detected_category"] = detected
    state["category_name"] = category_info["name"]
    state["db_path"] = category_info["db_path"]

    print(f"✅ 判定結果: {state['category_name']} ({detected})")

    return state


def router_decision(state: RouterThemeState) -> str:
    """ルーターの判定結果に基づいて次のノードを決定"""
    return state["detected_category"]


# 各系列専用の処理ノード

def process_technology(state: RouterThemeState) -> RouterThemeState:
    """テクノロジー系列専用の処理ノード"""
    print(f"\n🔧 [{state['category_name']}] 専用処理を開始")

    # データベースから既存テーマを取得
    init_database(state["db_path"])
    state["existing_themes"] = get_all_themes(state["db_path"])
    print(f"📚 既存テーマ数: {len(state['existing_themes'])}")

    # テーマ生成と類似度チェックのループ
    while state["attempt_count"] < state["max_attempts"]:
        state["attempt_count"] += 1
        print(f"\n🎲 テーマ生成試行: {state['attempt_count']}/{state['max_attempts']}")

        # テクノロジー系列専用のプロンプト
        llm = ChatAnthropic(model="claude-3-haiku-20240307", temperature=0.9)
        existing_themes_str = "\n".join([f"- {theme}" for theme in state["existing_themes"]])

        prompt = f"""あなたはテクノロジー分野の専門家です。以下の要望に基づいて、革新的で技術的に興味深いテーマを生成してください。

ユーザーの要望: {state['user_input']}

既存のテーマ:
{existing_themes_str if existing_themes_str else "（まだテーマは登録されていません）"}

要件:
- 最新技術トレンドを意識すること
- 技術的な実現可能性を考慮すること
- 既存テーマと差別化されていること
- 簡潔で魅力的なテーマ名（説明は不要）
"""

        response = llm.invoke(prompt)
        state["candidate_theme"] = response.content.strip()
        print(f"💡 生成されたテーマ: {state['candidate_theme']}")

        # 類似度チェック
        if not state["existing_themes"]:
            state["is_unique"] = True
            state["max_similarity"] = 0.0
            print("✅ 既存テーマがないため、ユニークと判定")
            break

        result = asyncio.run(
            check_similarity_via_mcp(
                candidate=state["candidate_theme"],
                threshold=state["similarity_threshold"],
            )
        )

        state["max_similarity"] = result["max_similarity"]
        state["is_unique"] = result["is_unique"]

        print(f"📊 類似度: {state['max_similarity']:.4f} (閾値: {state['similarity_threshold']})")

        if state["is_unique"]:
            print("✅ ユニークなテーマと判定")
            break
        else:
            print(f"⚠️  類似テーマ検出: '{result.get('most_similar_text')}' - 再生成します")

    return state


def process_art(state: RouterThemeState) -> RouterThemeState:
    """芸術・文化系列専用の処理ノード"""
    print(f"\n🎨 [{state['category_name']}] 専用処理を開始")

    init_database(state["db_path"])
    state["existing_themes"] = get_all_themes(state["db_path"])
    print(f"📚 既存テーマ数: {len(state['existing_themes'])}")

    while state["attempt_count"] < state["max_attempts"]:
        state["attempt_count"] += 1
        print(f"\n🎲 テーマ生成試行: {state['attempt_count']}/{state['max_attempts']}")

        llm = ChatAnthropic(model="claude-3-haiku-20240307", temperature=0.95)  # より高い創造性
        existing_themes_str = "\n".join([f"- {theme}" for theme in state["existing_themes"]])

        prompt = f"""あなたは芸術・文化分野の専門家です。創造的で美的センスのあるテーマを生成してください。

ユーザーの要望: {state['user_input']}

既存のテーマ:
{existing_themes_str if existing_themes_str else "（まだテーマは登録されていません）"}

要件:
- 芸術的な表現を重視すること
- 感性に訴える言葉選びをすること
- 既存テーマと差別化されていること
- 簡潔で魅力的なテーマ名（説明は不要）
"""

        response = llm.invoke(prompt)
        state["candidate_theme"] = response.content.strip()
        print(f"💡 生成されたテーマ: {state['candidate_theme']}")

        if not state["existing_themes"]:
            state["is_unique"] = True
            state["max_similarity"] = 0.0
            print("✅ 既存テーマがないため、ユニークと判定")
            break

        result = asyncio.run(
            check_similarity_via_mcp(
                candidate=state["candidate_theme"],
                threshold=state["similarity_threshold"],
            )
        )

        state["max_similarity"] = result["max_similarity"]
        state["is_unique"] = result["is_unique"]

        print(f"📊 類似度: {state['max_similarity']:.4f} (閾値: {state['similarity_threshold']})")

        if state["is_unique"]:
            print("✅ ユニークなテーマと判定")
            break
        else:
            print(f"⚠️  類似テーマ検出: '{result.get('most_similar_text')}' - 再生成します")

    return state


def process_business(state: RouterThemeState) -> RouterThemeState:
    """ビジネス系列専用の処理ノード"""
    print(f"\n💼 [{state['category_name']}] 専用処理を開始")

    init_database(state["db_path"])
    state["existing_themes"] = get_all_themes(state["db_path"])
    print(f"📚 既存テーマ数: {len(state['existing_themes'])}")

    while state["attempt_count"] < state["max_attempts"]:
        state["attempt_count"] += 1
        print(f"\n🎲 テーマ生成試行: {state['attempt_count']}/{state['max_attempts']}")

        llm = ChatAnthropic(model="claude-3-haiku-20240307", temperature=0.85)
        existing_themes_str = "\n".join([f"- {theme}" for theme in state["existing_themes"]])

        prompt = f"""あなたはビジネス分野の専門家です。実用的でビジネス価値のあるテーマを生成してください。

ユーザーの要望: {state['user_input']}

既存のテーマ:
{existing_themes_str if existing_themes_str else "（まだテーマは登録されていません）"}

要件:
- ビジネス視点での価値を明確にすること
- 実現可能性と市場性を考慮すること
- 既存テーマと差別化されていること
- 簡潔で魅力的なテーマ名（説明は不要）
"""

        response = llm.invoke(prompt)
        state["candidate_theme"] = response.content.strip()
        print(f"💡 生成されたテーマ: {state['candidate_theme']}")

        if not state["existing_themes"]:
            state["is_unique"] = True
            state["max_similarity"] = 0.0
            print("✅ 既存テーマがないため、ユニークと判定")
            break

        result = asyncio.run(
            check_similarity_via_mcp(
                candidate=state["candidate_theme"],
                threshold=state["similarity_threshold"],
            )
        )

        state["max_similarity"] = result["max_similarity"]
        state["is_unique"] = result["is_unique"]

        print(f"📊 類似度: {state['max_similarity']:.4f} (閾値: {state['similarity_threshold']})")

        if state["is_unique"]:
            print("✅ ユニークなテーマと判定")
            break
        else:
            print(f"⚠️  類似テーマ検出: '{result.get('most_similar_text')}' - 再生成します")

    return state


def process_nature(state: RouterThemeState) -> RouterThemeState:
    """自然・環境系列専用の処理ノード"""
    print(f"\n🌿 [{state['category_name']}] 専用処理を開始")

    init_database(state["db_path"])
    state["existing_themes"] = get_all_themes(state["db_path"])
    print(f"📚 既存テーマ数: {len(state['existing_themes'])}")

    while state["attempt_count"] < state["max_attempts"]:
        state["attempt_count"] += 1
        print(f"\n🎲 テーマ生成試行: {state['attempt_count']}/{state['max_attempts']}")

        llm = ChatAnthropic(model="claude-3-haiku-20240307", temperature=0.9)
        existing_themes_str = "\n".join([f"- {theme}" for theme in state["existing_themes"]])

        prompt = f"""あなたは環境・自然分野の専門家です。持続可能性と環境保護を重視したテーマを生成してください。

ユーザーの要望: {state['user_input']}

既存のテーマ:
{existing_themes_str if existing_themes_str else "（まだテーマは登録されていません）"}

要件:
- 環境への配慮と持続可能性を強調すること
- 自然との共生を意識すること
- 既存テーマと差別化されていること
- 簡潔で魅力的なテーマ名（説明は不要）
"""

        response = llm.invoke(prompt)
        state["candidate_theme"] = response.content.strip()
        print(f"💡 生成されたテーマ: {state['candidate_theme']}")

        if not state["existing_themes"]:
            state["is_unique"] = True
            state["max_similarity"] = 0.0
            print("✅ 既存テーマがないため、ユニークと判定")
            break

        result = asyncio.run(
            check_similarity_via_mcp(
                candidate=state["candidate_theme"],
                threshold=state["similarity_threshold"],
            )
        )

        state["max_similarity"] = result["max_similarity"]
        state["is_unique"] = result["is_unique"]

        print(f"📊 類似度: {state['max_similarity']:.4f} (閾値: {state['similarity_threshold']})")

        if state["is_unique"]:
            print("✅ ユニークなテーマと判定")
            break
        else:
            print(f"⚠️  類似テーマ検出: '{result.get('most_similar_text')}' - 再生成します")

    return state


def process_lifestyle(state: RouterThemeState) -> RouterThemeState:
    """ライフスタイル系列専用の処理ノード"""
    print(f"\n🏠 [{state['category_name']}] 専用処理を開始")

    init_database(state["db_path"])
    state["existing_themes"] = get_all_themes(state["db_path"])
    print(f"📚 既存テーマ数: {len(state['existing_themes'])}")

    while state["attempt_count"] < state["max_attempts"]:
        state["attempt_count"] += 1
        print(f"\n🎲 テーマ生成試行: {state['attempt_count']}/{state['max_attempts']}")

        llm = ChatAnthropic(model="claude-3-haiku-20240307", temperature=0.9)
        existing_themes_str = "\n".join([f"- {theme}" for theme in state["existing_themes"]])

        prompt = f"""あなたはライフスタイル分野の専門家です。日常生活を豊かにするテーマを生成してください。

ユーザーの要望: {state['user_input']}

既存のテーマ:
{existing_themes_str if existing_themes_str else "（まだテーマは登録されていません）"}

要件:
- 日常生活に根ざした実用性を重視すること
- 生活の質を向上させる視点を持つこと
- 既存テーマと差別化されていること
- 簡潔で魅力的なテーマ名（説明は不要）
"""

        response = llm.invoke(prompt)
        state["candidate_theme"] = response.content.strip()
        print(f"💡 生成されたテーマ: {state['candidate_theme']}")

        if not state["existing_themes"]:
            state["is_unique"] = True
            state["max_similarity"] = 0.0
            print("✅ 既存テーマがないため、ユニークと判定")
            break

        result = asyncio.run(
            check_similarity_via_mcp(
                candidate=state["candidate_theme"],
                threshold=state["similarity_threshold"],
            )
        )

        state["max_similarity"] = result["max_similarity"]
        state["is_unique"] = result["is_unique"]

        print(f"📊 類似度: {state['max_similarity']:.4f} (閾値: {state['similarity_threshold']})")

        if state["is_unique"]:
            print("✅ ユニークなテーマと判定")
            break
        else:
            print(f"⚠️  類似テーマ検出: '{result.get('most_similar_text')}' - 再生成します")

    return state


async def add_theme_to_vector_store(theme: str) -> dict:
    """MCPサーバを使ってChromaベクトルストアに新しいテーマを追加"""
    server_script = os.path.join(
        os.path.dirname(__file__), "similarity_checker_mcp_server.py"
    )

    # 環境変数を引き継ぐ
    server_params = StdioServerParameters(
        command=sys.executable, args=[server_script], env=os.environ.copy()
    )

    try:
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()

                result = await session.call_tool(
                    "add_theme",
                    arguments={"theme": theme},
                )

                if not result.content or len(result.content) == 0:
                    raise ValueError("MCPサーバからの応答が空です")

                response_text = result.content[0].text
                return json.loads(response_text)

    except Exception as e:
        print(f"❌ ベクトルストアへの追加エラー: {e}")
        return {"success": False, "error": str(e)}


async def check_similarity_via_mcp(
    candidate: str, threshold: float
) -> dict:
    """MCPサーバを使って類似度をチェック（既存テーマはベクトルDBから取得）"""
    server_script = os.path.join(
        os.path.dirname(__file__), "similarity_checker_mcp_server.py"
    )

    # 環境変数を引き継ぐ
    server_params = StdioServerParameters(
        command=sys.executable, args=[server_script], env=os.environ.copy()
    )

    try:
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()

                result = await session.call_tool(
                    "check_similarity",
                    arguments={
                        "candidate": candidate,
                        "threshold": threshold,
                    },
                )

                if not result.content or len(result.content) == 0:
                    raise ValueError("MCPサーバからの応答が空です")

                response_text = result.content[0].text

                if not response_text:
                    raise ValueError("MCPサーバからの応答テキストが空です")

                # エラーメッセージの場合は例外を発生
                if response_text.startswith("Error"):
                    raise ValueError(f"MCPサーバエラー: {response_text}")

                return json.loads(response_text)
    except Exception as e:
        print(f"❌ MCPサーバとの通信エラー: {e}")
        raise


def finalize(state: RouterThemeState) -> RouterThemeState:
    """最終処理：データベース保存と結果表示"""
    print(f"\n{'=' * 60}")

    if state["is_unique"]:
        message = f"🎉 ユニークなテーマが生成されました: '{state['candidate_theme']}'"
        print(message)

        # データベースに保存
        if state["save_to_db"] and state["candidate_theme"]:
            theme_id = add_theme_to_db(state["candidate_theme"], state["db_path"])
            save_msg = f"💾 [{state['category_name']}] データベースに保存しました (ID: {theme_id})"
            print(save_msg)
            message += f"\n{save_msg}"

            # Chromaベクトルストアにも追加
            vector_result = asyncio.run(add_theme_to_vector_store(state["candidate_theme"]))
            if vector_result.get("success"):
                vector_msg = f"🔍 ベクトルストアに追加しました"
                print(vector_msg)
                message += f"\n{vector_msg}"
            else:
                error_msg = f"⚠️  ベクトルストアへの追加に失敗: {vector_result.get('error')}"
                print(error_msg)
    else:
        message = f"⚠️  完全にユニークなテーマは生成できませんでしたが、最善の候補: '{state['candidate_theme']}'"
        print(message)

    state["final_message"] = message
    print(f"{'=' * 60}\n")

    return state


# LangGraphの構築
def create_router_theme_graph():
    """真のルーター型テーマ生成グラフを作成"""
    workflow = StateGraph(RouterThemeState)

    # ノードの追加
    workflow.add_node("route", route_category)

    # 各系列専用のノードを追加
    workflow.add_node("technology", process_technology)
    workflow.add_node("art", process_art)
    workflow.add_node("business", process_business)
    workflow.add_node("nature", process_nature)
    workflow.add_node("lifestyle", process_lifestyle)

    workflow.add_node("finalize", finalize)

    # エッジの定義
    workflow.set_entry_point("route")

    # ルーターから各系列ノードへの条件分岐エッジ
    workflow.add_conditional_edges(
        "route",
        router_decision,
        {
            "technology": "technology",
            "art": "art",
            "business": "business",
            "nature": "nature",
            "lifestyle": "lifestyle",
        }
    )

    # 各系列ノードから最終処理へ
    workflow.add_edge("technology", "finalize")
    workflow.add_edge("art", "finalize")
    workflow.add_edge("business", "finalize")
    workflow.add_edge("nature", "finalize")
    workflow.add_edge("lifestyle", "finalize")

    workflow.add_edge("finalize", END)

    return workflow.compile()


def generate_theme_from_input(
    user_input: str,
    similarity_threshold: float = 0.7,
    max_attempts: int = 5,
    save_to_db: bool = True,
) -> dict:
    """
    ユーザー入力に基づいてテーマを生成

    Args:
        user_input: ユーザーの入力文字列
        similarity_threshold: 類似度の閾値（この値未満ならユニーク）
        max_attempts: 最大試行回数
        save_to_db: 生成したテーマをデータベースに保存するか

    Returns:
        生成結果の辞書
    """
    # グラフの作成
    app = create_router_theme_graph()

    # 初期状態
    initial_state = {
        "user_input": user_input,
        "detected_category": "",
        "category_name": "",
        "db_path": "",
        "existing_themes": [],
        "candidate_theme": "",
        "is_unique": False,
        "attempt_count": 0,
        "max_attempts": max_attempts,
        "similarity_threshold": similarity_threshold,
        "max_similarity": 0.0,
        "save_to_db": save_to_db,
        "final_message": "",
    }

    # グラフの実行
    result = app.invoke(initial_state)

    return {
        "user_input": result["user_input"],
        "category": result["category_name"],
        "theme": result["candidate_theme"],
        "is_unique": result["is_unique"],
        "max_similarity": result["max_similarity"],
        "attempts": result["attempt_count"],
        "message": result["final_message"],
    }


if __name__ == "__main__":
    print("=" * 60)
    print("LangGraph ルーター型テーマ生成システム")
    print("=" * 60)

    # グラフの可視化
    print("\n📊 グラフ構造を可視化します...\n")
    app = create_router_theme_graph()

    # Mermaid記法で表示とファイル保存
    print("=== グラフ構造 (Mermaid) ===")
    mermaid_code = app.get_graph().draw_mermaid()
    print(mermaid_code)
    print()

    # Mermaidファイルとして保存
    mermaid_filename = "router_theme_generator_graph.md"
    with open(mermaid_filename, "w", encoding="utf-8") as f:
        f.write("# ルーター型テーマ生成グラフ構造\n\n")
        f.write("```mermaid\n")
        f.write(mermaid_code)
        f.write("\n```\n")
    print(f"✅ Mermaidグラフを {mermaid_filename} に保存しました")
    print()

    print("=" * 60)

    # 全データベースの初期化
    print("\n📝 系列別データベースを初期化します...")
    for category_key, category_info in THEME_CATEGORIES.items():
        init_database(category_info["db_path"])
        print(f"  ✓ {category_info['name']}: {category_info['db_path']}")

    # サンプルテーマを追加（初回のみ）
    print("\n📝 サンプルテーマを確認・追加します...")

    sample_data = {
        "technology": ["量子コンピューティングの実用化", "AI倫理ガイドライン"],
        "art": ["デジタルアートの未来", "音楽とテクノロジーの融合"],
        "business": ["リモートワーク時代の組織改革", "サステナブルビジネスモデル"],
        "nature": ["都市緑化プロジェクト", "海洋プラスチック削減"],
        "lifestyle": ["マインドフルネス実践", "デジタルデトックスの方法"],
    }

    for category_key, themes in sample_data.items():
        db_path = THEME_CATEGORIES[category_key]["db_path"]
        existing = get_all_themes(db_path)
        if len(existing) == 0:
            for theme in themes:
                add_theme_to_db(theme, db_path)
            print(f"  ✓ {THEME_CATEGORIES[category_key]['name']}: {len(themes)}件追加")
        else:
            print(
                f"  ✓ {THEME_CATEGORIES[category_key]['name']}: 既存テーマ{len(existing)}件"
            )

    # ユーザー入力を受け付けてテーマを生成
    print("\n" + "=" * 60)
    print("テーマ生成")
    print("=" * 60)
    print("\n💡 テーマに関連する文字列を入力してください")
    print("   例: '次世代のプログラミング言語について考える'")
    print("   例: '地球温暖化を防ぐための取り組み'")
    print("   例: 'デジタルアートの新しい表現方法'")
    print("\n終了するには 'q' または 'quit' を入力してください\n")

    while True:
        user_input = input("入力 > ").strip()

        if user_input.lower() in ["q", "quit", "exit"]:
            print("\n👋 終了します")
            break

        if not user_input:
            print("⚠️  空の入力です。もう一度入力してください。\n")
            continue

        print("\n" + "=" * 60)
        try:
            result = generate_theme_from_input(
                user_input=user_input,
                similarity_threshold=0.7,
                max_attempts=5,
                save_to_db=True,
            )

            print("\n📋 生成結果:")
            print(f"  入力: {result['user_input']}")
            print(f"  系列: {result['category']}")
            print(f"  テーマ: {result['theme']}")
            print(f"  ユニーク: {result['is_unique']}")
            print(f"  最大類似度: {result['max_similarity']:.4f}")
            print(f"  試行回数: {result['attempts']}")
            print("=" * 60 + "\n")

        except KeyboardInterrupt:
            print("\n\n👋 中断されました")
            break
        except Exception as e:
            print(f"\n❌ エラーが発生しました: {e}")
            print("=" * 60 + "\n")
            # レート制限エラーの場合は待機を促す
            if "rate limit" in str(e).lower():
                print("⏳ レート制限エラーです。30秒待機してから再試行してください。\n")

    print("\n" + "=" * 60)
    print("完了")
    print("=" * 60)
