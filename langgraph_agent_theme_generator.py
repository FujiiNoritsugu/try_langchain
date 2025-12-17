#!/usr/bin/env python3
"""
LangGraphエージェントアーキテクチャによるテーマ生成システム

ルーターアーキテクチャとの違い：
- ルーター：明示的な条件分岐でフローを制御
- エージェント：LLMが自律的にツールを選択・実行（ReActパターン）
"""
import asyncio
import json
import os
import sqlite3
import sys
from pathlib import Path
from typing import Annotated

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# 環境変数を読み込む
load_dotenv()

# カテゴリごとのデータベースパス
CATEGORY_DB_PATHS = {
    "technology": Path(__file__).parent / "themes_technology.db",
    "nature": Path(__file__).parent / "themes_nature.db",
    "lifestyle": Path(__file__).parent / "themes_lifestyle.db",
}


# ===== データベース操作関数 =====
def add_theme_to_db(theme: str, db_path: Path) -> int:
    """データベースに新しいテーマを追加"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # テーブルが存在しない場合は作成
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS themes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            theme_name TEXT UNIQUE NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )

    cursor.execute("INSERT INTO themes (theme_name) VALUES (?)", (theme,))
    theme_id = cursor.lastrowid
    conn.commit()
    conn.close()

    return theme_id


async def check_similarity_via_mcp(candidate: str, threshold: float) -> dict:
    """MCPサーバを使って類似度をチェック"""
    server_script = os.path.join(
        os.path.dirname(__file__), "similarity_checker_mcp_server.py"
    )

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

                return json.loads(response_text)

    except Exception as e:
        print(f"❌ MCP類似度チェックエラー: {e}")
        raise


async def add_theme_to_vector_store(theme: str) -> dict:
    """MCPサーバを使ってChromaベクトルストアに新しいテーマを追加"""
    server_script = os.path.join(
        os.path.dirname(__file__), "similarity_checker_mcp_server.py"
    )

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


# ===== エージェント用ツール定義 =====
@tool
def generate_theme(category: str) -> str:
    """指定されたカテゴリに基づいて新しいテーマを生成します。

    Args:
        category: テーマのカテゴリ (technology, nature, lifestyle)

    Returns:
        生成されたテーマ名
    """
    # カテゴリごとのプロンプト
    category_prompts = {
        "technology": "最新技術やIT分野に関する興味深いテーマを1つ生成してください。",
        "nature": "自然や環境に関する興味深いテーマを1つ生成してください。",
        "lifestyle": "ライフスタイルや日常生活に関する興味深いテーマを1つ生成してください。",
    }

    if category not in category_prompts:
        return f"エラー: 未知のカテゴリ '{category}'"

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.9)

    messages = [
        SystemMessage(
            content="""あなたはクリエイティブなテーマ生成の専門家です。
簡潔で魅力的なテーマ名のみを生成してください。
説明や追加コメントは不要です。テーマ名だけを返してください。"""
        ),
        HumanMessage(content=category_prompts[category]),
    ]

    response = llm.invoke(messages)
    theme = response.content.strip()

    print(f"✨ 生成されたテーマ候補: '{theme}' (カテゴリ: {category})")
    return theme


@tool
def check_theme_similarity(theme: str, threshold: float = 0.7) -> str:
    """生成されたテーマの類似度をチェックします。

    Args:
        theme: チェックするテーマ名
        threshold: 類似度の閾値（デフォルト: 0.7）

    Returns:
        類似度チェックの結果（JSON文字列）
    """
    result = asyncio.run(check_similarity_via_mcp(theme, threshold))

    is_unique = result.get("is_unique", False)
    max_similarity = result.get("max_similarity", 0.0)
    most_similar = result.get("most_similar_text", "N/A")

    print(f"🔍 類似度チェック結果: is_unique={is_unique}, max_similarity={max_similarity:.3f}")

    return json.dumps(result, ensure_ascii=False)


@tool
def save_theme(theme: str, category: str) -> str:
    """テーマをデータベースとベクトルストアに保存します。

    Args:
        theme: 保存するテーマ名
        category: テーマのカテゴリ

    Returns:
        保存結果のメッセージ
    """
    if category not in CATEGORY_DB_PATHS:
        return f"エラー: 未知のカテゴリ '{category}'"

    db_path = CATEGORY_DB_PATHS[category]

    # データベースに保存
    theme_id = add_theme_to_db(theme, db_path)
    db_msg = f"💾 データベースに保存しました (ID: {theme_id})"
    print(db_msg)

    # ベクトルストアに追加
    vector_result = asyncio.run(add_theme_to_vector_store(theme))
    if vector_result.get("success"):
        vector_msg = "🔍 ベクトルストアに追加しました"
        print(vector_msg)
        return f"{db_msg}\n{vector_msg}"
    else:
        error_msg = f"⚠️ ベクトルストアへの追加に失敗: {vector_result.get('error')}"
        print(error_msg)
        return f"{db_msg}\n{error_msg}"


# ===== エージェントの作成 =====
def create_theme_agent():
    """テーマ生成エージェントを作成"""
    # ツールのリスト
    tools = [generate_theme, check_theme_similarity, save_theme]

    # LLMの設定
    llm = ChatOpenAI(model="gpt-4o", temperature=0)

    # ReActエージェントの作成
    agent = create_react_agent(llm, tools)

    return agent


# ===== メイン実行 =====
def main():
    """エージェントを使ってテーマを生成"""
    print("=" * 60)
    print("LangGraph エージェントアーキテクチャ テーマ生成システム")
    print("=" * 60)

    # カテゴリを選択
    category = input("\nカテゴリを選択してください (technology/nature/lifestyle): ").strip().lower()

    if category not in CATEGORY_DB_PATHS:
        print(f"❌ エラー: 未知のカテゴリ '{category}'")
        return

    # エージェントを作成
    agent = create_theme_agent()

    # エージェントに指示
    system_prompt = f"""あなたはテーマ生成の専門家エージェントです。

以下のタスクを実行してください：

1. カテゴリ「{category}」のテーマを generate_theme ツールで生成
2. 生成したテーマの類似度を check_theme_similarity ツールでチェック（閾値: 0.7）
3. ユニーク（is_unique=true）であれば、save_theme ツールで保存
4. ユニークでなければ、最大3回まで再生成を試みる
5. 最終結果を報告

必ず上記の手順に従い、ツールを適切に使用してください。"""

    # エージェントを実行
    print("\n🤖 エージェントを起動しています...\n")

    result = agent.invoke(
        {
            "messages": [HumanMessage(content=system_prompt)],
        }
    )

    # 結果を表示
    print("\n" + "=" * 60)
    print("エージェントの最終レスポンス:")
    print("=" * 60)
    print(result["messages"][-1].content)
    print("=" * 60)


if __name__ == "__main__":
    main()
