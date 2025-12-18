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
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
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
def generate_theme(category: str, improvement_feedback: str = "") -> str:
    """指定されたカテゴリに基づいて新しいテーマを生成します。

    Args:
        category: テーマのカテゴリ (technology, nature, lifestyle)
        improvement_feedback: リフレクションからの改善フィードバック（オプション）

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

    base_instruction = category_prompts[category]

    # フィードバックがある場合は改善を促す
    if improvement_feedback:
        base_instruction += f"\n\n前回の評価フィードバック: {improvement_feedback}\n上記のフィードバックを踏まえて、より良いテーマを生成してください。"
        print(f"💡 改善フィードバックを適用して再生成します")

    messages = [
        SystemMessage(
            content="""あなたはクリエイティブなテーマ生成の専門家です。
簡潔で魅力的なテーマ名のみを生成してください。
説明や追加コメントは不要です。テーマ名だけを返してください。

重要な基準：
- 魅力度: 人々の興味を引く
- 独創性: 新鮮でありふれていない
- カテゴリ適合性: カテゴリとの関連が明確
- 明確性: 意味が分かりやすく具体的"""
        ),
        HumanMessage(content=base_instruction),
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
def review_theme(theme: str, category: str) -> str:
    """生成されたテーマの品質を多角的に評価します。

    Args:
        theme: 評価するテーマ名
        category: テーマのカテゴリ

    Returns:
        評価結果（JSON文字列）
    """
    llm = ChatOpenAI(model="gpt-4o", temperature=0)

    # カテゴリごとの評価基準
    category_criteria = {
        "technology": "最新技術やIT分野との関連性",
        "nature": "自然や環境との関連性",
        "lifestyle": "ライフスタイルや日常生活との関連性",
    }

    evaluation_prompt = f"""以下のテーマを厳格に評価してください：

テーマ: {theme}
カテゴリ: {category}

以下の4つの観点から1-10点で評価し、具体的なフィードバックを提供してください：

1. **魅力度** (1-10点): テーマが人々の興味を引くか、議論したくなるか
2. **独創性** (1-10点): アイデアが新鮮で、ありふれていないか
3. **カテゴリ適合性** (1-10点): {category_criteria.get(category, "カテゴリ")}が明確か
4. **明確性** (1-10点): テーマの意味が分かりやすく、具体的か

JSON形式で以下のように回答してください：
{{
    "scores": {{
        "attractiveness": <1-10の整数>,
        "originality": <1-10の整数>,
        "category_fit": <1-10の整数>,
        "clarity": <1-10の整数>
    }},
    "total_score": <4項目の平均点（小数点1桁）>,
    "feedback": "<具体的な評価コメント>",
    "improvement_suggestions": "<改善案（スコアが7.0未満の場合）>",
    "approved": <true/false: total_scoreが7.0以上ならtrue>
}}

重要：必ずJSONのみを返してください。他の文章は不要です。"""

    messages = [
        SystemMessage(
            content="あなたはテーマの品質を厳格に評価する批評家です。高い基準を持ち、具体的で建設的なフィードバックを提供します。"
        ),
        HumanMessage(content=evaluation_prompt),
    ]

    response = llm.invoke(messages)
    result_text = response.content.strip()

    # JSONブロックを抽出（```json ... ``` の場合に対応）
    if "```json" in result_text:
        import re
        json_match = re.search(r"```json\s*(.*?)\s*```", result_text, re.DOTALL)
        if json_match:
            result_text = json_match.group(1).strip()
    elif "```" in result_text:
        import re
        json_match = re.search(r"```\s*(.*?)\s*```", result_text, re.DOTALL)
        if json_match:
            result_text = json_match.group(1).strip()

    # JSONとして解析
    try:
        result = json.loads(result_text)
        total_score = result.get("total_score", 0)
        approved = result.get("approved", False)

        print(f"📊 品質評価: {total_score:.1f}/10.0 - {'✅ 承認' if approved else '❌ 要改善'}")
        print(f"   魅力度: {result['scores']['attractiveness']}/10")
        print(f"   独創性: {result['scores']['originality']}/10")
        print(f"   適合性: {result['scores']['category_fit']}/10")
        print(f"   明確性: {result['scores']['clarity']}/10")

        return json.dumps(result, ensure_ascii=False)

    except json.JSONDecodeError as e:
        print(f"⚠️ 評価結果のパースに失敗: {e}")
        # フォールバック
        return json.dumps(
            {
                "scores": {"attractiveness": 5, "originality": 5, "category_fit": 5, "clarity": 5},
                "total_score": 5.0,
                "feedback": "評価の解析に失敗しました",
                "improvement_suggestions": "",
                "approved": False,
            },
            ensure_ascii=False,
        )


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
    tools = [generate_theme, review_theme, check_theme_similarity, save_theme]

    # LLMの設定
    llm = ChatOpenAI(model="gpt-4o", temperature=0)

    # ReActエージェントの作成
    agent = create_agent(llm, tools)

    return agent


# ===== メイン実行 =====
def main():
    """エージェントを使ってテーマを生成"""
    print("=" * 70)
    print("LangGraph エージェント + リフレクション テーマ生成システム")
    print("=" * 70)

    # カテゴリを選択
    category = input("\nカテゴリを選択してください (technology/nature/lifestyle): ").strip().lower()

    if category not in CATEGORY_DB_PATHS:
        print(f"❌ エラー: 未知のカテゴリ '{category}'")
        return

    # エージェントを作成
    agent = create_theme_agent()

    # エージェントに指示
    system_prompt = f"""あなたはテーマ生成の専門家エージェントです。リフレクション（自己評価と改善）を活用して高品質なテーマを生成してください。

以下のタスクを実行してください：

【フェーズ1: 生成とリフレクション】
1. カテゴリ「{category}」のテーマを generate_theme ツールで生成
2. 生成したテーマを review_theme ツールで品質評価
   - 評価結果の "approved" が true なら次のフェーズへ
   - false なら "improvement_suggestions" を参考に最大3回まで再生成
   - 3回試しても approved=true にならない場合は、最も高スコアのテーマを採用

【フェーズ2: 類似度チェック】
3. 承認されたテーマの類似度を check_theme_similarity ツールでチェック（閾値: 0.7）
   - ユニーク（is_unique=true）ならフェーズ3へ
   - 重複（is_unique=false）ならフェーズ1に戻る（最大3回）

【フェーズ3: 保存】
4. save_theme ツールでテーマを保存

【最終報告】
5. 生成プロセス（試行回数、リフレクション結果）と最終テーマを報告

重要：
- review_theme の評価基準（魅力度、独創性、適合性、明確性）を理解し、フィードバックを活用すること
- 改善提案に基づいて具体的に改良すること
- 必ず上記の手順に従い、ツールを適切に使用してください"""

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
