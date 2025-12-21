#!/usr/bin/env python3
"""
LangGraphマルチエージェントアーキテクチャによるテーマ生成システム

スーパーバイザーアーキテクチャ：
- Supervisor: 全体のワークフローを管理し、専門エージェントに作業を委譲
- Generator Agent: テーマ生成専門（サブグラフ）
- Reviewer Agent: 品質評価・リフレクション専門（サブグラフ）
- Validator Agent: 類似度チェック専門（サブグラフ）
- Persistence Agent: DB/ベクトルストア保存専門（サブグラフ）
"""
import asyncio
import json
import os
import sqlite3
import sys
from pathlib import Path
from typing import Annotated, Literal, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph, add_messages
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


# ===== 共有状態の定義 =====
class AgentState(TypedDict):
    """全エージェントで共有される状態"""
    messages: Annotated[list[BaseMessage], add_messages]  # メッセージ履歴
    category: str  # テーマのカテゴリ
    current_theme: str  # 現在のテーマ候補
    review_result: dict  # 品質評価結果
    similarity_result: dict  # 類似度チェック結果
    generation_attempts: int  # 生成試行回数
    validation_attempts: int  # 類似度チェック試行回数
    next_agent: str  # 次に実行するエージェント名
    improvement_feedback: str  # 改善フィードバック
    final_theme: str  # 最終的に採用されたテーマ
    is_complete: bool  # 処理が完了したか


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


# ===== ヘルパー関数（各エージェントから呼び出される） =====
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


def check_theme_similarity(theme: str, threshold: float = 0.7) -> dict:
    """生成されたテーマの類似度をチェックします。

    Args:
        theme: チェックするテーマ名
        threshold: 類似度の閾値（デフォルト: 0.7）

    Returns:
        類似度チェックの結果（dict）
    """
    result = asyncio.run(check_similarity_via_mcp(theme, threshold))

    is_unique = result.get("is_unique", False)
    max_similarity = result.get("max_similarity", 0.0)
    most_similar = result.get("most_similar_text", "N/A")

    print(f"🔍 類似度チェック結果: is_unique={is_unique}, max_similarity={max_similarity:.3f}")

    return result


def review_theme(theme: str, category: str) -> dict:
    """生成されたテーマの品質を多角的に評価します。

    Args:
        theme: 評価するテーマ名
        category: テーマのカテゴリ

    Returns:
        評価結果（dict）
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

        return result

    except json.JSONDecodeError as e:
        print(f"⚠️ 評価結果のパースに失敗: {e}")
        # フォールバック
        return {
            "scores": {"attractiveness": 5, "originality": 5, "category_fit": 5, "clarity": 5},
            "total_score": 5.0,
            "feedback": "評価の解析に失敗しました",
            "improvement_suggestions": "",
            "approved": False,
        }


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


# ===== 各専門エージェント（サブグラフ）の実装 =====

# 1. Generator Agent - テーマ生成専門
def generator_agent(state: AgentState) -> AgentState:
    """テーマを生成するエージェント"""
    print(f"\n🎨 Generator Agent: テーマを生成します（試行 {state['generation_attempts'] + 1}回目）")

    theme = generate_theme(state["category"], state.get("improvement_feedback", ""))

    return {
        **state,
        "current_theme": theme,
        "generation_attempts": state["generation_attempts"] + 1,
        "next_agent": "reviewer",
    }


# 2. Reviewer Agent - 品質評価専門
def reviewer_agent(state: AgentState) -> AgentState:
    """テーマの品質を評価するエージェント"""
    print(f"\n📊 Reviewer Agent: テーマを評価します")

    review_result = review_theme(state["current_theme"], state["category"])

    approved = review_result.get("approved", False)
    improvement_suggestions = review_result.get("improvement_suggestions", "")

    # 承認されたか、または最大試行回数に達した場合は次へ
    if approved or state["generation_attempts"] >= 3:
        if state["generation_attempts"] >= 3 and not approved:
            print("⚠️ 最大試行回数に達しました。現在のテーマを採用します。")
        next_agent = "validator"
        improvement_feedback = ""
    else:
        # 不承認の場合は再生成
        next_agent = "generator"
        improvement_feedback = improvement_suggestions

    return {
        **state,
        "review_result": review_result,
        "improvement_feedback": improvement_feedback,
        "next_agent": next_agent,
    }


# 3. Validator Agent - 類似度チェック専門
def validator_agent(state: AgentState) -> AgentState:
    """テーマの類似度をチェックするエージェント"""
    print(f"\n🔍 Validator Agent: 類似度をチェックします（試行 {state['validation_attempts'] + 1}回目）")

    similarity_result = check_theme_similarity(state["current_theme"], threshold=0.7)

    is_unique = similarity_result.get("is_unique", False)

    # ユニークか、または最大試行回数に達した場合は保存へ
    if is_unique or state["validation_attempts"] >= 3:
        if state["validation_attempts"] >= 3 and not is_unique:
            print("⚠️ 最大類似度チェック試行回数に達しました。現在のテーマを採用します。")
        next_agent = "persistence"
    else:
        # 重複の場合は再生成（カウンターをリセットせず継続）
        next_agent = "generator"

    return {
        **state,
        "similarity_result": similarity_result,
        "validation_attempts": state["validation_attempts"] + 1,
        "next_agent": next_agent,
    }


# 4. Persistence Agent - 保存専門
def persistence_agent(state: AgentState) -> AgentState:
    """テーマを保存するエージェント"""
    print(f"\n💾 Persistence Agent: テーマを保存します")

    result = save_theme(state["current_theme"], state["category"])

    return {
        **state,
        "final_theme": state["current_theme"],
        "is_complete": True,
        "next_agent": "__end__",  # END定数の値
    }


# 5. Supervisor - 全体を管理するスーパーバイザー
def supervisor_node(state: AgentState) -> AgentState:
    """ワークフロー全体を管理するスーパーバイザー"""
    # 次のエージェントを決定（すでに各エージェントが設定している）
    print(f"\n👔 Supervisor: 次のエージェントは '{state['next_agent']}' です")
    return state


# ===== ルーティング関数 =====
def route_to_next_agent(state: AgentState) -> str:
    """次のエージェントへのルーティングを決定"""
    next_agent = state.get("next_agent", "generator")
    return next_agent


# ===== グラフ構築 =====
def create_multi_agent_graph() -> StateGraph:
    """マルチエージェントグラフを構築"""
    # グラフの初期化
    workflow = StateGraph(AgentState)

    # ノードの追加
    workflow.add_node("supervisor", supervisor_node)
    workflow.add_node("generator", generator_agent)
    workflow.add_node("reviewer", reviewer_agent)
    workflow.add_node("validator", validator_agent)
    workflow.add_node("persistence", persistence_agent)

    # エッジの追加
    # 開始 -> スーパーバイザー
    workflow.set_entry_point("supervisor")

    # スーパーバイザーから各エージェントへの条件分岐
    workflow.add_conditional_edges(
        "supervisor",
        route_to_next_agent,
        {
            "generator": "generator",
            "reviewer": "reviewer",
            "validator": "validator",
            "persistence": "persistence",
            "__end__": END,  # END定数の実際の値をキーに使用
        },
    )

    # 各エージェントからスーパーバイザーへ戻る
    workflow.add_edge("generator", "supervisor")
    workflow.add_edge("reviewer", "supervisor")
    workflow.add_edge("validator", "supervisor")
    workflow.add_edge("persistence", "supervisor")

    return workflow.compile()


# ===== メイン実行 =====
def main():
    """マルチエージェントシステムを使ってテーマを生成"""
    print("=" * 70)
    print("LangGraph マルチエージェント + スーパーバイザー テーマ生成システム")
    print("=" * 70)

    # カテゴリを選択
    category = input("\nカテゴリを選択してください (technology/nature/lifestyle): ").strip().lower()

    if category not in CATEGORY_DB_PATHS:
        print(f"❌ エラー: 未知のカテゴリ '{category}'")
        return

    # マルチエージェントグラフを作成
    graph = create_multi_agent_graph()

    # 初期状態を設定
    initial_state: AgentState = {
        "messages": [],
        "category": category,
        "current_theme": "",
        "review_result": {},
        "similarity_result": {},
        "generation_attempts": 0,
        "validation_attempts": 0,
        "next_agent": "generator",
        "improvement_feedback": "",
        "final_theme": "",
        "is_complete": False,
    }

    # グラフを実行
    print("\n🤖 マルチエージェントシステムを起動しています...\n")

    result = graph.invoke(initial_state)

    # 結果を表示
    print("\n" + "=" * 70)
    print("🎉 テーマ生成が完了しました！")
    print("=" * 70)
    print(f"最終テーマ: {result['final_theme']}")
    print(f"カテゴリ: {result['category']}")
    print(f"生成試行回数: {result['generation_attempts']}")
    print(f"類似度チェック試行回数: {result['validation_attempts']}")

    if result.get("review_result"):
        review = result["review_result"]
        print(f"\n品質評価スコア: {review.get('total_score', 'N/A')}/10.0")
        print(f"  - 魅力度: {review.get('scores', {}).get('attractiveness', 'N/A')}/10")
        print(f"  - 独創性: {review.get('scores', {}).get('originality', 'N/A')}/10")
        print(f"  - 適合性: {review.get('scores', {}).get('category_fit', 'N/A')}/10")
        print(f"  - 明確性: {review.get('scores', {}).get('clarity', 'N/A')}/10")

    if result.get("similarity_result"):
        sim = result["similarity_result"]
        print(f"\n類似度チェック:")
        print(f"  - ユニーク: {'はい' if sim.get('is_unique') else 'いいえ'}")
        print(f"  - 最大類似度: {sim.get('max_similarity', 0):.3f}")

    print("=" * 70)


if __name__ == "__main__":
    main()
