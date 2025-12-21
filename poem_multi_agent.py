#!/usr/bin/env python3
"""
LangGraphマルチエージェントアーキテクチャによる詩作成システム

スーパーバイザーアーキテクチャ：
- Supervisor: 全体のワークフローを管理
- Theme Agent: 詩のテーマを決定（ユーザー入力 or 自動生成）
- Composer Agent: テーマに基づいて詩を作成
- Critic Agent: 詩を評価し、改善提案を提供
- Finalizer Agent: 最終版を確定して出力
"""
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph, add_messages

# 環境変数を読み込む
load_dotenv()


# ===== 共有状態の定義 =====
class PoemState(TypedDict):
    """全エージェントで共有される状態"""
    messages: Annotated[list[BaseMessage], add_messages]  # メッセージ履歴
    theme: str  # 詩のテーマ
    current_poem: str  # 現在の詩
    critique: dict  # 批評結果
    improvement_count: int  # 改善回数
    next_agent: str  # 次に実行するエージェント名
    final_poem: str  # 最終的な詩
    is_complete: bool  # 処理が完了したか


# ===== 各専門エージェントの実装 =====

# 1. Theme Agent - テーマ決定専門
def theme_agent(state: PoemState) -> PoemState:
    """詩のテーマを決定するエージェント"""
    print("\n🎭 Theme Agent: 詩のテーマを決定します")

    # ユーザーが既にテーマを指定している場合はそのまま使用
    theme = state.get("theme", "")

    if not theme:
        # テーマが指定されていない場合は自動生成
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.8)

        messages = [
            SystemMessage(content="あなたは創造的な詩人です。美しく詩的なテーマを1つ提案してください。"),
            HumanMessage(content="詩のテーマを1つ提案してください。テーマ名のみを簡潔に返してください。")
        ]

        response = llm.invoke(messages)
        theme = response.content.strip()

    print(f"✨ 決定されたテーマ: '{theme}'")

    return {
        **state,
        "theme": theme,
        "next_agent": "composer",
    }


# 2. Composer Agent - 詩作成専門
def composer_agent(state: PoemState) -> PoemState:
    """詩を作成するエージェント"""
    print(f"\n✍️  Composer Agent: 詩を作成します（試行 {state['improvement_count'] + 1}回目）")

    llm = ChatOpenAI(model="gpt-4o", temperature=0.9)

    # 基本プロンプト
    base_prompt = f"テーマ: {state['theme']}\n\n上記のテーマで、心に響く美しい詩を作成してください。"

    # 改善フィードバックがある場合
    if state.get("critique") and state["improvement_count"] > 0:
        critique = state["critique"]
        feedback = critique.get("feedback", "")
        suggestions = critique.get("suggestions", "")

        base_prompt += f"\n\n【前回の批評】\n{feedback}\n\n【改善提案】\n{suggestions}\n\n上記のフィードバックを踏まえて、より良い詩を作成してください。"
        print("💡 批評家のフィードバックを適用して再作成します")

    messages = [
        SystemMessage(content="""あなたは感性豊かな詩人です。
以下の基準で詩を作成してください：
- 4-8行程度の長さ
- 美しい言葉選び
- 感情を揺さぶる表現
- リズムや韻を意識

詩のみを返してください。説明や前置きは不要です。"""),
        HumanMessage(content=base_prompt)
    ]

    response = llm.invoke(messages)
    poem = response.content.strip()

    print(f"\n📜 作成された詩:\n{poem}\n")

    return {
        **state,
        "current_poem": poem,
        "improvement_count": state["improvement_count"] + 1,
        "next_agent": "critic",
    }


# 3. Critic Agent - 評価・改善提案専門
def critic_agent(state: PoemState) -> PoemState:
    """詩を評価し、改善提案を行うエージェント"""
    print("\n📊 Critic Agent: 詩を評価します")

    llm = ChatOpenAI(model="gpt-4o", temperature=0.3)

    evaluation_prompt = f"""以下の詩を評価してください：

テーマ: {state['theme']}

詩:
{state['current_poem']}

以下の観点から評価してください：
1. **テーマ適合性** (1-10点): テーマがよく表現されているか
2. **言葉の美しさ** (1-10点): 言葉選びが美しく詩的か
3. **感情の深さ** (1-10点): 読む人の心を動かすか
4. **リズム・韻** (1-10点): 読みやすく、耳に心地よいか

総合評価が7.0以上なら承認、未満なら具体的な改善提案を提供してください。

JSON形式で回答してください：
{{
    "scores": {{
        "theme_fit": <1-10の整数>,
        "word_beauty": <1-10の整数>,
        "emotional_depth": <1-10の整数>,
        "rhythm": <1-10の整数>
    }},
    "total_score": <平均点（小数点1桁）>,
    "feedback": "<具体的な評価コメント>",
    "suggestions": "<改善提案（total_scoreが7.0未満の場合）>",
    "approved": <true/false: total_scoreが7.0以上ならtrue>
}}"""

    messages = [
        SystemMessage(content="あなたは厳格だが建設的な詩の批評家です。高い基準を持ち、具体的なフィードバックを提供します。"),
        HumanMessage(content=evaluation_prompt)
    ]

    response = llm.invoke(messages)
    result_text = response.content.strip()

    # JSONブロックを抽出
    import json
    import re

    if "```json" in result_text:
        json_match = re.search(r"```json\s*(.*?)\s*```", result_text, re.DOTALL)
        if json_match:
            result_text = json_match.group(1).strip()
    elif "```" in result_text:
        json_match = re.search(r"```\s*(.*?)\s*```", result_text, re.DOTALL)
        if json_match:
            result_text = json_match.group(1).strip()

    try:
        critique = json.loads(result_text)
        total_score = critique.get("total_score", 0)
        approved = critique.get("approved", False)

        print(f"📊 評価スコア: {total_score:.1f}/10.0 - {'✅ 承認' if approved else '❌ 要改善'}")
        print(f"   テーマ適合性: {critique['scores']['theme_fit']}/10")
        print(f"   言葉の美しさ: {critique['scores']['word_beauty']}/10")
        print(f"   感情の深さ: {critique['scores']['emotional_depth']}/10")
        print(f"   リズム・韻: {critique['scores']['rhythm']}/10")

        # 承認されたか、または最大改善回数に達したら次へ
        if approved or state["improvement_count"] >= 3:
            if state["improvement_count"] >= 3 and not approved:
                print("⚠️ 最大改善回数に達しました。現在の詩を採用します。")
            next_agent = "finalizer"
        else:
            # 改善が必要な場合は再作成
            next_agent = "composer"

        return {
            **state,
            "critique": critique,
            "next_agent": next_agent,
        }

    except json.JSONDecodeError as e:
        print(f"⚠️ 評価結果のパースに失敗: {e}")
        # フォールバック：承認として扱う
        return {
            **state,
            "critique": {
                "scores": {"theme_fit": 7, "word_beauty": 7, "emotional_depth": 7, "rhythm": 7},
                "total_score": 7.0,
                "feedback": "評価の解析に失敗しました",
                "suggestions": "",
                "approved": True,
            },
            "next_agent": "finalizer",
        }


# 4. Finalizer Agent - 最終確定専門
def finalizer_agent(state: PoemState) -> PoemState:
    """最終版を確定するエージェント"""
    print("\n🎉 Finalizer Agent: 詩を確定します")

    return {
        **state,
        "final_poem": state["current_poem"],
        "is_complete": True,
        "next_agent": "__end__",
    }


# 5. Supervisor - 全体を管理するスーパーバイザー
def supervisor_node(state: PoemState) -> PoemState:
    """ワークフロー全体を管理するスーパーバイザー"""
    print(f"\n👔 Supervisor: 次のエージェントは '{state['next_agent']}' です")
    return state


# ===== ルーティング関数 =====
def route_to_next_agent(state: PoemState) -> str:
    """次のエージェントへのルーティングを決定"""
    next_agent = state.get("next_agent", "theme")
    return next_agent


# ===== グラフ構築 =====
def create_poem_multi_agent_graph() -> StateGraph:
    """詩作成マルチエージェントグラフを構築"""
    # グラフの初期化
    workflow = StateGraph(PoemState)

    # ノードの追加
    workflow.add_node("supervisor", supervisor_node)
    workflow.add_node("theme", theme_agent)
    workflow.add_node("composer", composer_agent)
    workflow.add_node("critic", critic_agent)
    workflow.add_node("finalizer", finalizer_agent)

    # エッジの追加
    # 開始 -> スーパーバイザー
    workflow.set_entry_point("supervisor")

    # スーパーバイザーから各エージェントへの条件分岐
    workflow.add_conditional_edges(
        "supervisor",
        route_to_next_agent,
        {
            "theme": "theme",
            "composer": "composer",
            "critic": "critic",
            "finalizer": "finalizer",
            "__end__": END,
        },
    )

    # 各エージェントからスーパーバイザーへ戻る
    workflow.add_edge("theme", "supervisor")
    workflow.add_edge("composer", "supervisor")
    workflow.add_edge("critic", "supervisor")
    workflow.add_edge("finalizer", "supervisor")

    return workflow.compile()


# ===== メイン実行 =====
def main():
    """マルチエージェントシステムを使って詩を作成"""
    print("=" * 70)
    print("🎨 LangGraph マルチエージェント 詩作成システム")
    print("=" * 70)

    # テーマの入力（オプション）
    user_theme = input("\n詩のテーマを入力してください（空白でランダム生成）: ").strip()

    # マルチエージェントグラフを作成
    graph = create_poem_multi_agent_graph()

    # 初期状態を設定
    initial_state: PoemState = {
        "messages": [],
        "theme": user_theme,  # 空の場合はtheme_agentが自動生成
        "current_poem": "",
        "critique": {},
        "improvement_count": 0,
        "next_agent": "theme",
        "final_poem": "",
        "is_complete": False,
    }

    # グラフを実行
    print("\n🤖 詩作成マルチエージェントシステムを起動しています...\n")

    result = graph.invoke(initial_state)

    # 結果を表示
    print("\n" + "=" * 70)
    print("🎉 詩の作成が完了しました！")
    print("=" * 70)
    print(f"\nテーマ: {result['theme']}")
    print(f"\n【完成した詩】")
    print("-" * 40)
    print(result['final_poem'])
    print("-" * 40)

    print(f"\n改善回数: {result['improvement_count']}")

    if result.get("critique"):
        critique = result["critique"]
        print(f"\n最終評価スコア: {critique.get('total_score', 'N/A')}/10.0")
        print(f"  - テーマ適合性: {critique.get('scores', {}).get('theme_fit', 'N/A')}/10")
        print(f"  - 言葉の美しさ: {critique.get('scores', {}).get('word_beauty', 'N/A')}/10")
        print(f"  - 感情の深さ: {critique.get('scores', {}).get('emotional_depth', 'N/A')}/10")
        print(f"  - リズム・韻: {critique.get('scores', {}).get('rhythm', 'N/A')}/10")

    print("=" * 70)


if __name__ == "__main__":
    main()
