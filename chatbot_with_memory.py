"""
LangGraphを使用したメモリ付きチャットボット

このプログラムは、LangGraphのStateGraphとCheckpointSaverを使用して
会話履歴を保持するチャットボットを実装します。
"""

from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv

load_dotenv()

# Stateの定義
class State(TypedDict):
    messages: Annotated[list, add_messages]

# チャットボットノード
def chatbot(state: State):
    """LLMを呼び出してレスポンスを生成"""
    llm = ChatAnthropic(model="claude-3-haiku-20240307", temperature=0.7)
    response = llm.invoke(state["messages"])
    return {"messages": [response]}

# グラフの構築
def create_chatbot_graph():
    """メモリ付きチャットボットグラフを作成"""
    # StateGraphの作成
    graph_builder = StateGraph(State)

    # ノードの追加
    graph_builder.add_node("chatbot", chatbot)

    # エッジの追加
    graph_builder.add_edge(START, "chatbot")
    graph_builder.add_edge("chatbot", END)

    # メモリの追加
    memory = MemorySaver()

    # グラフのコンパイル
    graph = graph_builder.compile(checkpointer=memory)

    return graph

def main():
    """チャットボットの実行"""
    graph = create_chatbot_graph()

    # スレッドIDを使用して会話を管理
    config = {"configurable": {"thread_id": "1"}}

    print("チャットボット起動（'quit'で終了）\n")

    while True:
        user_input = input("You: ")
        if user_input.lower() in ["quit", "exit", "q"]:
            print("チャットを終了します。")
            break

        # ユーザーメッセージを送信
        events = graph.stream(
            {"messages": [("user", user_input)]},
            config,
            stream_mode="values"
        )

        # レスポンスを表示
        for event in events:
            if "messages" in event:
                event["messages"][-1].pretty_print()

if __name__ == "__main__":
    main()
