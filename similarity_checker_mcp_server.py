#!/usr/bin/env python3
"""
類似度チェック用MCPサーバ
VoyageAI Embeddingsを使って文字列間のコサイン類似度を計算
"""
import asyncio
import json
from typing import Any
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from langchain_voyageai import VoyageAIEmbeddings
from dotenv import load_dotenv

# 環境変数を読み込む
load_dotenv()

# MCPサーバのインスタンスを作成
app = Server("similarity-checker")

# VoyageAI Embeddingsのインスタンス
embeddings = VoyageAIEmbeddings(model="voyage-3-lite")


@app.list_tools()
async def list_tools() -> list[Tool]:
    """利用可能なツールのリストを返す"""
    return [
        Tool(
            name="check_similarity",
            description="候補テキストと既存テキストリストとのコサイン類似度を計算します。最大類似度と最も類似したテキストを返します。",
            inputSchema={
                "type": "object",
                "properties": {
                    "candidate": {
                        "type": "string",
                        "description": "類似度をチェックする候補テキスト"
                    },
                    "existing_texts": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "既存のテキストリスト"
                    },
                    "threshold": {
                        "type": "number",
                        "description": "類似度の閾値（0.0〜1.0）。この値未満ならユニークと判定",
                        "default": 0.7
                    }
                },
                "required": ["candidate", "existing_texts"]
            }
        )
    ]


@app.call_tool()
async def call_tool(name: str, arguments: Any) -> list[TextContent]:
    """ツールを実行"""
    try:
        import sys
        print(f"[MCP Server] ツール呼び出し: {name}", file=sys.stderr)
        print(f"[MCP Server] 引数: {arguments}", file=sys.stderr)

        if name != "check_similarity":
            raise ValueError(f"Unknown tool: {name}")

        candidate = arguments["candidate"]
        existing_texts = arguments["existing_texts"]
        threshold = arguments.get("threshold", 0.7)

        print(f"[MCP Server] 候補: {candidate}", file=sys.stderr)
        print(f"[MCP Server] 既存テキスト数: {len(existing_texts)}", file=sys.stderr)

        # 既存テキストが空の場合
        if not existing_texts:
            result = {
                "max_similarity": 0.0,
                "most_similar_text": None,
                "is_unique": True,
                "message": "既存テキストがないため、ユニークと判定"
            }
            return [TextContent(type="text", text=json.dumps(result, ensure_ascii=False))]

        # 埋め込みを取得
        print(f"[MCP Server] 埋め込みを取得中...", file=sys.stderr)
        candidate_embedding = embeddings.embed_query(candidate)
        existing_embeddings = embeddings.embed_documents(existing_texts)
        print(f"[MCP Server] 埋め込み取得完了", file=sys.stderr)

        # コサイン類似度を計算
        candidate_vector = np.array(candidate_embedding).reshape(1, -1)
        existing_vectors = np.array(existing_embeddings)

        similarities = cosine_similarity(candidate_vector, existing_vectors)[0]
        max_similarity = float(np.max(similarities))
        most_similar_idx = int(np.argmax(similarities))
        most_similar_text = existing_texts[most_similar_idx]

        # 結果を返す
        result = {
            "max_similarity": max_similarity,
            "most_similar_text": most_similar_text,
            "is_unique": max_similarity < threshold,
            "threshold": threshold,
            "all_similarities": similarities.tolist()
        }

        result_json = json.dumps(result, ensure_ascii=False)
        print(f"[MCP Server] 結果: {result_json}", file=sys.stderr)
        return [TextContent(type="text", text=result_json)]

    except Exception as e:
        import sys
        import traceback
        error_msg = f"Error in call_tool: {e}"
        print(f"[MCP Server] エラー: {error_msg}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        # エラーをクライアントに返す
        raise


async def main():
    """MCPサーバを起動"""
    async with stdio_server() as (read_stream, write_stream):
        await app.run(
            read_stream,
            write_stream,
            app.create_initialization_options()
        )


if __name__ == "__main__":
    asyncio.run(main())
