#!/usr/bin/env python3
"""
類似度チェック用MCPサーバ
OpenAI Embeddingsを使って文字列間のコサイン類似度を計算
Chromaベクトルデータベースで既存テーマを事前にベクトル化して保存
"""
import asyncio
import json
import sqlite3
from typing import Any
from pathlib import Path
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

# 環境変数を読み込む
load_dotenv()

# MCPサーバのインスタンスを作成
app = Server("similarity-checker")

# OpenAI Embeddingsのインスタンス
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Chromaベクトルストアのインスタンス（グローバル変数として保持）
vector_store = None

# データベースパス
DB_PATH = Path(__file__).parent / "themes_nature.db"
CHROMA_PERSIST_DIR = Path(__file__).parent / "chroma_db"


def load_themes_from_db():
    """データベースから既存のテーマ名を読み込む"""
    import sys
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT theme_name FROM themes")
        themes = [row[0] for row in cursor.fetchall()]
        conn.close()
        print(f"[MCP Server] DBから{len(themes)}件のテーマを読み込みました", file=sys.stderr)
        return themes
    except Exception as e:
        print(f"[MCP Server] DB読み込みエラー: {e}", file=sys.stderr)
        return []


def initialize_vector_store():
    """ベクトルストアを初期化し、既存テーマをベクトル化して保存"""
    global vector_store
    import sys

    print(f"[MCP Server] ベクトルストアを初期化中...", file=sys.stderr)

    # Chromaベクトルストアを作成（永続化）
    vector_store = Chroma(
        collection_name="themes",
        embedding_function=embeddings,
        persist_directory=str(CHROMA_PERSIST_DIR)
    )

    # 既存のコレクション内のドキュメント数を確認
    existing_count = vector_store._collection.count()
    print(f"[MCP Server] Chromaに保存済みのドキュメント数: {existing_count}", file=sys.stderr)

    # DBから最新のテーマを読み込む
    themes = load_themes_from_db()

    if themes:
        # 既存のテーマ名のセットを取得
        existing_docs = vector_store.get()
        existing_themes = set(existing_docs['documents']) if existing_docs['documents'] else set()

        # 新規テーマのみを追加
        new_themes = [theme for theme in themes if theme not in existing_themes]

        if new_themes:
            print(f"[MCP Server] {len(new_themes)}件の新規テーマをベクトル化中...", file=sys.stderr)
            vector_store.add_texts(
                texts=new_themes,
                metadatas=[{"theme_name": theme} for theme in new_themes]
            )
            print(f"[MCP Server] ベクトル化完了", file=sys.stderr)
        else:
            print(f"[MCP Server] 新規テーマなし（全て登録済み）", file=sys.stderr)

    print(f"[MCP Server] ベクトルストア初期化完了", file=sys.stderr)


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
                    "threshold": {
                        "type": "number",
                        "description": "類似度の閾値（0.0〜1.0）。この値未満ならユニークと判定",
                        "default": 0.7
                    }
                },
                "required": ["candidate"]
            }
        ),
        Tool(
            name="add_theme",
            description="新しいテーマをChromaベクトルストアに追加します",
            inputSchema={
                "type": "object",
                "properties": {
                    "theme": {
                        "type": "string",
                        "description": "追加するテーマ名"
                    }
                },
                "required": ["theme"]
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

        # ベクトルストアが初期化されていない場合は初期化
        if vector_store is None:
            initialize_vector_store()

        if name == "add_theme":
            # 新しいテーマをベクトルストアに追加
            theme = arguments["theme"]
            print(f"[MCP Server] テーマを追加: {theme}", file=sys.stderr)

            vector_store.add_texts(
                texts=[theme],
                metadatas=[{"theme_name": theme}]
            )

            result = {
                "success": True,
                "theme": theme,
                "message": f"テーマ '{theme}' をベクトルストアに追加しました"
            }
            result_json = json.dumps(result, ensure_ascii=False)
            print(f"[MCP Server] 結果: {result_json}", file=sys.stderr)
            return [TextContent(type="text", text=result_json)]

        elif name == "check_similarity":
            candidate = arguments["candidate"]
            threshold = arguments.get("threshold", 0.7)

            print(f"[MCP Server] 候補: {candidate}", file=sys.stderr)

            # Chromaに保存されているドキュメント数を確認
            doc_count = vector_store._collection.count()
            print(f"[MCP Server] ベクトルストア内のドキュメント数: {doc_count}", file=sys.stderr)

            # 既存テーマがない場合
            if doc_count == 0:
                result = {
                    "max_similarity": 0.0,
                    "most_similar_text": None,
                    "is_unique": True,
                    "message": "既存テーマがないため、ユニークと判定"
                }
                return [TextContent(type="text", text=json.dumps(result, ensure_ascii=False))]

            # 類似度検索を実行（k=1で最も類似したものを取得、score付き）
            print(f"[MCP Server] 類似度検索を実行中...", file=sys.stderr)
            results = vector_store.similarity_search_with_score(candidate, k=1)
            print(f"[MCP Server] 検索完了", file=sys.stderr)

            if results:
                doc, score = results[0]
                most_similar_text = doc.page_content
                # Chromaのスコアは距離（小さいほど類似）なので、類似度に変換
                # L2距離をコサイン類似度に変換（近似）
                # スコアが小さいほど類似なので、1 - normalized_score を使用
                # ただし、Chromaのスコアは実装依存なので、そのまま使用
                # 正確なコサイン類似度を取得するため、手動で計算
                from sklearn.metrics.pairwise import cosine_similarity
                import numpy as np

                candidate_embedding = embeddings.embed_query(candidate)
                similar_embedding = embeddings.embed_query(most_similar_text)

                candidate_vector = np.array(candidate_embedding).reshape(1, -1)
                similar_vector = np.array(similar_embedding).reshape(1, -1)

                max_similarity = float(cosine_similarity(candidate_vector, similar_vector)[0][0])

                # 結果を返す
                result = {
                    "max_similarity": max_similarity,
                    "most_similar_text": most_similar_text,
                    "is_unique": max_similarity < threshold,
                    "threshold": threshold
                }
            else:
                result = {
                    "max_similarity": 0.0,
                    "most_similar_text": None,
                    "is_unique": True,
                    "message": "類似テーマが見つかりませんでした"
                }

            result_json = json.dumps(result, ensure_ascii=False)
            print(f"[MCP Server] 結果: {result_json}", file=sys.stderr)
            return [TextContent(type="text", text=result_json)]

        else:
            raise ValueError(f"Unknown tool: {name}")

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
    import sys

    # サーバ起動時にベクトルストアを初期化
    print(f"[MCP Server] サーバを起動しています...", file=sys.stderr)
    initialize_vector_store()

    async with stdio_server() as (read_stream, write_stream):
        await app.run(
            read_stream,
            write_stream,
            app.create_initialization_options()
        )


if __name__ == "__main__":
    asyncio.run(main())
