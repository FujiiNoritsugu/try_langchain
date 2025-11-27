"""
LangChainでVector SearchとRAGを実装するプログラム
"""

import os
import json
from typing import List
from dotenv import load_dotenv
from google.cloud import aiplatform
from google.cloud.aiplatform import MatchingEngineIndexEndpoint
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_anthropic import ChatAnthropic
from langchain_core.documents import Document
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.retrievers import BaseRetriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv()


class VectorSearchRetriever(BaseRetriever):
    """
    GCP Vector SearchをLangChainのRetrieverとして使用するカスタムクラス
    """

    endpoint_id: str
    deployed_index_id: str
    embeddings: HuggingFaceEmbeddings
    num_neighbors: int = 3
    project_id: str
    location: str = "us-central1"

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun | None = None
    ) -> List[Document]:
        """
        クエリに関連するドキュメントをVector Searchから取得
        """
        # Vertex AI初期化
        aiplatform.init(project=self.project_id, location=self.location)

        # クエリをベクトル化
        query_vector = self.embeddings.embed_query(query)

        # Vector Searchエンドポイントから検索
        index_endpoint = MatchingEngineIndexEndpoint(
            index_endpoint_name=self.endpoint_id
        )

        response = index_endpoint.find_neighbors(
            deployed_index_id=self.deployed_index_id,
            queries=[query_vector],
            num_neighbors=self.num_neighbors,
            return_full_datapoint=True,
        )

        # 結果をLangChainのDocument形式に変換
        documents = []
        neighbors = response[0]

        for neighbor in neighbors:
            # restrictsからツアー情報（JSON）を取得
            tour_data = None
            tour_name = "不明なツアー"
            if hasattr(neighbor, "restricts") and neighbor.restricts:
                for restrict in neighbor.restricts:
                    # Namespaceオブジェクトの属性は'name'と'allow_tokens'
                    if hasattr(restrict, "name") and restrict.name == "tour_data":
                        if hasattr(restrict, "allow_tokens") and restrict.allow_tokens:
                            try:
                                tour_data = json.loads(restrict.allow_tokens[0])
                                tour_name = tour_data.get("name", "不明なツアー")
                            except json.JSONDecodeError:
                                pass
                            break

            # ツアー詳細情報を整形
            if tour_data:
                page_content = f"""ツアー名: {tour_data.get('name', 'N/A')}
場所: {tour_data.get('location', 'N/A')}
時期: {tour_data.get('season', 'N/A')}
説明: {tour_data.get('description', 'N/A')}
見どころ: {', '.join(tour_data.get('highlights', []))}"""
            else:
                page_content = tour_name

            # Documentオブジェクトを作成
            doc = Document(
                page_content=page_content,
                metadata={
                    "tour_id": neighbor.id,
                    "tour_name": tour_name,
                    "similarity_score": float(neighbor.distance),
                },
            )
            documents.append(doc)

        return documents


def format_docs(docs: List[Document]) -> str:
    """ドキュメントをフォーマット"""
    formatted = []
    for idx, doc in enumerate(docs, 1):
        formatted.append(f"【ツアー {idx}】\n{doc.page_content}\n(類似度スコア: {doc.metadata['similarity_score']:.4f})")
    return "\n\n".join(formatted)


def create_rag_chain():
    """
    RAGチェーンを作成（LCEL形式）
    """
    # 環境変数から設定を取得
    project_id = os.getenv("GCP_PROJECT_ID")
    location = os.getenv("GCP_LOCATION", "us-central1")
    endpoint_id = os.getenv("VECTOR_SEARCH_ENDPOINT_ID")
    deployed_index_id = os.getenv("DEPLOYED_INDEX_ID")

    if not all([project_id, endpoint_id, deployed_index_id]):
        raise ValueError(
            "GCP_PROJECT_ID, VECTOR_SEARCH_ENDPOINT_ID, DEPLOYED_INDEX_IDを設定してください"
        )

    # Embeddingsモデルを初期化
    embeddings = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large")

    # カスタムRetrieverを作成
    retriever = VectorSearchRetriever(
        endpoint_id=endpoint_id,
        deployed_index_id=deployed_index_id,
        embeddings=embeddings,
        num_neighbors=3,
        project_id=project_id,
        location=location,
    )

    # LLMを初期化（Anthropic Claude）
    llm = ChatAnthropic(
        model="claude-3-opus-20240229",
        temperature=0.3,
    )

    # プロンプトテンプレートを作成
    prompt = ChatPromptTemplate.from_template(
        """以下の旅行ツアー情報を参考にして、質問に答えてください。

関連するツアー:
{context}

質問: {question}

回答: 上記のツアー情報を踏まえて、おすすめのツアーとその理由を説明してください。"""
    )

    # RAGチェーンを作成（LCEL形式）
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain, retriever


def run_rag_query(query: str):
    """
    RAGクエリを実行
    """
    print(f"\n{'='*60}")
    print(f"質問: {query}")
    print(f"{'='*60}\n")

    # RAGチェーンを作成
    rag_chain, retriever = create_rag_chain()

    # まず、検索されたドキュメントを取得
    source_docs = retriever.invoke(query)

    # RAGチェーンでクエリを実行
    result = rag_chain.invoke(query)

    # 結果を表示
    print("【回答】")
    print(result)

    print(f"\n{'='*60}")
    print("【検索されたソース】")
    print(f"{'='*60}")

    for idx, doc in enumerate(source_docs, 1):
        print(f"\n{idx}. {doc.metadata['tour_name']}")
        print(f"   ツアーID: {doc.metadata['tour_id']}")
        print(f"   類似度スコア: {doc.metadata['similarity_score']:.4f}")
        print(f"\n   詳細:")
        # page_contentを整形して表示
        for line in doc.page_content.split('\n'):
            if line.strip():
                print(f"   {line}")


if __name__ == "__main__":
    import sys

    # クエリを引数から取得（デフォルト値を設定）
    query = sys.argv[1] if len(sys.argv) > 1 else "紅葉が見たいです"

    # RAGクエリを実行
    run_rag_query(query)
