"""
LangChainでVector SearchとRAGを実装するプログラム
"""

import os
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
            # restrictsからツアー名を取得
            tour_name = "不明なツアー"
            if hasattr(neighbor, "restricts") and neighbor.restricts:
                for restrict in neighbor.restricts:
                    if restrict.namespace == "tour_name" and restrict.allow_list:
                        tour_name = restrict.allow_list[0]
                        break

            # Documentオブジェクトを作成
            doc = Document(
                page_content=tour_name,
                metadata={
                    "tour_id": neighbor.id,
                    "similarity_score": float(neighbor.distance),
                },
            )
            documents.append(doc)

        return documents


def format_docs(docs: List[Document]) -> str:
    """ドキュメントをフォーマット"""
    return "\n\n".join(
        [
            f"- {doc.page_content} (ツアーID: {doc.metadata['tour_id']}, 類似度: {doc.metadata['similarity_score']:.4f})"
            for doc in docs
        ]
    )


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
        print(f"\n{idx}. {doc.page_content}")
        print(f"   ツアーID: {doc.metadata['tour_id']}")
        print(f"   類似度スコア: {doc.metadata['similarity_score']:.4f}")


if __name__ == "__main__":
    import sys

    # クエリを引数から取得（デフォルト値を設定）
    query = sys.argv[1] if len(sys.argv) > 1 else "紅葉が見たいです"

    # RAGクエリを実行
    run_rag_query(query)
