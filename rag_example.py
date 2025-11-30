"""
LangChainでVector SearchとRAGを実装するプログラム

MultiVectorRetrieverを使用して、子ドキュメント（要約）で検索し、
親ドキュメント（完全な情報）を返すことができます。

使用方法:
1. 子ドキュメント（要約）を生成:
   python rag_example.py --generate-summaries

2. RAGクエリを実行（MultiVectorRetriever使用）:
   python rag_example.py "紅葉が見たいです"

3. RAGクエリを実行（通常Retriever使用）:
   python rag_example.py --no-multi-vector "紅葉が見たいです"

注意:
- MultiVectorRetrieverを使用する場合、tour_data.jsonが必要です
- 子ドキュメントの要約はVector Searchに登録する必要があります
"""

import os
import json
from typing import List, Dict
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
from langchain.storage import InMemoryStore

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
                    "parent_id": neighbor.id,  # MultiVectorRetriever用
                },
            )
            documents.append(doc)

        return documents


class MultiVectorSearchRetriever(BaseRetriever):
    """
    GCP Vector Searchと親ドキュメントストアを組み合わせたMultiVectorRetriever
    子ドキュメント（要約など）で検索し、親ドキュメント（完全な情報）を返す
    """

    vector_retriever: VectorSearchRetriever
    docstore: InMemoryStore
    id_key: str = "parent_id"

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun | None = None
    ) -> List[Document]:
        """
        クエリに関連する親ドキュメントを取得
        1. Vector Searchで子ドキュメントを検索
        2. 子ドキュメントのparent_idから親ドキュメントを取得
        """
        # Vector Searchで子ドキュメントを検索
        child_docs = self.vector_retriever._get_relevant_documents(query, run_manager=run_manager)

        # 親ドキュメントのIDを取得（重複を除く）
        parent_ids = []
        for doc in child_docs:
            parent_id = doc.metadata.get(self.id_key)
            if parent_id and parent_id not in parent_ids:
                parent_ids.append(parent_id)

        # docstoreから親ドキュメントを取得
        parent_docs = []
        for parent_id in parent_ids:
            parent_doc = self.docstore.mget([parent_id])[0]
            if parent_doc:
                parent_docs.append(parent_doc)

        return parent_docs


def initialize_docstore(tour_data_file: str = "tour_data.json") -> InMemoryStore:
    """
    親ドキュメント用のdocstoreを初期化
    ツアーデータのJSONファイルから親ドキュメントを読み込む
    """
    docstore = InMemoryStore()

    # ツアーデータファイルが存在する場合は読み込み
    if os.path.exists(tour_data_file):
        with open(tour_data_file, 'r', encoding='utf-8') as f:
            tours = json.load(f)

        # 各ツアーを親ドキュメントとしてdocstoreに保存
        for tour in tours:
            tour_id = tour.get('id', tour.get('name', ''))

            # 親ドキュメントのコンテンツを作成（詳細な情報を含む）
            page_content = f"""ツアー名: {tour.get('name', 'N/A')}
場所: {tour.get('location', 'N/A')}
時期: {tour.get('season', 'N/A')}
説明: {tour.get('description', 'N/A')}
見どころ: {', '.join(tour.get('highlights', []))}
価格: {tour.get('price', 'N/A')}
期間: {tour.get('duration', 'N/A')}
難易度: {tour.get('difficulty', 'N/A')}"""

            parent_doc = Document(
                page_content=page_content,
                metadata={
                    "tour_id": tour_id,
                    "tour_name": tour.get('name', 'N/A'),
                    "type": "parent_document"
                }
            )

            docstore.mset([(tour_id, parent_doc)])

    return docstore


def format_docs(docs: List[Document]) -> str:
    """ドキュメントをフォーマット"""
    formatted = []
    for idx, doc in enumerate(docs, 1):
        # similarity_scoreがメタデータにない場合の対応
        score = doc.metadata.get('similarity_score', 'N/A')
        score_text = f"{score:.4f}" if isinstance(score, (int, float)) else score
        formatted.append(f"【ツアー {idx}】\n{doc.page_content}\n(類似度スコア: {score_text})")
    return "\n\n".join(formatted)


def create_rag_chain(use_multi_vector: bool = True):
    """
    RAGチェーンを作成（LCEL形式）

    Args:
        use_multi_vector: MultiVectorRetrieverを使用するかどうか（デフォルト: True）
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

    # VectorSearchRetrieverを作成
    vector_retriever = VectorSearchRetriever(
        endpoint_id=endpoint_id,
        deployed_index_id=deployed_index_id,
        embeddings=embeddings,
        num_neighbors=3,
        project_id=project_id,
        location=location,
    )

    # MultiVectorRetrieverを使用する場合
    if use_multi_vector:
        # docstoreを初期化
        docstore = initialize_docstore()

        # MultiVectorSearchRetrieverを作成
        retriever = MultiVectorSearchRetriever(
            vector_retriever=vector_retriever,
            docstore=docstore,
            id_key="parent_id"
        )
    else:
        retriever = vector_retriever

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


def run_rag_query(query: str, use_multi_vector: bool = True):
    """
    RAGクエリを実行

    Args:
        query: 検索クエリ
        use_multi_vector: MultiVectorRetrieverを使用するかどうか
    """
    print(f"\n{'='*60}")
    print(f"質問: {query}")
    print(f"モード: {'MultiVectorRetriever' if use_multi_vector else '通常Retriever'}")
    print(f"{'='*60}\n")

    # RAGチェーンを作成
    rag_chain, retriever = create_rag_chain(use_multi_vector=use_multi_vector)

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


def generate_child_documents(tour_data_file: str = "tour_data.json",
                             output_file: str = "child_docs.json"):
    """
    親ドキュメントから子ドキュメント（要約）を生成
    これらの要約をVector Searchに登録することで、MultiVectorRetrieverが機能する

    Args:
        tour_data_file: ツアーデータのJSONファイルパス
        output_file: 生成された子ドキュメントの出力ファイルパス
    """
    # LLMを初期化
    llm = ChatAnthropic(
        model="claude-3-haiku-20240307",  # 要約には軽量モデルを使用
        temperature=0.3,
    )

    # 要約プロンプト
    summary_prompt = ChatPromptTemplate.from_template(
        """以下の旅行ツアー情報を簡潔に要約してください。
要約は検索用に最適化し、主要な特徴とキーワードを含めてください。

ツアー情報:
{tour_info}

要約（100-150文字程度）:"""
    )

    # ツアーデータを読み込み
    if not os.path.exists(tour_data_file):
        raise FileNotFoundError(f"{tour_data_file}が見つかりません")

    with open(tour_data_file, 'r', encoding='utf-8') as f:
        tours = json.load(f)

    child_docs = []

    print(f"子ドキュメント（要約）を生成中...")

    for tour in tours:
        tour_id = tour.get('id', tour.get('name', ''))

        # 親ドキュメントの情報
        tour_info = f"""ツアー名: {tour.get('name', 'N/A')}
場所: {tour.get('location', 'N/A')}
時期: {tour.get('season', 'N/A')}
説明: {tour.get('description', 'N/A')}
見どころ: {', '.join(tour.get('highlights', []))}"""

        # LLMで要約を生成
        summary_chain = summary_prompt | llm | StrOutputParser()
        summary = summary_chain.invoke({"tour_info": tour_info})

        # 子ドキュメントを作成
        child_doc = {
            "id": f"{tour_id}_summary",
            "parent_id": tour_id,
            "content": summary.strip(),
            "type": "summary"
        }

        child_docs.append(child_doc)
        print(f"  生成完了: {tour.get('name', tour_id)}")

    # 子ドキュメントをJSONファイルに保存
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(child_docs, f, ensure_ascii=False, indent=2)

    print(f"\n子ドキュメントを{output_file}に保存しました")
    print(f"これらの要約をVector Searchに登録してください")

    return child_docs


if __name__ == "__main__":
    import sys

    # コマンドライン引数で動作を切り替え
    if len(sys.argv) > 1 and sys.argv[1] == "--generate-summaries":
        # 子ドキュメント（要約）を生成
        generate_child_documents()
    elif len(sys.argv) > 1 and sys.argv[1] == "--no-multi-vector":
        # MultiVectorRetrieverを使わない通常モード
        query = sys.argv[2] if len(sys.argv) > 2 else "紅葉が見たいです"
        run_rag_query(query, use_multi_vector=False)
    else:
        # クエリを引数から取得（デフォルト値を設定）
        query = sys.argv[1] if len(sys.argv) > 1 else "紅葉が見たいです"

        # RAGクエリを実行（MultiVectorRetrieverを使用）
        run_rag_query(query, use_multi_vector=True)
