"""
ColBERTを使用したRAG実装

ColBERTは密ベクトルではなくトークンレベルの疎な表現を使用し、
MaxSim演算によって高精度な検索を実現します。

RAGatouille（ragatouille）ライブラリを使用してColBERTを統合。

使用方法:
1. RAGatouilleをインストール:
   pip install ragatouille

2. ColBERTインデックスを初期化:
   python rag_colbert.py --init

3. RAGクエリを実行:
   python rag_colbert.py "紅葉が見たいです"

4. MultiVectorモードで実行（要約で検索、親ドキュメントを返す）:
   python rag_colbert.py --multi-vector "アクティビティが楽しいツアーは?"
"""

import os
import json
from typing import List, Optional
from dotenv import load_dotenv

# ColBERT関連のインポート
try:
    from ragatouille import RAGPretrainedModel
    RAGATOUILLE_AVAILABLE = True
except ImportError:
    RAGATOUILLE_AVAILABLE = False
    print("警告: ragatouilleがインストールされていません")
    print("インストール方法: pip install ragatouille")

from langchain_anthropic import ChatAnthropic
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.retrievers import BaseRetriever
from langchain_core.stores import InMemoryStore

load_dotenv()


def load_tour_data(tour_data_file: str = "tour_data.json") -> List[dict]:
    """ツアーデータをJSONファイルから読み込む"""
    if not os.path.exists(tour_data_file):
        raise FileNotFoundError(f"{tour_data_file}が見つかりません")

    with open(tour_data_file, 'r', encoding='utf-8') as f:
        tours = json.load(f)

    return tours


def create_parent_documents(tours: List[dict]) -> List[Document]:
    """親ドキュメント（完全な情報）を作成"""
    parent_docs = []

    for tour in tours:
        tour_id = tour.get('id', tour.get('name', ''))

        # 親ドキュメントのコンテンツ（詳細な情報を含む）
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
                "doc_id": tour_id,
            }
        )

        parent_docs.append(parent_doc)

    return parent_docs


def generate_summaries(parent_docs: List[Document]) -> List[Document]:
    """親ドキュメントから要約（子ドキュメント）を生成"""
    llm = ChatAnthropic(
        model="claude-3-haiku-20240307",
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

    summary_chain = summary_prompt | llm | StrOutputParser()
    child_docs = []

    print("要約を生成中...")
    for parent_doc in parent_docs:
        summary = summary_chain.invoke({"tour_info": parent_doc.page_content})

        child_doc = Document(
            page_content=summary.strip(),
            metadata={
                "doc_id": parent_doc.metadata["doc_id"],
                "tour_name": parent_doc.metadata["tour_name"],
                "type": "summary"
            }
        )

        child_docs.append(child_doc)
        print(f"  生成完了: {parent_doc.metadata['tour_name']}")

    return child_docs


class ColBERTRetriever(BaseRetriever):
    """
    ColBERTを使用したカスタムRetriever

    RAGatouille (ragatouille) を使用してColBERTモデルを統合
    """

    colbert_model: RAGPretrainedModel
    num_results: int = 3

    class Config:
        arbitrary_types_allowed = True

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun | None = None
    ) -> List[Document]:
        """
        ColBERTを使用してクエリに関連するドキュメントを取得
        """
        # ColBERTで検索
        results = self.colbert_model.search(query, k=self.num_results)

        # 結果をLangChainのDocument形式に変換
        documents = []
        for result in results:
            doc = Document(
                page_content=result["content"],
                metadata={
                    "score": result["score"],
                    "doc_id": result.get("document_id", ""),
                    **result.get("document_metadata", {})
                }
            )
            documents.append(doc)

        return documents


class MultiVectorColBERTRetriever(BaseRetriever):
    """
    ColBERTとdocstoreを組み合わせたMultiVectorRetriever

    子ドキュメント（要約）で検索し、親ドキュメント（完全な情報）を返す
    """

    colbert_retriever: ColBERTRetriever
    docstore: InMemoryStore
    id_key: str = "doc_id"

    class Config:
        arbitrary_types_allowed = True

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun | None = None
    ) -> List[Document]:
        """
        クエリに関連する親ドキュメントを取得
        """
        # ColBERTで子ドキュメントを検索
        child_docs = self.colbert_retriever._get_relevant_documents(query, run_manager=run_manager)

        # 親ドキュメントのIDを取得（重複を除く）
        parent_ids = []
        seen_ids = set()
        for doc in child_docs:
            parent_id = doc.metadata.get(self.id_key)
            if parent_id and parent_id not in seen_ids:
                parent_ids.append(parent_id)
                seen_ids.add(parent_id)

        # docstoreから親ドキュメントを取得
        parent_docs = []
        for parent_id in parent_ids:
            parent_doc = self.docstore.mget([parent_id])[0]
            if parent_doc:
                parent_docs.append(parent_doc)

        return parent_docs


def initialize_colbert_index(
    index_name: str = "tour_colbert_index",
    use_multi_vector: bool = False
) -> None:
    """
    ColBERTインデックスを初期化

    Args:
        index_name: インデックス名
        use_multi_vector: MultiVectorモード（要約で検索）を使用するか
    """
    if not RAGATOUILLE_AVAILABLE:
        raise ImportError("ragatouille がインストールされていません")

    # ツアーデータを読み込み
    tours = load_tour_data()

    # 親ドキュメントを作成
    parent_docs = create_parent_documents(tours)

    # インデックス化するドキュメントを選択
    if use_multi_vector:
        # MultiVectorモード: 要約をインデックス化
        docs_to_index = generate_summaries(parent_docs)
        print(f"\nMultiVectorモード: 要約をインデックス化します")
    else:
        # 通常モード: 親ドキュメントをインデックス化
        docs_to_index = parent_docs
        print(f"\n通常モード: 親ドキュメントをインデックス化します")

    # ColBERTモデルを初期化
    print("\nColBERTモデルを初期化中...")
    colbert = RAGPretrainedModel.from_pretrained("colbert-ir/colbertv2.0")

    # ドキュメントをColBERT形式に変換
    documents = [doc.page_content for doc in docs_to_index]
    document_ids = [doc.metadata["doc_id"] for doc in docs_to_index]
    document_metadatas = [
        {
            "tour_id": doc.metadata["doc_id"],
            "tour_name": doc.metadata["tour_name"],
        }
        for doc in docs_to_index
    ]

    # インデックスを作成
    print(f"\nColBERTインデックスを作成中（{len(documents)}件のドキュメント）...")
    colbert.index(
        collection=documents,
        document_ids=document_ids,
        document_metadatas=document_metadatas,
        index_name=index_name,
        max_document_length=256,
        split_documents=True,
    )

    print(f"\nColBERTインデックス'{index_name}'の初期化が完了しました")
    print(f"  インデックス化されたドキュメント数: {len(documents)}")


def load_colbert_retriever(
    index_name: str = "tour_colbert_index",
    num_results: int = 3
) -> ColBERTRetriever:
    """保存済みのColBERTインデックスを読み込む"""
    if not RAGATOUILLE_AVAILABLE:
        raise ImportError("ragatouille がインストールされていません")

    # ColBERTモデルを読み込み
    print(f"ColBERTインデックス'{index_name}'を読み込み中...")
    colbert = RAGPretrainedModel.from_index(index_name)

    # ColBERTRetrieverを作成
    retriever = ColBERTRetriever(
        colbert_model=colbert,
        num_results=num_results,
    )

    return retriever


def load_multi_vector_colbert_retriever(
    index_name: str = "tour_colbert_index",
    num_results: int = 3
) -> MultiVectorColBERTRetriever:
    """MultiVectorモードのColBERTRetrieverを読み込む"""
    # ColBERTRetrieverを読み込み
    colbert_retriever = load_colbert_retriever(index_name, num_results)

    # docstoreを再構築
    tours = load_tour_data()
    parent_docs = create_parent_documents(tours)

    docstore = InMemoryStore()
    doc_ids = [doc.metadata["doc_id"] for doc in parent_docs]
    docstore.mset(list(zip(doc_ids, parent_docs)))

    # MultiVectorColBERTRetrieverを作成
    retriever = MultiVectorColBERTRetriever(
        colbert_retriever=colbert_retriever,
        docstore=docstore,
        id_key="doc_id",
    )

    return retriever


def format_docs(docs: List[Document]) -> str:
    """ドキュメントをフォーマット"""
    formatted = []
    for idx, doc in enumerate(docs, 1):
        score = doc.metadata.get('score', 'N/A')
        score_text = f"{score:.4f}" if isinstance(score, (int, float)) else score
        formatted.append(f"【ツアー {idx}】\n{doc.page_content}\n(スコア: {score_text})")
    return "\n\n".join(formatted)


def create_rag_chain(retriever: BaseRetriever):
    """RAGチェーンを作成"""
    # LLMを初期化
    llm = ChatAnthropic(
        model="claude-3-opus-20240229",
        temperature=0.3,
    )

    # プロンプトテンプレート
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

    return rag_chain


def run_rag_query(
    query: str,
    index_name: str = "tour_colbert_index",
    use_multi_vector: bool = False
):
    """RAGクエリを実行"""
    print(f"\n{'='*60}")
    print(f"質問: {query}")
    print(f"モード: {'MultiVector ColBERT' if use_multi_vector else 'ColBERT'}")
    print(f"{'='*60}\n")

    # Retrieverを読み込み
    if use_multi_vector:
        retriever = load_multi_vector_colbert_retriever(index_name)
    else:
        retriever = load_colbert_retriever(index_name)

    # RAGチェーンを作成
    rag_chain = create_rag_chain(retriever)

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
        tour_name = doc.metadata.get('tour_name', '不明なツアー')
        tour_id = doc.metadata.get('tour_id', doc.metadata.get('doc_id', 'N/A'))

        print(f"\n{idx}. {tour_name}")
        print(f"   ツアーID: {tour_id}")

        if 'score' in doc.metadata:
            print(f"   スコア: {doc.metadata['score']:.4f}")

        print(f"\n   詳細:")
        for line in doc.page_content.split('\n'):
            if line.strip():
                print(f"   {line}")


if __name__ == "__main__":
    import sys

    # ragatouille の可用性をチェック
    if not RAGATOUILLE_AVAILABLE:
        print("\nエラー: ragatouille がインストールされていません")
        print("インストール方法:")
        print("  pip install ragatouille")
        sys.exit(1)

    # コマンドライン引数の解析
    if len(sys.argv) > 1 and sys.argv[1] == "--init":
        # 初期化モード
        use_multi_vector = False
        index_name = "tour_colbert_index"

        # MultiVectorモードのチェック
        if len(sys.argv) > 2 and sys.argv[2] == "--multi-vector":
            use_multi_vector = True
            index_name = "tour_colbert_index_mv"

        print(f"ColBERTインデックスを初期化します...")
        initialize_colbert_index(index_name, use_multi_vector)
        print("\n初期化完了！")

    elif len(sys.argv) > 1 and sys.argv[1] == "--multi-vector":
        # MultiVectorモードでクエリ実行
        query = sys.argv[2] if len(sys.argv) > 2 else "紅葉が見たいです"
        run_rag_query(query, index_name="tour_colbert_index_mv", use_multi_vector=True)

    else:
        # クエリを引数から取得
        query = sys.argv[1] if len(sys.argv) > 1 else "紅葉が見たいです"

        # RAGクエリを実行（通常モード）
        run_rag_query(query)
