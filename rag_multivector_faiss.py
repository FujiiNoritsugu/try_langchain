"""
LangChainのMultiVectorRetrieverとFAISSを使ったRAG実装

FAISS/Chromaなどの標準ベクトルストアを使用し、
子ドキュメント（要約、仮想質問）で検索して親ドキュメントを返す。

使用方法:
1. ベクトルストアとdocstoreを初期化:
   python rag_multivector_faiss.py --init

2. RAGクエリを実行:
   python rag_multivector_faiss.py "紅葉が見たいです"

3. 要約モードと仮想質問モードを切り替え:
   python rag_multivector_faiss.py --mode hypothetical "アクティビティが楽しいツアーは?"
"""

import os
import json
import uuid
from typing import List
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_anthropic import ChatAnthropic
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.storage import InMemoryStore

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
                "doc_id": str(uuid.uuid4()),  # ユニークなID
            }
        )

        parent_docs.append(parent_doc)

    return parent_docs


def generate_summaries(parent_docs: List[Document]) -> List[Document]:
    """親ドキュメントから要約（子ドキュメント）を生成"""
    # LLMを初期化（要約には軽量モデル）
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
                "type": "summary"
            }
        )

        child_docs.append(child_doc)
        print(f"  生成完了: {parent_doc.metadata['tour_name']}")

    return child_docs


def generate_hypothetical_questions(parent_docs: List[Document]) -> List[Document]:
    """親ドキュメントから仮想質問（子ドキュメント）を生成"""
    # LLMを初期化
    llm = ChatAnthropic(
        model="claude-3-haiku-20240307",
        temperature=0.3,
    )

    # 仮想質問プロンプト
    question_prompt = ChatPromptTemplate.from_template(
        """以下の旅行ツアー情報に対して、ユーザーが尋ねそうな質問を3つ生成してください。
各質問は1行で、改行で区切ってください。

ツアー情報:
{tour_info}

質問:"""
    )

    question_chain = question_prompt | llm | StrOutputParser()
    child_docs = []

    print("仮想質問を生成中...")
    for parent_doc in parent_docs:
        questions_text = question_chain.invoke({"tour_info": parent_doc.page_content})

        # 複数の質問を分割
        questions = [q.strip() for q in questions_text.split('\n') if q.strip()]

        for question in questions:
            child_doc = Document(
                page_content=question,
                metadata={
                    "doc_id": parent_doc.metadata["doc_id"],
                    "type": "hypothetical_question"
                }
            )
            child_docs.append(child_doc)

        print(f"  生成完了: {parent_doc.metadata['tour_name']} ({len(questions)}個の質問)")

    return child_docs


def initialize_multi_vector_retriever(
    mode: str = "summary",
    persist_dir: str = "./faiss_index"
) -> MultiVectorRetriever:
    """
    MultiVectorRetrieverを初期化

    Args:
        mode: "summary" (要約モード) または "hypothetical" (仮想質問モード)
        persist_dir: FAISSインデックスの保存ディレクトリ
    """
    # ツアーデータを読み込み
    tours = load_tour_data()

    # 親ドキュメントを作成
    parent_docs = create_parent_documents(tours)

    # 子ドキュメントを生成（モードに応じて）
    if mode == "summary":
        child_docs = generate_summaries(parent_docs)
    elif mode == "hypothetical":
        child_docs = generate_hypothetical_questions(parent_docs)
    else:
        raise ValueError(f"未対応のモード: {mode}")

    # Embeddingsモデルを初期化
    embeddings = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large")

    # ベクトルストアを作成（FAISSを使用）
    vectorstore = FAISS.from_documents(child_docs, embeddings)

    # ベクトルストアを保存
    os.makedirs(persist_dir, exist_ok=True)
    vectorstore.save_local(persist_dir)
    print(f"\nベクトルストアを{persist_dir}に保存しました")

    # docstore（親ドキュメント用）を作成
    docstore = InMemoryStore()
    doc_ids = [doc.metadata["doc_id"] for doc in parent_docs]
    docstore.mset(list(zip(doc_ids, parent_docs)))

    # MultiVectorRetrieverを作成
    retriever = MultiVectorRetriever(
        vectorstore=vectorstore,
        docstore=docstore,
        id_key="doc_id",
    )

    print(f"MultiVectorRetriever初期化完了（モード: {mode}）")
    print(f"  親ドキュメント数: {len(parent_docs)}")
    print(f"  子ドキュメント数: {len(child_docs)}")

    return retriever


def load_multi_vector_retriever(persist_dir: str = "./faiss_index") -> MultiVectorRetriever:
    """保存済みのMultiVectorRetrieverを読み込む"""
    if not os.path.exists(persist_dir):
        raise FileNotFoundError(
            f"{persist_dir}が見つかりません。先に --init で初期化してください"
        )

    # Embeddingsモデルを初期化
    embeddings = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large")

    # ベクトルストアを読み込み
    vectorstore = FAISS.load_local(
        persist_dir,
        embeddings,
        allow_dangerous_deserialization=True
    )

    # docstoreを再構築
    tours = load_tour_data()
    parent_docs = create_parent_documents(tours)

    docstore = InMemoryStore()
    doc_ids = [doc.metadata["doc_id"] for doc in parent_docs]
    docstore.mset(list(zip(doc_ids, parent_docs)))

    # MultiVectorRetrieverを作成
    retriever = MultiVectorRetriever(
        vectorstore=vectorstore,
        docstore=docstore,
        id_key="doc_id",
    )

    return retriever


def format_docs(docs: List[Document]) -> str:
    """ドキュメントをフォーマット"""
    formatted = []
    for idx, doc in enumerate(docs, 1):
        formatted.append(f"【ツアー {idx}】\n{doc.page_content}")
    return "\n\n".join(formatted)


def create_rag_chain(retriever: MultiVectorRetriever):
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


def run_rag_query(query: str, persist_dir: str = "./faiss_index"):
    """RAGクエリを実行"""
    print(f"\n{'='*60}")
    print(f"質問: {query}")
    print(f"{'='*60}\n")

    # MultiVectorRetrieverを読み込み
    retriever = load_multi_vector_retriever(persist_dir)

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
        print(f"\n{idx}. {doc.metadata['tour_name']}")
        print(f"   ツアーID: {doc.metadata['tour_id']}")
        print(f"\n   詳細:")
        for line in doc.page_content.split('\n'):
            if line.strip():
                print(f"   {line}")


if __name__ == "__main__":
    import sys

    # コマンドライン引数の解析
    if len(sys.argv) > 1 and sys.argv[1] == "--init":
        # 初期化モード
        mode = "summary"  # デフォルトは要約モード
        if len(sys.argv) > 2 and sys.argv[2] == "--mode":
            mode = sys.argv[3] if len(sys.argv) > 3 else "summary"

        print(f"MultiVectorRetrieverを初期化します（モード: {mode}）...")
        initialize_multi_vector_retriever(mode=mode)
        print("\n初期化完了！")

    elif len(sys.argv) > 1 and sys.argv[1] == "--mode":
        # モード指定でクエリ実行
        mode = sys.argv[2] if len(sys.argv) > 2 else "summary"
        persist_dir = f"./faiss_index_{mode}"

        # まず初期化
        print(f"MultiVectorRetrieverを初期化します（モード: {mode}）...")
        initialize_multi_vector_retriever(mode=mode, persist_dir=persist_dir)

        # クエリを取得
        query = sys.argv[3] if len(sys.argv) > 3 else "紅葉が見たいです"

        # RAGクエリを実行
        run_rag_query(query, persist_dir=persist_dir)

    else:
        # クエリを引数から取得
        query = sys.argv[1] if len(sys.argv) > 1 else "紅葉が見たいです"

        # RAGクエリを実行
        run_rag_query(query)
