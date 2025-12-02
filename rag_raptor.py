"""
RAPTOR (Recursive Abstractive Processing for Tree-Organized Retrieval) Implementation

階層的要約を使用したRAG実装:
- レベル0: 元の親ドキュメント（詳細情報）
- レベル1: 第1階層の要約（中程度の抽象化）
- レベル2: 第2階層の要約（高度な抽象化）
- レベル3: 第3階層の要約（最も抽象的）

使用方法:
1. 階層的インデックスを初期化:
   python rag_raptor.py --init

2. RAGクエリを実行:
   python rag_raptor.py "紅葉が見たいです"

3. 特定の階層で検索:
   python rag_raptor.py --level 2 "アクティビティが楽しいツアーは?"
"""

import os
import json
import uuid
from typing import List, Dict, Optional
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_anthropic import ChatAnthropic
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_classic.retrievers.multi_vector import MultiVectorRetriever
from langchain_core.stores import InMemoryStore
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.retrievers import BaseRetriever

load_dotenv()


def load_tour_data(tour_data_file: str = "tour_data.json") -> List[dict]:
    """ツアーデータをJSONファイルから読み込む"""
    if not os.path.exists(tour_data_file):
        raise FileNotFoundError(f"{tour_data_file}が見つかりません")

    with open(tour_data_file, 'r', encoding='utf-8') as f:
        tours = json.load(f)

    return tours


def create_parent_documents(tours: List[dict]) -> List[Document]:
    """親ドキュメント（レベル0：最も詳細な情報）を作成"""
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
                "level": 0,  # レベル0: 親ドキュメント
            }
        )

        parent_docs.append(parent_doc)

    return parent_docs


def generate_hierarchical_summaries(
    parent_docs: List[Document],
    num_levels: int = 3
) -> Dict[int, List[Document]]:
    """
    階層的要約を生成（RAPTOR方式）

    Args:
        parent_docs: 親ドキュメントのリスト
        num_levels: 階層のレベル数（デフォルト3）

    Returns:
        各レベルの要約ドキュメント
    """
    llm = ChatAnthropic(
        model="claude-3-haiku-20240307",
        temperature=0.3,
    )

    hierarchical_docs = {0: parent_docs}  # レベル0は親ドキュメント

    # 各階層の要約プロンプト
    level_prompts = {
        1: """以下のツアー情報を要約してください。
主要な特徴、場所、時期、見どころを含めた中程度の詳細レベルで要約してください。

ツアー情報:
{content}

要約（150-200文字）:""",
        2: """以下のツアー情報を簡潔に要約してください。
最も重要な特徴とキーワードのみを含めてください。

ツアー情報:
{content}

要約（100-150文字）:""",
        3: """以下のツアー情報を非常に簡潔に要約してください。
最も本質的な特徴だけを1-2文で表現してください。

ツアー情報:
{content}

要約（50-100文字）:"""
    }

    # 各レベルの要約を生成
    for level in range(1, num_levels + 1):
        print(f"\nレベル{level}の要約を生成中...")

        # 前のレベルのドキュメントを取得
        previous_level_docs = hierarchical_docs[level - 1]
        current_level_docs = []

        # プロンプトを作成
        prompt_template = level_prompts.get(level, level_prompts[3])
        summary_prompt = ChatPromptTemplate.from_template(prompt_template)
        summary_chain = summary_prompt | llm | StrOutputParser()

        for parent_doc in previous_level_docs:
            # 要約を生成
            summary = summary_chain.invoke({"content": parent_doc.page_content})

            # 子ドキュメントを作成
            child_doc = Document(
                page_content=summary.strip(),
                metadata={
                    "doc_id": parent_doc.metadata["doc_id"],
                    "tour_name": parent_doc.metadata["tour_name"],
                    "level": level,
                    "parent_level": level - 1,
                }
            )

            current_level_docs.append(child_doc)
            print(f"  レベル{level}生成完了: {parent_doc.metadata['tour_name']}")

        hierarchical_docs[level] = current_level_docs

    return hierarchical_docs


class RaptorRetriever(BaseRetriever):
    """
    RAPTOR方式の階層的検索を行うRetriever

    複数の階層から関連ドキュメントを検索し、親ドキュメントを返す
    """

    vectorstores: Dict[int, FAISS]
    docstore: InMemoryStore
    id_key: str = "doc_id"
    num_neighbors: int = 3
    levels_to_search: Optional[List[int]] = None  # Noneの場合は全レベルを検索

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun | None = None
    ) -> List[Document]:
        """
        クエリに関連する親ドキュメントを階層的検索で取得
        """
        # 検索対象のレベルを決定
        if self.levels_to_search is None:
            levels = list(self.vectorstores.keys())
        else:
            levels = self.levels_to_search

        # 各レベルから検索
        all_child_docs = []
        for level in levels:
            vectorstore = self.vectorstores[level]
            docs = vectorstore.similarity_search(query, k=self.num_neighbors)
            all_child_docs.extend(docs)

        # 親ドキュメントのIDを取得（重複を除く）
        parent_ids = []
        seen_ids = set()
        for doc in all_child_docs:
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


def initialize_raptor_retriever(
    num_levels: int = 3,
    persist_dir: str = "./raptor_index"
) -> RaptorRetriever:
    """
    RAPTORベースのMultiVectorRetrieverを初期化

    Args:
        num_levels: 階層のレベル数
        persist_dir: インデックスの保存ディレクトリ
    """
    # ツアーデータを読み込み
    tours = load_tour_data()

    # 親ドキュメントを作成
    parent_docs = create_parent_documents(tours)

    # 階層的要約を生成
    hierarchical_docs = generate_hierarchical_summaries(parent_docs, num_levels)

    # Embeddingsモデルを初期化
    embeddings = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large")

    # 各レベルのベクトルストアを作成
    vectorstores = {}
    os.makedirs(persist_dir, exist_ok=True)

    for level in range(1, num_levels + 1):  # レベル1以上の要約のみインデックス化
        level_docs = hierarchical_docs[level]

        # FAISSベクトルストアを作成
        vectorstore = FAISS.from_documents(level_docs, embeddings)

        # レベルごとに保存
        level_dir = os.path.join(persist_dir, f"level_{level}")
        os.makedirs(level_dir, exist_ok=True)
        vectorstore.save_local(level_dir)

        vectorstores[level] = vectorstore
        print(f"レベル{level}のベクトルストアを{level_dir}に保存しました（{len(level_docs)}件）")

    # docstore（親ドキュメント用）を作成
    docstore = InMemoryStore()
    doc_ids = [doc.metadata["doc_id"] for doc in parent_docs]
    docstore.mset(list(zip(doc_ids, parent_docs)))

    # RaptorRetrieverを作成
    retriever = RaptorRetriever(
        vectorstores=vectorstores,
        docstore=docstore,
        id_key="doc_id",
        num_neighbors=2,  # 各レベルから2件ずつ
    )

    print(f"\nRaptorRetriever初期化完了")
    print(f"  親ドキュメント数: {len(parent_docs)}")
    print(f"  階層レベル数: {num_levels}")
    for level, docs in hierarchical_docs.items():
        if level > 0:
            print(f"  レベル{level}の要約数: {len(docs)}")

    return retriever


def load_raptor_retriever(
    persist_dir: str = "./raptor_index",
    num_levels: int = 3,
    levels_to_search: Optional[List[int]] = None
) -> RaptorRetriever:
    """保存済みのRaptorRetrieverを読み込む"""
    if not os.path.exists(persist_dir):
        raise FileNotFoundError(
            f"{persist_dir}が見つかりません。先に --init で初期化してください"
        )

    # Embeddingsモデルを初期化
    embeddings = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large")

    # 各レベルのベクトルストアを読み込み
    vectorstores = {}
    for level in range(1, num_levels + 1):
        level_dir = os.path.join(persist_dir, f"level_{level}")
        if os.path.exists(level_dir):
            vectorstore = FAISS.load_local(
                level_dir,
                embeddings,
                allow_dangerous_deserialization=True
            )
            vectorstores[level] = vectorstore
            print(f"レベル{level}のベクトルストアを読み込みました")

    # docstoreを再構築
    tours = load_tour_data()
    parent_docs = create_parent_documents(tours)

    docstore = InMemoryStore()
    doc_ids = [doc.metadata["doc_id"] for doc in parent_docs]
    docstore.mset(list(zip(doc_ids, parent_docs)))

    # RaptorRetrieverを作成
    retriever = RaptorRetriever(
        vectorstores=vectorstores,
        docstore=docstore,
        id_key="doc_id",
        num_neighbors=2,
        levels_to_search=levels_to_search,
    )

    return retriever


def format_docs(docs: List[Document]) -> str:
    """ドキュメントをフォーマット"""
    formatted = []
    for idx, doc in enumerate(docs, 1):
        formatted.append(f"【ツアー {idx}】\n{doc.page_content}")
    return "\n\n".join(formatted)


def create_rag_chain(retriever: RaptorRetriever):
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
    persist_dir: str = "./raptor_index",
    levels_to_search: Optional[List[int]] = None
):
    """RAGクエリを実行"""
    print(f"\n{'='*60}")
    print(f"質問: {query}")
    if levels_to_search:
        print(f"検索レベル: {levels_to_search}")
    else:
        print(f"検索レベル: 全レベル")
    print(f"{'='*60}\n")

    # RaptorRetrieverを読み込み
    retriever = load_raptor_retriever(
        persist_dir,
        levels_to_search=levels_to_search
    )

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
        num_levels = 3
        if len(sys.argv) > 2 and sys.argv[2] == "--levels":
            num_levels = int(sys.argv[3]) if len(sys.argv) > 3 else 3

        print(f"RaptorRetrieverを初期化します（階層レベル: {num_levels}）...")
        initialize_raptor_retriever(num_levels=num_levels)
        print("\n初期化完了！")

    elif len(sys.argv) > 1 and sys.argv[1] == "--level":
        # 特定レベルで検索
        level = int(sys.argv[2]) if len(sys.argv) > 2 else 1
        query = sys.argv[3] if len(sys.argv) > 3 else "紅葉が見たいです"

        # RAGクエリを実行
        run_rag_query(query, levels_to_search=[level])

    else:
        # クエリを引数から取得
        query = sys.argv[1] if len(sys.argv) > 1 else "紅葉が見たいです"

        # RAGクエリを実行（全レベルを検索）
        run_rag_query(query)
