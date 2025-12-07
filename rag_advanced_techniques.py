"""
LangChainを使った高度なRAG手法の実装

このファイルでは以下の手法を実装します：
1. クエリ変換（Query Transformation）
   - Multi-Query: 複数のクエリバリエーションを生成
   - RAG-Fusion: 複数クエリ + Reciprocal Rank Fusion
   - HyDE: 仮想的な回答を生成して検索
   - Step-back Prompting: より抽象的な質問を生成

2. クエリルーティング（Query Routing）
   - 論理的ルーティング: LLMで適切なデータソースを選択
   - 意味的ルーティング: 埋め込みベースでルーティング

3. クエリ構築（Query Construction）
   - Self-Query Retriever: メタデータフィルターを自動生成
   - 構造化クエリ生成

使用方法:
python rag_advanced_techniques.py --technique <technique_name> "<query>"

例:
python rag_advanced_techniques.py --technique multi-query "紅葉が見たいです"
python rag_advanced_techniques.py --technique rag-fusion "温泉に行きたい"
python rag_advanced_techniques.py --technique hyde "家族旅行におすすめは？"
python rag_advanced_techniques.py --technique routing "京都の観光地を教えて"
python rag_advanced_techniques.py --technique self-query "秋に行ける初心者向けのツアー"
"""

import os
import json
from typing import List, Dict, Tuple
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_anthropic import ChatAnthropic
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_community.vectorstores import FAISS
from collections import defaultdict

load_dotenv()


# =====================================================
# 基本的なRetrieverとヘルパー関数
# =====================================================

def get_faiss_retriever(k: int = 3) -> FAISS:
    """FAISSベースのretrieverを取得（ローカルで実行可能）"""
    embeddings = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large")

    # tour_data.jsonからドキュメントを読み込み
    if os.path.exists("tour_data.json"):
        with open("tour_data.json", 'r', encoding='utf-8') as f:
            tours = json.load(f)

        documents = []
        for tour in tours:
            page_content = f"""ツアー名: {tour.get('name', 'N/A')}
場所: {tour.get('location', 'N/A')}
時期: {tour.get('season', 'N/A')}
説明: {tour.get('description', 'N/A')}
見どころ: {', '.join(tour.get('highlights', []))}
価格: {tour.get('price', 'N/A')}
期間: {tour.get('duration', 'N/A')}
難易度: {tour.get('difficulty', 'N/A')}"""

            doc = Document(
                page_content=page_content,
                metadata={
                    "tour_id": tour.get('id', tour.get('name', '')),
                    "tour_name": tour.get('name', 'N/A'),
                    "location": tour.get('location', 'N/A'),
                    "season": tour.get('season', 'N/A'),
                    "difficulty": tour.get('difficulty', 'N/A'),
                    "price": tour.get('price', 'N/A'),
                }
            )
            documents.append(doc)

        vectorstore = FAISS.from_documents(documents, embeddings)
        return vectorstore.as_retriever(search_kwargs={"k": k})
    else:
        raise FileNotFoundError("tour_data.jsonが見つかりません")


def format_docs(docs: List[Document]) -> str:
    """ドキュメントをフォーマット"""
    formatted = []
    for idx, doc in enumerate(docs, 1):
        formatted.append(f"【ツアー {idx}: {doc.metadata.get('tour_name', 'N/A')}】\n{doc.page_content}")
    return "\n\n".join(formatted)


# =====================================================
# 1. クエリ変換（Query Transformation）
# =====================================================

class MultiQueryRAG:
    """Multi-Query: 1つのクエリから複数のバリエーションを生成して検索"""

    def __init__(self):
        self.llm = ChatAnthropic(model="claude-3-haiku-20240307", temperature=0.7)
        self.retriever = get_faiss_retriever(k=2)

        # 複数クエリ生成用のプロンプト
        self.query_prompt = PromptTemplate(
            input_variables=["question"],
            template="""あなたは旅行プランナーのアシスタントです。
ユーザーの質問に対して、異なる視点から3つの検索クエリを生成してください。

元の質問: {question}

3つの検索クエリ（1行に1つずつ）:"""
        )

    def generate_queries(self, question: str) -> List[str]:
        """元の質問から複数のクエリバリエーションを生成"""
        chain = self.query_prompt | self.llm | StrOutputParser()
        response = chain.invoke({"question": question})

        # レスポンスを行ごとに分割
        queries = [q.strip() for q in response.split('\n') if q.strip() and not q.strip().startswith('-')]
        queries = [question] + queries[:3]  # 元のクエリも含める

        return queries

    def retrieve_unique_docs(self, queries: List[str]) -> List[Document]:
        """複数のクエリで検索し、重複を除いた結果を返す"""
        all_docs = []
        seen_ids = set()

        for query in queries:
            docs = self.retriever.invoke(query)
            for doc in docs:
                doc_id = doc.metadata.get('tour_id', doc.page_content[:50])
                if doc_id not in seen_ids:
                    seen_ids.add(doc_id)
                    all_docs.append(doc)

        return all_docs

    def run(self, question: str) -> str:
        """Multi-Query RAGを実行"""
        print(f"\n{'='*60}")
        print("Multi-Query RAG")
        print(f"{'='*60}\n")

        # 複数クエリを生成
        queries = self.generate_queries(question)
        print("生成されたクエリ:")
        for i, q in enumerate(queries, 1):
            print(f"  {i}. {q}")

        # 検索実行
        docs = self.retrieve_unique_docs(queries)
        print(f"\n検索されたドキュメント数: {len(docs)}\n")

        # 回答生成
        answer_llm = ChatAnthropic(model="claude-3-opus-20240229", temperature=0.3)
        prompt = ChatPromptTemplate.from_template(
            """以下の旅行ツアー情報を参考にして、質問に答えてください。

関連するツアー:
{context}

質問: {question}

回答:"""
        )

        chain = (
            {"context": RunnableLambda(lambda x: format_docs(docs)), "question": RunnablePassthrough()}
            | prompt
            | answer_llm
            | StrOutputParser()
        )

        result = chain.invoke(question)
        return result


class RAGFusion:
    """RAG-Fusion: 複数クエリ + Reciprocal Rank Fusion (RRF)"""

    def __init__(self):
        self.llm = ChatAnthropic(model="claude-3-haiku-20240307", temperature=0.7)
        self.retriever = get_faiss_retriever(k=5)

        self.query_prompt = PromptTemplate(
            input_variables=["question"],
            template="""以下の質問に対して、異なる視点から4つの検索クエリを生成してください。

元の質問: {question}

4つの検索クエリ（1行に1つずつ）:"""
        )

    def generate_queries(self, question: str) -> List[str]:
        """複数のクエリバリエーションを生成"""
        chain = self.query_prompt | self.llm | StrOutputParser()
        response = chain.invoke({"question": question})

        queries = [q.strip() for q in response.split('\n') if q.strip() and not q.strip().startswith('-')]
        queries = [question] + queries[:4]

        return queries

    def reciprocal_rank_fusion(self, results: List[List[Document]], k: int = 60) -> List[Tuple[Document, float]]:
        """Reciprocal Rank Fusion (RRF)でドキュメントをre-rank"""
        doc_scores = defaultdict(float)
        doc_objects = {}

        for doc_list in results:
            for rank, doc in enumerate(doc_list):
                doc_id = doc.metadata.get('tour_id', doc.page_content[:50])
                # RRFスコア: 1 / (k + rank)
                doc_scores[doc_id] += 1.0 / (k + rank + 1)
                if doc_id not in doc_objects:
                    doc_objects[doc_id] = doc

        # スコアでソート
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        return [(doc_objects[doc_id], score) for doc_id, score in sorted_docs]

    def run(self, question: str) -> str:
        """RAG-Fusionを実行"""
        print(f"\n{'='*60}")
        print("RAG-Fusion")
        print(f"{'='*60}\n")

        # 複数クエリを生成
        queries = self.generate_queries(question)
        print("生成されたクエリ:")
        for i, q in enumerate(queries, 1):
            print(f"  {i}. {q}")

        # 各クエリで検索
        all_results = []
        for query in queries:
            docs = self.retriever.invoke(query)
            all_results.append(docs)

        # RRFでre-rank
        reranked_docs = self.reciprocal_rank_fusion(all_results)
        print(f"\nRRFスコアでre-rankされたドキュメント数: {len(reranked_docs)}")

        # 上位3件を使用
        top_docs = [doc for doc, score in reranked_docs[:3]]

        print("\n上位3件のRRFスコア:")
        for i, (doc, score) in enumerate(reranked_docs[:3], 1):
            print(f"  {i}. {doc.metadata.get('tour_name', 'N/A')} (スコア: {score:.4f})")

        # 回答生成
        answer_llm = ChatAnthropic(model="claude-3-opus-20240229", temperature=0.3)
        prompt = ChatPromptTemplate.from_template(
            """以下の旅行ツアー情報を参考にして、質問に答えてください。

関連するツアー:
{context}

質問: {question}

回答:"""
        )

        chain = (
            {"context": RunnableLambda(lambda x: format_docs(top_docs)), "question": RunnablePassthrough()}
            | prompt
            | answer_llm
            | StrOutputParser()
        )

        result = chain.invoke(question)
        return result


class HyDERAG:
    """HyDE (Hypothetical Document Embeddings): 仮想的な回答を生成してそれで検索"""

    def __init__(self):
        self.llm = ChatAnthropic(model="claude-3-haiku-20240307", temperature=0.7)
        self.embeddings = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large")
        self.retriever = get_faiss_retriever(k=3)

        self.hyde_prompt = ChatPromptTemplate.from_template(
            """以下の質問に対する理想的な旅行ツアーの説明を生成してください。
実際のツアーである必要はありませんが、質問に完璧に答える架空のツアーを想像してください。

質問: {question}

理想的なツアーの説明:"""
        )

    def generate_hypothetical_document(self, question: str) -> str:
        """仮想的なドキュメントを生成"""
        chain = self.hyde_prompt | self.llm | StrOutputParser()
        hypothetical_doc = chain.invoke({"question": question})
        return hypothetical_doc

    def run(self, question: str) -> str:
        """HyDE RAGを実行"""
        print(f"\n{'='*60}")
        print("HyDE (Hypothetical Document Embeddings)")
        print(f"{'='*60}\n")

        # 仮想ドキュメントを生成
        hypothetical_doc = self.generate_hypothetical_document(question)
        print("生成された仮想ドキュメント:")
        print(f"{hypothetical_doc[:200]}...\n")

        # 仮想ドキュメントで検索
        docs = self.retriever.invoke(hypothetical_doc)
        print(f"検索されたドキュメント数: {len(docs)}\n")

        # 回答生成
        answer_llm = ChatAnthropic(model="claude-3-opus-20240229", temperature=0.3)
        prompt = ChatPromptTemplate.from_template(
            """以下の旅行ツアー情報を参考にして、質問に答えてください。

関連するツアー:
{context}

質問: {question}

回答:"""
        )

        chain = (
            {"context": RunnableLambda(lambda x: format_docs(docs)), "question": RunnablePassthrough()}
            | prompt
            | answer_llm
            | StrOutputParser()
        )

        result = chain.invoke(question)
        return result


# =====================================================
# 2. クエリルーティング（Query Routing）
# =====================================================

class LogicalRoutingRAG:
    """論理的ルーティング: LLMでクエリの種類を判断してデータソースを選択"""

    def __init__(self):
        self.llm = ChatAnthropic(model="claude-3-haiku-20240307", temperature=0)
        self.retriever = get_faiss_retriever(k=3)

        # ルーティング用プロンプト
        self.routing_prompt = ChatPromptTemplate.from_template(
            """以下の質問を分析し、最適な検索戦略を選択してください。

質問: {question}

以下から1つ選んでください：
- "vector_search": 一般的な旅行の質問（場所、時期、アクティビティなど）
- "metadata_filter": 特定の条件でフィルタリング（価格、難易度、期間など）
- "direct_answer": データベース検索が不要な一般的な質問

選択した戦略のみを1語で答えてください:"""
        )

    def route_query(self, question: str) -> str:
        """クエリをルーティング"""
        chain = self.routing_prompt | self.llm | StrOutputParser()
        route = chain.invoke({"question": question}).strip().lower()

        # "vector_search"などの部分を抽出
        if "vector" in route:
            return "vector_search"
        elif "metadata" in route:
            return "metadata_filter"
        elif "direct" in route:
            return "direct_answer"
        else:
            return "vector_search"  # デフォルト

    def run(self, question: str) -> str:
        """ルーティングRAGを実行"""
        print(f"\n{'='*60}")
        print("Logical Routing RAG")
        print(f"{'='*60}\n")

        # クエリをルーティング
        route = self.route_query(question)
        print(f"選択されたルート: {route}\n")

        if route == "direct_answer":
            # データベース検索なしで直接回答
            answer_llm = ChatAnthropic(model="claude-3-opus-20240229", temperature=0.3)
            prompt = ChatPromptTemplate.from_template(
                "以下の質問に簡潔に答えてください：\n\n{question}"
            )
            chain = prompt | answer_llm | StrOutputParser()
            result = chain.invoke({"question": question})

        elif route == "metadata_filter":
            # メタデータフィルタリング（簡易版）
            print("メタデータフィルタリングを使用します")
            docs = self.retriever.invoke(question)

            answer_llm = ChatAnthropic(model="claude-3-opus-20240229", temperature=0.3)
            prompt = ChatPromptTemplate.from_template(
                """以下の旅行ツアー情報から、質問の条件に合うものを選んで答えてください。

関連するツアー:
{context}

質問: {question}

回答:"""
            )
            chain = (
                {"context": RunnableLambda(lambda x: format_docs(docs)), "question": RunnablePassthrough()}
                | prompt
                | answer_llm
                | StrOutputParser()
            )
            result = chain.invoke(question)

        else:  # vector_search
            # 通常のベクトル検索
            print("ベクトル検索を使用します")
            docs = self.retriever.invoke(question)

            answer_llm = ChatAnthropic(model="claude-3-opus-20240229", temperature=0.3)
            prompt = ChatPromptTemplate.from_template(
                """以下の旅行ツアー情報を参考にして、質問に答えてください。

関連するツアー:
{context}

質問: {question}

回答:"""
            )
            chain = (
                {"context": RunnableLambda(lambda x: format_docs(docs)), "question": RunnablePassthrough()}
                | prompt
                | answer_llm
                | StrOutputParser()
            )
            result = chain.invoke(question)

        return result


# =====================================================
# 3. クエリ構築（Query Construction）
# =====================================================

class SelfQueryRAG:
    """Self-Query: メタデータフィルターを自動生成"""

    def __init__(self):
        self.llm = ChatAnthropic(model="claude-3-haiku-20240307", temperature=0)
        self.embeddings = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large")

        # メタデータ抽出用プロンプト
        self.metadata_prompt = ChatPromptTemplate.from_template(
            """以下の質問から、検索に使用するメタデータフィルターを抽出してください。

質問: {question}

利用可能なメタデータフィールド:
- season: 時期（春、夏、秋、冬）
- difficulty: 難易度（初心者向け、中級者向け、上級者向け）
- location: 場所（京都、北海道、沖縄など）

JSON形式で出力してください（該当するフィルターがない場合は空のオブジェクト）:
{{"season": "秋", "difficulty": "初心者向け"}}

JSON:"""
        )

    def extract_metadata_filters(self, question: str) -> Dict:
        """質問からメタデータフィルターを抽出"""
        chain = self.metadata_prompt | self.llm | StrOutputParser()
        response = chain.invoke({"question": question})

        try:
            # JSONをパース
            filters = json.loads(response.strip())
            return filters
        except json.JSONDecodeError:
            return {}

    def run(self, question: str) -> str:
        """Self-Query RAGを実行"""
        print(f"\n{'='*60}")
        print("Self-Query RAG")
        print(f"{'='*60}\n")

        # メタデータフィルターを抽出
        filters = self.extract_metadata_filters(question)
        print(f"抽出されたメタデータフィルター: {filters}\n")

        # FAISSでフィルタリング付き検索
        retriever = get_faiss_retriever(k=10)
        all_docs = retriever.invoke(question)

        # メタデータでフィルタリング
        filtered_docs = []
        for doc in all_docs:
            match = True
            for key, value in filters.items():
                if doc.metadata.get(key, '').lower() != value.lower():
                    match = False
                    break
            if match:
                filtered_docs.append(doc)

        # フィルター結果が少ない場合はフィルターなしの結果も使用
        if len(filtered_docs) < 2:
            print(f"フィルター条件に一致: {len(filtered_docs)}件（少ないため全結果も使用）\n")
            filtered_docs = all_docs[:3]
        else:
            print(f"フィルター条件に一致: {len(filtered_docs)}件\n")
            filtered_docs = filtered_docs[:3]

        # 回答生成
        answer_llm = ChatAnthropic(model="claude-3-opus-20240229", temperature=0.3)
        prompt = ChatPromptTemplate.from_template(
            """以下の旅行ツアー情報を参考にして、質問に答えてください。

関連するツアー:
{context}

質問: {question}

回答:"""
        )

        chain = (
            {"context": RunnableLambda(lambda x: format_docs(filtered_docs)), "question": RunnablePassthrough()}
            | prompt
            | answer_llm
            | StrOutputParser()
        )

        result = chain.invoke(question)
        return result


# =====================================================
# メイン関数
# =====================================================

def main():
    import sys

    # コマンドライン引数の解析
    if len(sys.argv) < 2:
        print("使用方法: python rag_advanced_techniques.py --technique <technique_name> \"<query>\"")
        print("\n利用可能な手法:")
        print("  multi-query    : Multi-Query RAG")
        print("  rag-fusion     : RAG-Fusion (RRF)")
        print("  hyde           : HyDE (Hypothetical Document Embeddings)")
        print("  routing        : Logical Routing RAG")
        print("  self-query     : Self-Query RAG")
        print("\n例:")
        print("  python rag_advanced_techniques.py --technique multi-query \"紅葉が見たいです\"")
        return

    # デフォルト値
    technique = "multi-query"
    query = "紅葉が見たいです"

    # 引数解析
    if "--technique" in sys.argv:
        idx = sys.argv.index("--technique")
        if idx + 1 < len(sys.argv):
            technique = sys.argv[idx + 1]
        if idx + 2 < len(sys.argv):
            query = sys.argv[idx + 2]
    elif len(sys.argv) > 1:
        query = sys.argv[1]

    # 手法に応じて実行
    print(f"\n質問: {query}")

    if technique == "multi-query":
        rag = MultiQueryRAG()
        result = rag.run(query)
    elif technique == "rag-fusion":
        rag = RAGFusion()
        result = rag.run(query)
    elif technique == "hyde":
        rag = HyDERAG()
        result = rag.run(query)
    elif technique == "routing":
        rag = LogicalRoutingRAG()
        result = rag.run(query)
    elif technique == "self-query":
        rag = SelfQueryRAG()
        result = rag.run(query)
    else:
        print(f"不明な手法: {technique}")
        return

    # 結果を表示
    print(f"\n{'='*60}")
    print("【回答】")
    print(f"{'='*60}")
    print(result)
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
