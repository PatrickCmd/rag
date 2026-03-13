import os
from pathlib import Path
from langchain_core import embeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.messages import SystemMessage, HumanMessage, convert_to_messages
from langchain_core.documents import Document
from tenacity import retry, wait_exponential
from concurrent.futures import ThreadPoolExecutor
from sentence_transformers import CrossEncoder
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv(override=True)

MODEL = "gpt-4.1-mini"
# MODEL = "openai/gpt-oss-20b"
# groq_api_key = os.getenv('GROQ_API_KEY')
# groq_url = "https://api.groq.com/openai/v1"
DB_NAME = str(Path(__file__).parent.parent / "vector_db")

EMBEDDING_MODEL = "google/embeddinggemma-300m"

embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
# embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
RETRIEVAL_K_PER_QUERY = 20
RERANK_TOP_K = 5
NUM_SUB_QUESTIONS = 3
wait = wait_exponential(multiplier=1, min=10, max=240)


SYSTEM_PROMPT = """
You are a knowledgeable, friendly assistant representing the company Insurellm.
You are chatting with a user about Insurellm.
Your answer will be evaluated for accuracy, relevance and completeness, so make sure it only answers the question and fully answers it.
If you don't know the answer, say so.
For context, here are specific extracts from the Knowledge Base that might be directly relevant to the user's question:
{context}

With this context, please answer the user's question.
Include ALL relevant details, names, numbers, and specifics from the context. Do not summarize or omit details.
"""

vectorstore = Chroma(persist_directory=DB_NAME, embedding_function=embeddings)
retriever = vectorstore.as_retriever()
# llm = ChatOpenAI(api_key=groq_api_key, base_url=groq_url, temperature=0, model_name=MODEL
llm = ChatOpenAI(temperature=0, model_name=MODEL)

SUB_QUESTIONS_PROMPT = """\
You are in a conversation with a user, answering questions about the company Insurellm. \
You are about to look up information in a Knowledge Base to answer the user's question.

Generate exactly {n} related sub-questions that cover different facets or aspects of the question. \
Each sub-question should help retrieve relevant information from the Insurellm knowledge base \
(e.g. products, services, policies, company details, contracts information, or employee details).

This is the history of your conversation so far with the user:
{history}

And this is the user's current question:
{question}

Output only the sub-questions, one per line, without numbering or bullets. \
Keep each sub-question short and specific so it is likely to surface relevant content.\
"""


def _format_history(history: list[dict]) -> str:
    """Format chat history for use in prompts."""
    if not history:
        return "(No prior messages in this conversation.)"
    lines = []
    for turn in history:
        role = "User" if turn.get("role") == "user" else "Assistant"
        content = turn.get("content", "")
        if isinstance(content, list):
            content = " ".join(
                p.get("text", "") for p in content if isinstance(p, dict)
            )
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


@retry(wait=wait)
def generate_sub_questions(query: str, n: int = 5, history: list[dict] | None = None) -> list[str]:
    """
    Generate n related sub-questions from the user query using an LLM, using conversation history for context.
    """
    history = history or []
    prompt = SUB_QUESTIONS_PROMPT.format(
        n=n, history=_format_history(history), question=query
    )
    response = llm.invoke([HumanMessage(content=prompt)])
    text = response.content.strip()
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    # Remove leading numbers/bullets if present
    cleaned = []
    for line in lines[:n]:
        for sep in (". ", ") ", "- "):
            if sep in line and line.split(sep)[0].replace(".", "").isdigit():
                line = line.split(sep, 1)[-1].strip()
                break
        cleaned.append(line)
    return cleaned[:n]


def retrieve_chunks(queries: list[str], k_per_query: int = 10) -> list[Document]:
    def _search(q):
        return vectorstore.similarity_search(q, k=k_per_query)
    
    with ThreadPoolExecutor(max_workers=len(queries)) as pool:
        results = pool.map(_search, queries)
    docs = []
    for result in results:
        docs.extend(result)
    return docs


def deduplicate_chunks(docs: list[Document]) -> list[Document]:
    """
    Deduplicate by page_content (first occurrence wins), preserving order.
    """
    seen = set()
    out = []
    for doc in docs:
        key = doc.page_content
        if key not in seen:
            seen.add(key)
            out.append(doc)
    return out


_cross_encoder = None


def _get_cross_encoder():
    global _cross_encoder
    if _cross_encoder is None:
        from sentence_transformers import CrossEncoder
        _cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return _cross_encoder


def rerank_chunks(query: str, docs: list[Document], top_k: int = 5) -> list[Document]:
    """
    Score each chunk against the original query with a cross-encoder; return top_k.
    """
    if not docs:
        return []
    encoder = _get_cross_encoder()
    pairs = [(query, doc.page_content) for doc in docs]
    scores = encoder.predict(pairs)
    indexed = list(zip(scores, docs))
    indexed.sort(key=lambda x: x[0], reverse=True)
    return [doc for _, doc in indexed[:top_k]]


def format_context(chunks: list[Document]) -> str:
    """
    Format top-ranked chunks into a single context string for the LLM.
    """
    return "\n\n".join(doc.page_content for doc in chunks)


def fetch_context(question: str, history: list[dict] | None = None) -> list[Document]:
    """
    Retrieve relevant context: query expansion → multi-query retrieval → dedupe → re-rank (top-k).
    """
    history = history or []
    with ThreadPoolExecutor(max_workers=2) as pool:
            # Start original-query retrieval immediately
            original_future = pool.submit(
                vectorstore.similarity_search, question, k=RETRIEVAL_K_PER_QUERY
            )
            # Generate sub-questions in parallel
            sub_q_future = pool.submit(
                generate_sub_questions, question, NUM_SUB_QUESTIONS, history
            )
            original_docs = original_future.result()
            sub_questions = sub_q_future.result()
    
    # Retrieve for sub-questions only
    sub_docs = retrieve_chunks(sub_questions, k_per_query=RETRIEVAL_K_PER_QUERY)
    docs = original_docs + sub_docs
    docs = deduplicate_chunks(docs)
    docs = rerank_chunks(question, docs, top_k=RERANK_TOP_K)
    return docs

@retry(wait=wait)
def answer_question(question: str, history: list[dict] = []) -> tuple[str, list[Document]]:
    """
    Answer the given question with RAG; return the answer and the context documents.
    """
    docs = fetch_context(question, history=history)
    context = format_context(docs)
    system_prompt = SYSTEM_PROMPT.format(context=context)
    messages = [SystemMessage(content=system_prompt)]
    messages.extend(convert_to_messages(history))
    messages.append(HumanMessage(content=question))
    response = llm.invoke(messages)
    return response.content, docs
