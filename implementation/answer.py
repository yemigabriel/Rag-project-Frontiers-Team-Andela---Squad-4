import time
from pathlib import Path
import hashlib
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.messages import SystemMessage, HumanMessage, convert_to_messages
from langchain_core.documents import Document
from pydantic import BaseModel, Field
from openai import APIConnectionError, RateLimitError

from dotenv import load_dotenv


load_dotenv(override=True)

MODEL = "gpt-5-mini"
DB_NAME = str(Path(__file__).parent.parent / "vector_db")
MAX_RERANK_DOCS = 60

# Max characters of context to send to the answer LLM (team guideline).
MAX_CONTEXT_CHARS = 5000

RETRY_EXCEPTIONS = (APIConnectionError, RateLimitError, ConnectionError, OSError)
MAX_RETRIES = 4
RETRY_BASE_DELAY = 2.0

embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
RETRIEVAL_K = 10
N_QUERIES = 5

SYSTEM_PROMPT = """
You are a knowledgeable, friendly assistant representing the company Insurellm.
You are chatting with a user about Insurellm.
Your answer will be evaluated for accuracy, relevance and completeness.

COMPLETENESS REQUIREMENTS:
- Include ALL relevant facts, figures, names, and dates from the context
- If the question asks for a list or multiple items, provide the COMPLETE list — never truncate it
- If there are multiple aspects to the question, address every single one
- Include supporting details (e.g. job titles, locations, exact numbers) that make the answer fully informative
- Do NOT omit relevant information that is present in the context
- If the question has multiple sub-questions, answer each one explicitly in its own sentence or bullet, so that nothing is implicitly assumed.
- If the question uses words like "all", "any", "which", or "what are", ensure you enumerate every relevant item from the context instead of summarizing them away.
- At the end of your reasoning, quickly check if there are any remaining relevant details in the context that you have not mentioned yet, and add them to your answer.

If you don't know the answer, say so.
For context, here are specific extracts from the Knowledge Base that might be directly relevant to the user's question:
{context}

With this context, please answer the user's question. Be accurate, relevant and fully comprehensive.
"""

vectorstore = Chroma(persist_directory=DB_NAME, embedding_function=embeddings)
retriever = vectorstore.as_retriever()
llm = ChatOpenAI(temperature=0, model_name=MODEL)
# Structured output LLM for JSON responses
structured_llm = ChatOpenAI(temperature=0, model_name=MODEL)


class RankOrder(BaseModel):
    order: list[int] = Field(
        description="The order of relevance of chunks, from most relevant to least relevant, by chunk id number. Each ID must be unique and within the valid range.",
        min_length=1,
    )

    def validate_against_chunks(self, num_chunks: int) -> bool:
        """Check if order contains valid, unique IDs for the given number of chunks."""
        return (
            len(self.order) == num_chunks and
            len(set(self.order)) == num_chunks and
            all(1 <= i <= num_chunks for i in self.order)
        )

def rerank(question, chunks):
    num_chunks = len(chunks)
    system_prompt = f"""
You are a document re-ranker.
You are provided with a question and a list of relevant chunks of text from a query of a knowledge base.
The chunks are provided in the order they were retrieved; this should be approximately ordered by relevance, but you may be able to improve on that.
You must rank order the provided chunks by relevance to the question, with the most relevant chunk first.

CRITICAL REQUIREMENTS:
- You will receive exactly {num_chunks} chunks with IDs from 1 to {num_chunks}
- You MUST return ALL {num_chunks} chunk IDs in your ranking
- Each ID must appear exactly once
- IDs must be integers between 1 and {num_chunks} inclusive
- Return them in order from most relevant to least relevant
"""
    user_prompt = f"The user has asked the following question:\n\n{question}\n\n"
    user_prompt += f"Rank ALL {num_chunks} chunks by relevance (most relevant first).\n\n"
    user_prompt += "Here are the chunks:\n\n"
    for index, chunk in enumerate(chunks):
        user_prompt += f"# CHUNK ID: {index + 1}:\n\n{chunk.page_content}\n\n"
    user_prompt += f"\nReturn a JSON object with 'order' containing all {num_chunks} chunk IDs ranked by relevance."
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ]

    # Create structured LLM for this specific response format
    rank_llm = structured_llm.with_structured_output(RankOrder)

    # Try to get valid response (with one retry)
    for attempt in range(2):
        rank_result = rank_llm.invoke(messages)

        # Check if response is valid
        if rank_result.validate_against_chunks(num_chunks):
            print(f"LLM returned valid order: {rank_result.order}")
            return [chunks[i - 1] for i in rank_result.order]

        print(f"Attempt {attempt + 1}: LLM returned invalid order: {rank_result.order} (expected {num_chunks} unique IDs)")

        # On first failure, add a correction message and retry
        if attempt == 0:
            messages.append(HumanMessage(content=f"That response is invalid. You must return exactly {num_chunks} unique chunk IDs (1 to {num_chunks}). Try again."))

    # If both attempts fail, fall back to validation logic
    order = rank_result.order
    print(f"LLM failed validation after retries. Falling back to repair logic.")
    print(f"LLM returned order: {order}, num_chunks: {num_chunks}")

    # Validate and filter: keep only valid 1-indexed IDs within range
    valid_order = [i for i in order if 1 <= i <= len(chunks)]

    # If LLM missed some chunks or returned invalid IDs, log a warning
    if len(valid_order) != len(chunks):
        print(f"Warning: LLM returned {len(order)} IDs but we have {len(chunks)} chunks. Using {len(valid_order)} valid IDs.")

    # Deduplicate while preserving order (in case LLM returned duplicates)
    seen = set()
    reranked = []
    for i in valid_order:
        if i not in seen:
            seen.add(i)
            reranked.append(chunks[i - 1])

    # If some chunks are missing from reranked, append them at the end
    if len(reranked) < len(chunks):
        missing_indices = set(range(len(chunks))) - {i - 1 for i in valid_order}
        for idx in sorted(missing_indices):
            reranked.append(chunks[idx])

    return reranked

def fetch_context_unranked(question: str) -> list[Document]:
    """
    Retrieve relevant context documents for a question.
    Retries on transient connection/SSL and rate-limit errors.
    """
    last_error = None
    for attempt in range(MAX_RETRIES):
        try:
            return retriever.invoke(question, k=RETRIEVAL_K)
        except RETRY_EXCEPTIONS as e:
            last_error = e
            if attempt < MAX_RETRIES - 1:
                delay = RETRY_BASE_DELAY * (2**attempt)
                time.sleep(delay)
            else:
                raise last_error


def generate_sub_queries(question: str, n: int = N_QUERIES - 1) -> list[str]:
    """Generate n alternative search queries to improve retrieval coverage."""
    message = f"""You are searching a Knowledge Base about the company Insurellm to answer this question:

{question}

Generate {n} different search queries that together will help find ALL information needed to fully answer this question.
Each query should target a different aspect or use different terminology than the original.
Respond with exactly {n} queries, one per line, nothing else.
"""
    response = llm.invoke([SystemMessage(content=message)])
    return [q.strip() for q in response.content.strip().split("\n") if q.strip()][:n]


def fetch_context(question: str) -> list[Document]:
    """
    Retrieve context using multi-query strategy for better coverage.
    Generates sub-queries, retrieves docs for each, deduplicates, then reranks.
    """
    queries = [question] + generate_sub_queries(question)

    all_docs: list[Document] = []
    seen_content = set()
    for query in queries:
        if len(all_docs) >= MAX_RERANK_DOCS:
            break
        for doc in fetch_context_unranked(query):
            if len(all_docs) >= MAX_RERANK_DOCS:
                break
            content_key = hashlib.sha256(doc.page_content.encode("utf-8")).hexdigest()
            if content_key not in seen_content:
                seen_content.add(content_key)
                all_docs.append(doc)

    if not all_docs:
        return []

    reranked = rerank(question, all_docs)
    return reranked[:15]  # Top 15 after reranking



def rewrite_query(question, history=[]):
    """Rewrite the user's question to be a more specific question that is more likely to surface relevant content in the Knowledge Base with focus on completeness."""
    message = f"""
You are in a conversation with a user, answering questions about the company Insurellm.
You are about to look up information in a Knowledge Base to answer the user's question.

This is the history of your conversation so far with the user:
{history}

And this is the user's current question:
{question}

Respond only with a single, refined question that you will use to search the Knowledge Base with focus on completeness of answers from the knowledge base.
It should be a VERY short specific question most likely to surface detailed content from the knowledge base. Focus on the question details.
Make sure the refined question still covers every aspect of the user's original question so that all required information can be retrieved.
Don't mention the company name unless it's a general question about the company.
IMPORTANT: Respond ONLY with the knowledgebase query, nothing else.
"""
    response = llm.invoke([SystemMessage(content=message)])
    return response.content


def _context_within_limit(docs: list[Document], max_chars: int = MAX_CONTEXT_CHARS) -> str:
    """Build context string from docs without exceeding max_chars. Docs are in rerank order (most relevant first)."""
    if not docs:
        return ""
    first = docs[0].page_content
    if len(first) >= max_chars:
        return first[:max_chars]
    parts = [first]
    total = len(first)
    sep = "\n\n"
    sep_len = len(sep)
    for doc in docs[1:]:
        content = doc.page_content
        need = len(content) + sep_len
        if total + need > max_chars:
            break
        parts.append(content)
        total += need
    return sep.join(parts)


def answer_question(question: str, history: list[dict] = []) -> tuple[str, list[Document]]:
    """
    Answer the given question with RAG; return the answer and the context documents.
    """

    query = rewrite_query(question, history)
    print(query)
    docs = fetch_context(query)
    context = _context_within_limit(docs)
    system_prompt = SYSTEM_PROMPT.format(context=context)
    messages = [SystemMessage(content=system_prompt)]
    messages.extend(convert_to_messages(history))
    messages.append(HumanMessage(content=question))
    response = llm.invoke(messages)
    return response.content, docs
