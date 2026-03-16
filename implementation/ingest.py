import os
import glob
from pathlib import Path
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from pydantic import BaseModel, Field
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.documents import Document
from tqdm import tqdm
import uuid


# Load environment variables from .env (e.g. API keys).
from dotenv import load_dotenv

# Load .env and override any existing env vars.
load_dotenv(override=True)

MODEL = "gpt-4.1-mini"

# Path to the Chroma vector DB directory (project root / vector_db).
DB_NAME = str(Path(__file__).parent.parent / "vector_db")
# Path to the knowledge base directory (project root / knowledge-base).
KNOWLEDGE_BASE = str(Path(__file__).parent.parent / "knowledge-base")

# Embedding model used to turn text into vectors (OpenAI text-embedding-3-small by default).
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")



AVERAGE_CHUNK_SIZE = 500

# LLM for structured output (chunking)
llm = ChatOpenAI(temperature=0, model_name=MODEL)

class Result(BaseModel):
    page_content: str
    metadata: dict
    id: str | None = None  # Chroma.from_documents expects .id on each doc


class Chunk(BaseModel):
    headline: str = Field(description="A brief heading for this chunk, typically a few words, that is most likely to be surfaced in a query")
    summary: str = Field(description="A few sentences summarizing the content of this chunk to answer common questions")
    original_text: str = Field(description="The original text of this chunk from the provided document, exactly as is, not changed in any way")

    def as_result(self, document):
        metadata = {"source": document.metadata.get("source"), "type": document.metadata.get("doc_type")}
        return Result(
            page_content=self.headline + "\n\n" + self.summary + "\n\n" + self.original_text,
            metadata=metadata,
            id=str(uuid.uuid4()),
        )


class Chunks(BaseModel):
    chunks: list[Chunk]



class DocumentChunks(BaseModel):
    """Chunks associated with a particular document in a batch.

    The `doc_index` field is 1-based and refers to the position of the document
    in the batch that was sent to the LLM.
    """

    doc_index: int = Field(
        description="1-based index of the document in the batch this chunk list belongs to.",
        ge=1,
    )
    chunks: list[Chunk]


class BatchChunks(BaseModel):
    """Structured response for multiple documents chunked in a single LLM call."""

    documents: list[DocumentChunks] = Field(
        description="For each input document, the chunks that cover that document."
    )



def fetch_documents():
    # List all top-level folders under the knowledge base.
    folders = glob.glob(str(Path(KNOWLEDGE_BASE) / "*"))
    # Accumulate loaded documents.
    documents = []
    for folder in folders:
        # Use folder name as document type (e.g. "docs", "faq").
        doc_type = os.path.basename(folder)
        # Loader: all .md files under this folder, UTF-8 text loader.
        loader = DirectoryLoader(
            folder, glob="**/*.md", loader_cls=TextLoader, loader_kwargs={"encoding": "utf-8"}
        )
        # Load documents from this folder.
        folder_docs = loader.load()
        for doc in folder_docs:
            # Tag each doc with its folder (doc_type).
            doc.metadata["doc_type"] = doc_type
            documents.append(doc)
    return documents

def make_prompt(document):
    how_many = (len(document.page_content) // AVERAGE_CHUNK_SIZE) + 1
    type = document.metadata.get("doc_type")
    source = document.metadata.get("source")
    return f"""
    You take a document and you split the document into overlapping chunks for a KnowledgeBase.

    The document is from the shared drive of a company called Insurellm.
    The document is of type: {type}
    The document has been retrieved from: {source}

    A chatbot will use these chunks to answer questions about the company.
    You should divide up the document as you see fit, being sure that the entire document is returned in the chunks - don't leave anything out.
    This document should probably be split into {how_many} chunks, but you can have more or less as appropriate with focus on completeness and relevance.
    There should be overlap between the chunks as appropriate; typically about 25% overlap or about 50 words, so you have the same text in multiple chunks for best retrieval results.

    For each chunk, you should provide a headline, a summary, and the original text of the chunk.
    Together your chunks should represent the entire document with overlap.

    Here is the document:

    {document.page_content}

    Respond with the chunks.
    """


def process_document(document):
    """Keep single-document processing available (e.g. for debugging)."""
    messages = [HumanMessage(content=make_prompt(document))]
    chunk_llm = llm.with_structured_output(Chunks)
    chunks_result = chunk_llm.invoke(messages)
    return [chunk.as_result(document) for chunk in chunks_result.chunks]


def make_batch_prompt(documents: list[Document]) -> str:
    """Build a prompt that asks the LLM to chunk multiple documents at once.

    Each document is given an index; the model must return chunks grouped per document.
    """
    parts: list[str] = [
        "You will take multiple documents and split EACH document into overlapping chunks for a Knowledge Base.",
        "",
        "GENERAL INSTRUCTIONS (APPLY TO EVERY DOCUMENT):",
        "- The documents are from the shared drive of a company called Insurellm.",
        "- For EACH document, you must return chunks that TOGETHER cover the ENTIRE document.",
        "- Do NOT omit any parts of any document; ensure every sentence appears in at least one chunk.",
        "- Use overlapping chunks (about 25% overlap or ~50 words) so important text appears in multiple chunks.",
        "- For each chunk, provide a headline, a summary, and the original text of the chunk.",
        "- Focus on **completeness and relevance** of the chunks so that a RAG system can later answer questions fully.",
        "",
        "You will receive several documents. For EACH document you must return:",
        "- Its 1-based `doc_index` (matching the index provided below).",
        "- A list of chunks for that single document only.",
        "",
        "IMPORTANT:",
        "- Never mix content from different documents into the same chunk.",
        "- For every document index I provide, you must return at least one chunk.",
        "- Together, your chunks for a document must represent the entire document with overlap.",
        "",
        "Here are the documents:",
    ]

    for idx, document in enumerate(documents, start=1):
        how_many = (len(document.page_content) // AVERAGE_CHUNK_SIZE) + 1
        doc_type = document.metadata.get("doc_type")
        source = document.metadata.get("source")
        parts.append(
            f"\n---\nDOCUMENT {idx}:\n"
            f"- doc_index: {idx}\n"
            f"- type: {doc_type}\n"
            f"- source: {source}\n"
            f"- suggested_number_of_chunks: {how_many}\n\n"
            f"{document.page_content}\n"
        )

    parts.append(
        "\nRespond with a structured JSON object describing ALL documents and their chunks."
    )
    return "\n".join(parts)


def process_documents_batch(documents: list[Document]) -> list[Result]:
    """Chunk multiple documents in a single LLM call.

    This reduces the number of requests by batching documents together while
    still preserving which chunks belong to which original document.
    """
    if not documents:
        return []

    messages = [HumanMessage(content=make_batch_prompt(documents))]
    batch_llm = llm.with_structured_output(BatchChunks)
    batch_result = batch_llm.invoke(messages)

    results: list[Result] = []
    used_indices: set[int] = set()

    for doc_group in batch_result.documents:
        idx = doc_group.doc_index - 1
        if idx < 0 or idx >= len(documents):
            continue
        document = documents[idx]
        # Treat an empty chunks list as a failure so the fallback can handle it.
        if not doc_group.chunks:
            continue
        used_indices.add(idx)
        for chunk in doc_group.chunks:
            results.append(chunk.as_result(document))

    # Fallback: ensure every document in this batch is processed at least once.
    # If the LLM failed to return a valid doc_index for a document, fall back
    # to per-document processing so we do not silently drop it.
    for idx, document in enumerate(documents):
        if idx not in used_indices:
            fallback_results = process_document(document)
            results.extend(fallback_results)

    return results


def create_chunks(documents, batch_size: int = 3):
    """Create chunks for all documents, batching several documents per LLM call.

    `batch_size` controls how many documents are sent in one prompt. Increasing it
    will generally reduce total request overhead, at the cost of larger prompts.
    """
    # Ensure batch_size is always a positive step for range().
    if batch_size <= 0:
        batch_size = 1

    chunks: list[Result] = []
    # Process documents in batches to reduce total number of LLM calls.
    for i in tqdm(range(0, len(documents), batch_size)):
        batch_docs = documents[i : i + batch_size]
        chunks.extend(process_documents_batch(batch_docs))
    return chunks

def create_embeddings(chunks):
    # If DB already exists, delete its collection so we rebuild from scratch.
    if os.path.exists(DB_NAME):
        Chroma(persist_directory=DB_NAME, embedding_function=embeddings).delete_collection()

    # Build a new Chroma store from chunks using our embedding model.
    vectorstore = Chroma.from_documents(
        documents=chunks, embedding=embeddings, persist_directory=DB_NAME
    )

    # Access Chroma's underlying collection for count and sample.
    collection = vectorstore._collection
    count = collection.count()

    # Get one embedding to report vector dimensions.
    sample_embedding = collection.get(limit=1, include=["embeddings"])["embeddings"][0]
    dimensions = len(sample_embedding)
    print(f"There are {count:,} vectors with {dimensions:,} dimensions in the vector store")
    return vectorstore

if __name__ == "__main__":
    # Load all .md files from knowledge-base subfolders.
    documents = fetch_documents()
    # Chunk each document via LLM (headline, summary, original_text per chunk).
    chunks = create_chunks(documents)
    # Embed chunks and persist to vector_db (Chroma).
    create_embeddings(chunks)
    print("Ingestion complete")
