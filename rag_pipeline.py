"""
RAG pipeline: PDF loading, chunking, embedding, retrieval, and answer
generation.

Uses local Sentence Transformers embeddings (no external embedding API,
so no dependency on any one provider's account approval) + FAISS for
retrieval, and Groq's free LLM API for generating answers.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage


CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
TOP_K = 4  # number of chunks retrieved per question

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

SYSTEM_PROMPT = """You are a helpful assistant that answers questions strictly using the \
provided excerpts from a PDF document. Follow these rules:

1. Answer using ONLY the information in the provided context. If the context doesn't contain \
the answer, say so clearly instead of guessing or using outside knowledge.
2. When you use conversation history to resolve a follow-up question (e.g. "what about the \
second one?"), make your answer self-contained -- don't assume the user can see the earlier \
messages.
3. Be concise and direct. Use bullet points for lists of items.
4. If helpful, mention which part of the document the answer came from (e.g. "According to the \
section on...")."""

_embedder = None


def get_embedder() -> HuggingFaceEmbeddings:
    """Lazily load the local embedding model once (it's slow to init)."""
    global _embedder
    if _embedder is None:
        _embedder = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return _embedder


def load_and_split_pdf(pdf_path: str) -> list[Document]:
    """Load a PDF and split it into overlapping chunks for embedding."""
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(pages)
    return chunks


def build_vector_store(chunks: list[Document], api_key: str = "") -> FAISS:
    """Embed chunks locally with Sentence Transformers and build an in-memory FAISS index.

    api_key is accepted for interface compatibility but unused -- embeddings run locally.
    """
    vector_store = FAISS.from_documents(chunks, get_embedder())
    return vector_store


def retrieve_context(vector_store: FAISS, question: str, k: int = TOP_K) -> list[Document]:
    """Retrieve the top-k most relevant chunks for a question."""
    return vector_store.similarity_search(question, k=k)


@dataclass
class ConversationMemory:
    """Simple conversation history: a list of (role, content) turns.

    Kept as a small, explicit class (rather than a LangChain memory abstraction) so the
    conversation logic is transparent and has no dependency on classes that have moved
    between LangChain packages across recent versions.
    """

    turns: list[tuple[str, str]] = field(default_factory=list)

    def add_user_turn(self, content: str) -> None:
        self.turns.append(("user", content))

    def add_assistant_turn(self, content: str) -> None:
        self.turns.append(("assistant", content))

    def as_messages(self) -> list[HumanMessage | AIMessage]:
        messages: list[HumanMessage | AIMessage] = []
        for role, content in self.turns:
            if role == "user":
                messages.append(HumanMessage(content=content))
            else:
                messages.append(AIMessage(content=content))
        return messages

    def clear(self) -> None:
        self.turns.clear()


def answer_question(
    llm: ChatGroq,
    vector_store: FAISS,
    question: str,
    memory: ConversationMemory,
    k: int = TOP_K,
) -> tuple[str, list[Document]]:
    """Retrieve relevant chunks and generate an answer, using conversation history for context.

    Returns the answer text and the list of source chunks used, so the UI can display citations.
    """
    retrieved_docs = retrieve_context(vector_store, question, k=k)
    context_text = "\n\n---\n\n".join(
        f"[Page {doc.metadata.get('page', '?')}]\n{doc.page_content}" for doc in retrieved_docs
    )

    user_message = (
        f"CONTEXT FROM THE PDF:\n---\n{context_text}\n---\n\nQUESTION: {question}"
    )

    messages = [SystemMessage(content=SYSTEM_PROMPT)]
    messages.extend(memory.as_messages())
    messages.append(HumanMessage(content=user_message))

    response = llm.invoke(messages)
    answer = response.content

    memory.add_user_turn(question)
    memory.add_assistant_turn(answer)

    return answer, retrieved_docs


def build_llm(api_key: str, model_name: str = "llama-3.1-8b-instant", temperature: float = 0.2) -> ChatGroq:
    return ChatGroq(groq_api_key=api_key, model_name=model_name, temperature=temperature)