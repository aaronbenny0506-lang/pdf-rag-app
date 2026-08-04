"""
Epochs '26 - Assignment 11: PDF Question Answering Application (RAG)
---------------------------------------------------------------------
Loads a PDF, chunks + embeds it, stores it in a ChromaDB vector store,
and answers user questions about it using a free LLM (Groq API) with
conversation memory for natural follow-up questions.

Run locally:
    export GROQ_API_KEY="your-key-here"   # get one free at https://console.groq.com
    pip install -r requirements.txt
    python app.py
"""

import os
import shutil
import tempfile

import gradio as gr
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "llama-3.1-8b-instant"   # fast + free on Groq
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
PERSIST_ROOT = os.path.join(tempfile.gettempdir(), "pdf_rag_chroma")

# Loaded once, reused across requests
_embeddings = None


def get_embeddings():
    """Lazily load the embedding model once (it's slow to init)."""
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return _embeddings


# ---------------------------------------------------------------------
# Core RAG pipeline
# ---------------------------------------------------------------------
def build_chain(pdf_path: str, api_key: str):
    """Load a PDF, chunk it, embed it, and build a conversational
    retrieval chain with memory."""

    if not api_key:
        raise gr.Error("Please enter your Groq API key first (get a free one at console.groq.com).")

    if not pdf_path:
        raise gr.Error("Please upload a PDF file first.")

    # 1. Load the PDF
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    # 2. Split into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(documents)

    # 3. Embed + store in a fresh Chroma collection for this session
    persist_dir = os.path.join(PERSIST_ROOT, next(tempfile._get_candidate_names()))
    os.makedirs(persist_dir, exist_ok=True)

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=get_embeddings(),
        persist_directory=persist_dir,
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    # 4. Free LLM via Groq
    llm = ChatGroq(
        groq_api_key=api_key,
        model_name=LLM_MODEL,
        temperature=0.2,
    )

    # 5. Conversation memory so follow-up questions work naturally
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer",
    )

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
    )

    return chain, persist_dir


# ---------------------------------------------------------------------
# Gradio callbacks
# ---------------------------------------------------------------------
def process_pdf(pdf_file, api_key, state):
    """Called when the user uploads a PDF / clicks 'Process PDF'."""
    # Clean up any previous session's vector store
    if state.get("persist_dir") and os.path.exists(state["persist_dir"]):
        shutil.rmtree(state["persist_dir"], ignore_errors=True)

    chain, persist_dir = build_chain(pdf_file, api_key)
    state["chain"] = chain
    state["persist_dir"] = persist_dir
    return state, "✅ PDF processed! You can start asking questions below.", []


def chat(user_message, chat_history, state):
    if not user_message:
        return "", chat_history, state

    chain = state.get("chain")
    if chain is None:
        chat_history = chat_history + [(user_message, "⚠️ Please upload and process a PDF first.")]
        return "", chat_history, state

    result = chain.invoke({"question": user_message})
    answer = result["answer"]

    # Optionally show which pages the answer came from
    sources = result.get("source_documents", [])
    pages = sorted({str(doc.metadata.get("page", "?")) for doc in sources})
    if pages:
        answer += f"\n\n*Source page(s): {', '.join(pages)}*"

    chat_history = chat_history + [(user_message, answer)]
    return "", chat_history, state


def reset_session(state):
    if state.get("persist_dir") and os.path.exists(state["persist_dir"]):
        shutil.rmtree(state["persist_dir"], ignore_errors=True)
    return {}, [], "Session reset. Upload a new PDF to begin."


# ---------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------
with gr.Blocks(title="PDF Q&A with RAG") as demo:
    gr.Markdown(
        """
        # 📄 PDF Question Answering (RAG)
        Upload a PDF, click **Process PDF**, then ask questions about it below.
        The assistant remembers the conversation so you can ask natural follow-ups.
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            api_key_box = gr.Textbox(
                label="Groq API Key",
                placeholder="gsk_...",
                type="password",
                info="Free key: https://console.groq.com/keys",
            )
            pdf_upload = gr.File(label="Upload PDF", file_types=[".pdf"], type="filepath")
            process_btn = gr.Button("🔍 Process PDF", variant="primary")
            status_box = gr.Textbox(label="Status", interactive=False)
            reset_btn = gr.Button("♻️ Reset Session")

        with gr.Column(scale=2):
            chatbot = gr.Chatbot(label="Chat", height=500)
            msg_box = gr.Textbox(label="Ask a question", placeholder="What is this document about?")
            clear_btn = gr.Button("Clear Chat")

    session_state = gr.State({})

    process_btn.click(
        fn=process_pdf,
        inputs=[pdf_upload, api_key_box, session_state],
        outputs=[session_state, status_box, chatbot],
    )

    msg_box.submit(
        fn=chat,
        inputs=[msg_box, chatbot, session_state],
        outputs=[msg_box, chatbot, session_state],
    )

    clear_btn.click(fn=lambda: [], outputs=[chatbot])

    reset_btn.click(
        fn=reset_session,
        inputs=[session_state],
        outputs=[session_state, chatbot, status_box],
    )


if __name__ == "__main__":
    demo.launch()
