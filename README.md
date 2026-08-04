# PDF Question Answering Application (RAG) - Epochs '26, Assignment 11

## Participant Name
AARON BENNY PHILIP

## MUID
aaronbennyphilip@mulearn

## Project Overview
This is a Retrieval-Augmented Generation (RAG) application that lets a user
upload a PDF, then ask natural-language questions about its contents. The
app retrieves the most relevant chunks of the document and passes them to
an LLM to generate grounded, context-aware answers. Follow-up questions are
supported through conversation memory.

## Technologies Used
- **LangChain** - orchestration of the RAG pipeline
- **PyPDFLoader** - PDF loading/parsing
- **RecursiveCharacterTextSplitter** - document chunking
- **Sentence Transformers** (`all-MiniLM-L6-v2`) - embeddings
- **ChromaDB** - vector store for similarity search
- **Groq API** (`llama-3.1-8b-instant`) - free, fast LLM for response generation
- **Gradio** - interactive web UI

## Memory Implementation
Conversation history is maintained using LangChain's `ConversationBufferMemory`,
wired into a `ConversationalRetrievalChain`. Each new user question is
combined with the prior chat history to reformulate a standalone query before
retrieval, so the app can correctly resolve follow-up questions like "what
about the second one?" that depend on earlier turns.

## Challenges Faced
- **Chunk size tuning**: Too-small chunks fragmented context and hurt retrieval
  quality; too-large chunks diluted relevance and ate up LLM context. Settled
  on 1000 characters with 150-character overlap as a balance between recall
  and precision.
- **Follow-up question resolution**: Vague follow-ups (e.g. "what about the
  second one?") need the prior conversation turns to be resolved into a
  standalone query before retrieval, otherwise the retriever pulls irrelevant
  chunks. `ConversationalRetrievalChain` combined with `ConversationBufferMemory`
  solved this by condensing history + new question before searching.
- **Embedding model load time**: Sentence Transformers' first load is slow
  (downloading + initializing the model), which caused noticeable lag on the
  first PDF upload. Cached the embedding model as a singleton so it only
  loads once per app lifetime instead of per request.
- **Free-tier LLM rate limits**: Groq's free tier has request-per-minute caps,
  so rapid-fire questions during testing occasionally hit 429 errors. Handled
  by keeping temperature low and batching test queries with small delays.
- **Session isolation**: Since multiple users could use the same deployed app,
  each upload needed its own isolated vector store rather than one shared
  Chroma collection, to avoid answers leaking between unrelated PDFs.

## Future Improvements
- **Multi-PDF support**: Allow uploading and querying across several documents
  at once, with source attribution per file.
- **Streaming responses**: Stream LLM tokens to the UI as they're generated
  instead of waiting for the full answer, for a snappier feel.
- **Citation highlighting**: Link answers back to the exact page/paragraph in
  the original PDF (e.g. an embedded PDF viewer that jumps to the cited page).
- **Persistent chat history**: Save conversations per user/session so they
  survive a page refresh, instead of resetting on reload.
- **Offline/local LLM option**: Add an Ollama backend as a fallback so the app
  can run fully offline without depending on a hosted API or rate limits.
- **Better chunking**: Explore semantic or layout-aware chunking (e.g.
  respecting headings/tables) instead of fixed-size character splitting, to
  improve retrieval quality on structured PDFs.

---

## Setup & Running Locally

1. Clone the repo and install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Get a free Groq API key at https://console.groq.com/keys
3. Run the app:
   ```bash
   python app.py
   ```
4. Open the local URL Gradio prints, paste in your Groq API key, upload a
   PDF, click **Process PDF**, and start chatting.

## Deployment
This app can be deployed for free on **Hugging Face Spaces** (Gradio SDK):
1. Create a new Space, choose the **Gradio** SDK.
2. Push `app.py`, `requirements.txt`, and this `README.md` to the Space repo.
3. In Space **Settings → Repository secrets**, add `GROQ_API_KEY` (optional —
   users can also paste their own key in the UI).
4. The Space will build and give you a public URL, that's your deployment link.
