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
- **PyPDFLoader** (LangChain) - PDF loading/parsing
- **RecursiveCharacterTextSplitter** - document chunking (1000 chars, 150 overlap)
- **Sentence Transformers** (`all-MiniLM-L6-v2`, run locally) - embeddings
- **FAISS** - in-memory vector store for similarity search
- **Groq API** (`llama-3.1-8b-instant`) - free, fast LLM for response generation
- **Streamlit** - interactive web UI

## Memory Implementation
Conversation history is kept in a small custom `ConversationMemory` class
rather than a LangChain memory abstraction. Each turn is stored as a
(role, content) pair; on every new question, the full history is converted
to message objects and replayed to the LLM alongside the newly retrieved
context, so the model can resolve follow-up questions like "what about the
second one?" that depend on earlier turns. Keeping this logic explicit
avoids depending on LangChain memory classes that have moved between
packages across recent releases.

## Challenges Faced
- **Chunk size tuning**: Too-small chunks fragmented context and hurt retrieval
  quality; too-large chunks diluted relevance and ate up LLM context. Settled
  on 1000 characters with 150-character overlap as a balance between recall
  and precision.
- **Dependency and environment issues**: Early attempts using Gradio and
  ChromaDB ran into repeated Windows-specific problems: a schema-generation
  bug in a specific Gradio release, ChromaDB's `chroma-hnswlib` dependency
  needing a C++ compiler unavailable on the machine, and a local network that
  blocked access to `localhost`, which Gradio explicitly checks for and
  refuses to start without. Switching to Streamlit (no localhost-reachability
  check) and FAISS (ships prebuilt wheels, no compiler needed) resolved all
  three at once.
- **LangChain version drift**: Pinning exact old LangChain versions caused
  import errors against newer Python releases, since APIs like
  `ConversationalRetrievalChain` and `langchain.memory` were restructured
  across versions. Resolved by using loosely-pinned, current package versions
  and writing a small custom memory class instead of relying on LangChain's
  memory abstractions.
- **Free-tier LLM rate limits**: Groq's free tier has request-per-minute
  caps, so rapid-fire questions during testing occasionally hit errors.
  Handled with a clear on-screen message suggesting the user wait and retry.
- **Embedding provider outage**: An initial version used Google's Gemini
  embeddings API, but hit a `403 PERMISSION_DENIED` error tied to the Gemini
  project itself (a known, currently-widespread issue on Google's side,
  unrelated to this app's code or API key validity). Switched to running
  embeddings locally with Sentence Transformers, removing the dependency on
  any single provider's account approval for that step.
- **Session isolation**: Since multiple users could use the same deployed app,
  each upload's vector store, memory, and chat history needed to live in that
  user's own Streamlit session state, to avoid answers leaking between
  unrelated PDFs or users.

## Future Improvements
- **Multi-PDF support**: Allow uploading and querying across several documents
  at once, with source attribution per file.
- **Streaming responses**: Stream LLM tokens to the UI as they're generated
  instead of waiting for the full answer, for a snappier feel.
- **Citation highlighting**: Link answers back to the exact page/paragraph in
  the original PDF (eg: an embedded PDF viewer that jumps to the cited page).
- **Persistent chat history**: Save conversations per user so they survive a
  page refresh, instead of resetting on reload.
- **Offline/local LLM option**: Add an Ollama backend as a fallback so the app
  can run fully offline without depending on a hosted API or rate limits.
- **Better chunking**: Explore semantic or layout-aware chunking (eg:
  respecting headings/tables) instead of fixed-size character splitting, to
  improve retrieval quality on structured PDFs.

---

## Setup & Running Locally

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Get a free Groq API key at https://console.groq.com/keys
3. Run the app:
   ```bash
   streamlit run app.py
   ```
4. Streamlit opens automatically in your browser (usually at
   `http://localhost:8501`). Paste in your Groq API key in the sidebar,
   upload a PDF, and start chatting.

## Deployment
This app can be deployed for free on **Streamlit Community Cloud**:
1. Push `app.py`, `rag_pipeline.py`, and `requirements.txt` to a public
   GitHub repository.
2. Go to https://share.streamlit.io, sign in, and click **New app**.
3. Point it at your repo and `app.py` as the entry file.
4. In the app's **Settings → Secrets**, add:
   ```toml
   GROQ_API_KEY = "your-key-here"
   ```
5. Deploy — you'll get a public URL like
   `https://your-app-name.streamlit.app`. That's your deployment link.
