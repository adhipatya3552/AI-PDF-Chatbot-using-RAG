# 🤖 RAG PDF Chatbot

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-Web%20App-red?style=for-the-badge&logo=streamlit&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-RAG%20Framework-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white)
![FAISS](https://img.shields.io/badge/FAISS-Vector%20Search-blue?style=for-the-badge)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Embeddings-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)
![OpenRouter](https://img.shields.io/badge/OpenRouter-GPT--3.5--turbo-412991?style=for-the-badge)

**A fully working AI-powered PDF chatbot built with Retrieval-Augmented Generation (RAG). Upload any PDF, ask questions about it in natural language, and get accurate, context-aware answers — powered by a local FAISS vector store, HuggingFace embeddings, and GPT-3.5-turbo via OpenRouter.**

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [How RAG Works in This Project](#-how-rag-works-in-this-project)
- [Architecture](#-architecture)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Getting Started](#-getting-started)
- [Running the Project](#-running-the-project)
- [Module Reference](#-module-reference)
- [Environment Variables](#-environment-variables)
- [Known Limitations](#-known-limitations)
- [Roadmap](#-roadmap)

---

## 🏗 Overview

This project is a **RAG-based PDF Chatbot** built entirely in Python. It allows you to upload one or more PDF files and ask questions about their content in plain English — and get accurate, grounded answers from the LLM.

It works using a technique called **Retrieval-Augmented Generation (RAG)**:

- Instead of sending the entire PDF to the LLM (which is too large), the PDF is split into small chunks and converted into numerical vectors (embeddings)
- These vectors are stored in a **FAISS vector database** — a local, blazing-fast similarity search engine
- When you ask a question, the most relevant chunks are retrieved from FAISS and passed as **context** to the LLM
- The LLM (GPT-3.5-turbo via OpenRouter) reads only that context and answers your question — no hallucination from outside knowledge

You can run it as a **Streamlit web app** using `app/app.py`, or build the vector database from a local `data/` folder using `main.py`.

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 📄 **Multi-PDF Upload** | Upload one or more PDFs simultaneously from the browser sidebar |
| ⚡ **One-Click Processing** | Click "Process PDFs" to chunk, embed, and index everything into FAISS |
| 🔍 **Semantic Search** | FAISS retrieves the 4 most relevant text chunks for every question |
| 🤖 **LLM-Powered Answers** | GPT-3.5-turbo (via OpenRouter) answers using only the retrieved context |
| 💬 **Persistent Chat History** | All messages are saved to `chat_history/chat_history.json` and reload on app restart |
| 🗑️ **Clear Chat** | One-click button to clear all messages from session and disk |
| ⬇️ **Download Chat as TXT** | Download the full conversation as a plain-text `.txt` file |
| 📄 **Export Chat as PDF** | Export the entire chat history as a formatted `.pdf` file using ReportLab |
| ⚠️ **Toast Warnings** | Animated, auto-dismissing toast notifications for user errors (no PDF uploaded, empty chat, etc.) |
| 🎨 **Dark-Mode UI** | Clean dark-themed Streamlit interface with custom CSS chat bubbles |
| 🧠 **Context-Grounded Prompting** | Prompt template enforces "answer from context only" — no hallucination |
| 💾 **Persistent Vector DB** | FAISS index is saved to `vector_db/` after every processing run |

---

## ⚙️ How RAG Works in This Project

RAG (Retrieval-Augmented Generation) is the core idea behind this project. Here is a step-by-step breakdown of what happens from PDF upload to final answer:

---

### Step 1: 📥 PDF Loading (`src/pdf_loader.py`)

When you click "Process PDFs", the app saves the uploaded files into the `uploaded_pdfs/` folder and then loads them using LangChain's `PyPDFLoader`.

Each page of each PDF becomes a `Document` object that carries:
- The extracted text content (`page_content`)
- Metadata like the page number and source filename (`metadata`)

```python
loader = PyPDFLoader("uploaded_pdfs/my_document.pdf")
documents = loader.load()
# Returns a list of Document objects, one per page
```

---

### Step 2: ✂️ Text Chunking (`src/text_chunker.py`)

PDFs can be huge. Sending a 50-page document to the LLM all at once is not practical — LLMs have token limits, and it's also slow and expensive. So we split each document into smaller **chunks** using LangChain's `RecursiveCharacterTextSplitter`.

Each chunk is at most **500 characters long**, with an **overlap of 100 characters** between consecutive chunks. The overlap ensures that important context at the boundary between two chunks is not lost.

```python
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)
chunks = splitter.split_documents(documents)
```

> **Why 500 characters?** It's a good balance — small enough to fit meaningfully in a retrieval context, and large enough to contain a coherent passage of text.

---

### Step 3: 🧬 Embedding (`src/embedding_model.py`)

Each text chunk is then converted into a **dense numerical vector** (embedding) using the `all-MiniLM-L6-v2` model from HuggingFace via `langchain-huggingface`.

Embeddings capture the **semantic meaning** of text. Two chunks that say the same thing in different words will have very similar (close) embedding vectors. This is what makes semantic search possible.

```python
from langchain_huggingface import HuggingFaceEmbeddings

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
```

> `all-MiniLM-L6-v2` is a lightweight but powerful sentence transformer model. It produces 384-dimensional embeddings and runs locally on CPU — no API key needed for this step.

---

### Step 4: 🗃️ Vector Store Creation (`src/vector_store.py`)

All chunk embeddings are stored in a **FAISS (Facebook AI Similarity Search)** vector database. FAISS is an extremely fast library for finding similar vectors in a large collection.

The FAISS index is saved to disk at `vector_db/` so it persists between sessions.

```python
from langchain_community.vectorstores import FAISS

vector_db = FAISS.from_documents(chunks, embedding_model)
vector_db.save_local("vector_db")
```

---

### Step 5: 🔍 Retrieval (`src/retriever.py`)

When the user types a question, it is embedded using the same model into a query vector. FAISS then performs **cosine similarity search** across all stored chunk vectors and returns the **top 4 most relevant chunks**.

```python
retriever = vector_db.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 4}
)
docs = retriever.invoke(query)
```

> Only the 4 most relevant chunks are passed forward — not the entire document. This keeps the LLM prompt short, focused, and accurate.

---

### Step 6: 📝 Prompt Construction (`src/prompt_template.py`)

The retrieved chunks are combined with the user's question into a structured prompt. The prompt instructs the LLM to answer **only from the provided context**, and to say "I could not find the answer in the document" if the answer is not present.

Each chunk is labeled with its page number so the model can reference it:

```
Source 1 (Page 3):
<chunk text here>

Source 2 (Page 7):
<chunk text here>
...

Question:
What is the main conclusion of the paper?

Answer:
```

---

### Step 7: 🤖 LLM Response (`src/llm_response.py`)

The final prompt is sent to **GPT-3.5-turbo** via the **OpenRouter API**. OpenRouter is an API gateway that allows you to call multiple LLMs (OpenAI, Anthropic, Mistral, etc.) through a single unified endpoint using one API key.

```python
response = requests.post(
    url="https://openrouter.ai/api/v1/chat/completions",
    headers={"Authorization": f"Bearer {API_KEY}"},
    json={
        "model": "openai/gpt-3.5-turbo",
        "messages": [{"role": "user", "content": prompt}]
    }
)
```

The model's answer is extracted from the response and displayed in the chat UI.

---

### Step 8: 💾 Chat Storage (`src/chat_storage.py`)

Every user message and assistant response is saved to `chat_history/chat_history.json`. The file is loaded on app startup, so your conversation survives page refreshes.

```python
def save_chat(messages):
    with open("chat_history/chat_history.json", "w") as f:
        json.dump(messages, f, indent=4)

def load_chat():
    if os.path.exists(CHAT_FILE):
        with open(CHAT_FILE, "r") as f:
            return json.load(f)
    return []
```

---

## 🏛 Architecture

```
┌────────────────────────────────────────────────────────────────┐
│                        USER (Browser)                          │
│              Uploads PDFs + Types Questions                     │
└────────────────────────────┬───────────────────────────────────┘
                             │
              ┌──────────────▼──────────────┐
              │     app/app.py              │
              │  (Streamlit Web Interface)  │
              └────┬──────────────────┬─────┘
                   │                  │
     ┌─────────────▼──────┐    ┌──────▼──────────────────┐
     │  PDF PROCESSING    │    │   QUESTION ANSWERING     │
     │  PIPELINE          │    │   PIPELINE               │
     │                    │    │                          │
     │  pdf_processor.py  │    │  retriever.py            │
     │       │            │    │  (FAISS similarity k=4)  │
     │  pdf_loader.py     │    │       │                  │
     │  (PyPDFLoader)     │    │  prompt_template.py      │
     │       │            │    │  (Context + Question)    │
     │  text_chunker.py   │    │       │                  │
     │  (500 chars, 100   │    │  llm_response.py         │
     │   overlap)         │    │  (OpenRouter GPT-3.5)    │
     │       │            │    └──────────────────────────┘
     │  embedding_model   │
     │  (MiniLM-L6-v2)    │
     │       │            │
     │  vector_store.py   │
     │  (FAISS Index)     │
     └────────────────────┘
                   │
     ┌─────────────▼──────────────────────────┐
     │           STORAGE LAYER                 │
     │                                         │
     │  vector_db/         ← FAISS index       │
     │  chat_history/      ← JSON messages     │
     │  uploaded_pdfs/     ← temp PDF files    │
     └─────────────────────────────────────────┘
```

**End-to-end Data Flow:**

```
User uploads PDFs
        │
        ▼
pdf_processor.py  ──▶  Saves to uploaded_pdfs/
        │
        ▼
pdf_loader.py  ──▶  PyPDFLoader  ──▶  List[Document]
        │
        ▼
text_chunker.py  ──▶  RecursiveCharacterTextSplitter  ──▶  List[Chunk]
        │
        ▼
embedding_model.py  ──▶  all-MiniLM-L6-v2  ──▶  Embeddings
        │
        ▼
vector_store.py  ──▶  FAISS.from_documents()  ──▶  vector_db/
        │
        ▼ (at question time)
retriever.py  ──▶  similarity search (k=4)  ──▶  Top 4 Chunks
        │
        ▼
prompt_template.py  ──▶  Context + Question  ──▶  Final Prompt
        │
        ▼
llm_response.py  ──▶  OpenRouter API (GPT-3.5-turbo)  ──▶  Answer
        │
        ▼
chat_storage.py  ──▶  Saved to chat_history.json
        │
        ▼
app.py  ──▶  Displayed in chat UI
```

---

## 🛠 Tech Stack

| Layer | Technology | Purpose |
|-------|------------|---------|
| **Language** | Python 3.10+ | Core programming language |
| **Web UI** | Streamlit 1.45.1 | Browser-based chat interface with sidebar |
| **RAG Framework** | LangChain 0.3.25 | Orchestrates PDF loading, chunking, retrieval |
| **PDF Parsing** | pypdf 5.5.0 + LangChain's PyPDFLoader | Extracts text and metadata from PDFs |
| **Text Splitting** | LangChain RecursiveCharacterTextSplitter | Splits pages into 500-char chunks with 100-char overlap |
| **Embeddings** | sentence-transformers/all-MiniLM-L6-v2 | Converts text chunks to 384-dim semantic vectors |
| **HuggingFace Integration** | langchain-huggingface 0.2.0 | Loads and runs the embedding model locally |
| **Vector Database** | FAISS (faiss-cpu 1.11.0) | Fast local similarity search over chunk embeddings |
| **LLM** | GPT-3.5-turbo (via OpenRouter) | Generates answers from retrieved context |
| **LLM API Gateway** | OpenRouter API | Unified endpoint for calling multiple LLMs |
| **Chat Export** | ReportLab 4.4.0 | Generates formatted PDF exports of chat history |
| **Env Management** | python-dotenv 1.1.0 | Loads API keys from `.env` file |
| **HTTP Client** | requests 2.32.3 | Sends requests to the OpenRouter API |

---

## 📁 Project Structure

```
rag-pdf-chatbot/
│
├── app/
│   └── app.py                     # Streamlit web app — main UI entry point
│
├── src/
│   ├── pdf_loader.py              # Loads PDFs from a folder using PyPDFLoader
│   ├── pdf_processor.py           # Orchestrates full PDF pipeline for Streamlit uploads
│   ├── text_chunker.py            # Splits documents into 500-char chunks (100 overlap)
│   ├── embedding_model.py         # Loads HuggingFace all-MiniLM-L6-v2 embedding model
│   ├── vector_store.py            # Creates FAISS vector DB from chunks and saves it locally
│   ├── retriever.py               # Retrieves top-4 relevant chunks via similarity search
│   ├── prompt_template.py         # Builds the RAG prompt (context + question + rules)
│   ├── llm_response.py            # Calls OpenRouter API (GPT-3.5-turbo) and returns answer
│   ├── chat_storage.py            # Saves and loads chat history from JSON
│   ├── pdf_export.py              # Exports chat history as a PDF using ReportLab
│   ├── config.py                  # (Reserved for future config constants)
│   └── utils.py                   # (Reserved for future utility functions)
│
├── chat_history/
│   └── chat_history.json          # Persisted chat messages (auto-created)
│
├── uploaded_pdfs/                 # Temp folder — uploaded PDFs are saved here before processing
│
├── vector_db/
│   ├── index.faiss                # FAISS vector index (binary)
│   └── index.pkl                  # FAISS metadata/docstore (pickled)
│
├── data/                          # (For CLI use) Place PDFs here and run main.py
│
├── notebooks/
│   └── experiments.ipynb          # Jupyter notebook for experimenting with the pipeline
│
├── screenshots/                   # Folder for storing UI screenshots
│
├── main.py                        # CLI script — builds vector DB from data/ folder
├── requirements.txt               # Python dependencies with pinned versions
└── .env                           # API keys (OPENROUTER_API_KEY, HF_TOKEN)
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10 or above
- pip
- An [OpenRouter](https://openrouter.ai/) account and API key (free tier available)
- Optionally, a HuggingFace account and token (for private models — not required for `all-MiniLM-L6-v2`)

---

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/rag-pdf-chatbot.git
cd rag-pdf-chatbot
```

---

### 2. Create a Virtual Environment

It is strongly recommended to use a virtual environment so all dependencies are installed in isolation and don't conflict with your system Python packages.

```bash
# Create virtual environment
python -m venv venv

# Activate — Windows
venv\Scripts\activate

# Activate — Mac/Linux
source venv/bin/activate
```

---

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

The `requirements.txt` installs the following pinned versions:

```
langchain
langchain-community
langchain-huggingface
faiss-cpu
streamlit
sentence-transformers
pypdf
python-dotenv
requests
numpy
pandas
reportlab
```

---

### 4. Set Up Environment Variables

Create a `.env` file in the root of the project:

```bash
touch .env
```

Add the following keys:

```
OPENROUTER_API_KEY=your_openrouter_api_key_here
HF_TOKEN=your_huggingface_token_here
```

- **`OPENROUTER_API_KEY`** — Get this from [https://openrouter.ai/keys](https://openrouter.ai/keys). This is required to call GPT-3.5-turbo.
- **`HF_TOKEN`** — Get this from [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens). Only required if you switch to a gated/private HuggingFace model. The default model (`all-MiniLM-L6-v2`) is public and does not need a token.

> ⚠️ **Never commit your `.env` file to GitHub.** Add it to `.gitignore`.

---

## ▶️ Running the Project

> 🌐 **This project is deployed on Streamlit Cloud — try it live:**
> **[https://ai-pdf-chatbot-using-rag.streamlit.app/](https://ai-pdf-chatbot-using-rag.streamlit.app/)**

### Option 1 — Streamlit Web App (Recommended)


```bash
streamlit run app/app.py
```

The app opens in your browser at `http://localhost:8501`.

**How to use it:**

1. In the **left sidebar**, upload one or more PDF files using the file uploader
2. Click **⚡ Process PDFs** — the app will load, chunk, embed, and index all the PDFs into FAISS
3. Once you see "PDFs processed successfully!", type your question in the chat input at the bottom
4. The chatbot retrieves relevant chunks from your PDFs and generates a grounded answer
5. Your conversation is automatically saved and will reload if you refresh the page

**Sidebar Options:**

| Button | What it does |
|--------|--------------|
| ⚡ Process PDFs | Runs the full RAG pipeline on uploaded files |
| 🗑️ Clear Chat History | Deletes all messages from session and disk |
| ⬇ Download Chat | Downloads the chat as a `.txt` file |
| 📄 Export Chat as PDF | Downloads the chat as a formatted `.pdf` file |

> If you try to chat without processing PDFs first, or try to export an empty chat, you'll see an animated toast warning in the bottom-right corner.

---

### Option 2 — CLI (Build Vector DB Locally)

If you want to pre-build the vector database from PDFs stored in the `data/` folder (without using the Streamlit UI), use `main.py`:

1. Place your PDF files inside the `data/` folder
2. Run:

```bash
python main.py
```

This will:
- Load all PDFs from `data/`
- Split them into chunks
- Create embeddings using the HuggingFace model
- Save the FAISS index to `vector_db/`

> Note: `main.py` only builds the vector database. To query it, you still need to use the Streamlit app.

---

## 📦 Module Reference

### `src/pdf_loader.py`

Scans a given folder for `.pdf` files and loads each one using LangChain's `PyPDFLoader`. Returns a flat list of `Document` objects (one per page), each with `page_content` and `metadata` (page number, source file).

```python
load_pdfs(folder_path: str) -> List[Document]
```

---

### `src/pdf_processor.py`

Orchestrates the entire PDF processing pipeline for Streamlit file uploads. It:
1. Clears any previously uploaded PDFs from `uploaded_pdfs/`
2. Saves the new uploaded files to disk
3. Calls `load_pdfs()`, `split_documents()`, `load_embedding_model()`, and `create_vector_store()` in sequence
4. Returns the ready-to-use FAISS vector database object

```python
process_uploaded_pdfs(uploaded_files: List[UploadedFile]) -> FAISS
```

---

### `src/text_chunker.py`

Wraps LangChain's `RecursiveCharacterTextSplitter` with `chunk_size=500` and `chunk_overlap=100`. Takes a list of `Document` objects and returns a larger list of smaller `Document` chunks.

```python
split_documents(documents: List[Document]) -> List[Document]
```

---

### `src/embedding_model.py`

Loads the `sentence-transformers/all-MiniLM-L6-v2` model from HuggingFace using `langchain-huggingface`. The model runs entirely locally on CPU — no API call is made during embedding.

```python
load_embedding_model() -> HuggingFaceEmbeddings
```

---

### `src/vector_store.py`

Takes a list of document chunks and an embedding model, creates a FAISS vector database from them, saves it locally to `vector_db/`, and returns the database object.

```python
create_vector_store(chunks: List[Document], embedding_model: HuggingFaceEmbeddings) -> FAISS
```

---

### `src/retriever.py`

Wraps the FAISS vector database as a LangChain retriever using cosine similarity search. Returns the top 4 most semantically similar chunks for a given query string.

```python
retrieve_chunks(vector_db: FAISS, query: str) -> List[Document]
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `vector_db` | `FAISS` | The loaded FAISS vector store |
| `query` | `str` | The user's question |

**Returns:** A list of up to 4 `Document` objects with `page_content` and `metadata` (page number).

---

### `src/prompt_template.py`

Builds the final prompt string that gets sent to the LLM. Combines the retrieved context chunks (labeled by source and page number) with the user's question and a strict set of answering rules.

```python
create_prompt(context: str, question: str) -> str
```

The prompt enforces:
- Answer only from the provided context
- Use bullet points when helpful
- If the answer is not found, say so explicitly — do not make up information

---

### `src/llm_response.py`

Sends the constructed prompt to the OpenRouter API endpoint using the `openai/gpt-3.5-turbo` model. Returns the plain text answer from the LLM response.

```python
generate_response(prompt: str) -> str
```

> Requires `OPENROUTER_API_KEY` to be set in the `.env` file.

---

### `src/chat_storage.py`

Manages reading and writing the chat history to `chat_history/chat_history.json`.

```python
save_chat(messages: List[dict]) -> None
load_chat() -> List[dict]
```

Each message is a dictionary: `{"role": "user" | "assistant", "content": "..."}`.

---

### `src/pdf_export.py`

Converts the chat history (list of message dicts) into a formatted PDF document using ReportLab and returns it as a `BytesIO` buffer, which Streamlit can serve directly as a download.

```python
export_chat_to_pdf(messages: List[dict]) -> BytesIO
```

---

## 🔑 Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENROUTER_API_KEY` | ✅ Yes | API key for calling GPT-3.5-turbo via OpenRouter |
| `HF_TOKEN` | ⚠️ Optional | HuggingFace token — only needed for private/gated models |

---

## ⚠️ Known Limitations

| Issue | Details |
|-------|---------|
| **PDF re-processing required** | Every time you upload new PDFs, you must click "Process PDFs" again — the app doesn't auto-detect new files |
| **No memory across questions** | The LLM sees only the retrieved chunks for each question independently — it doesn't remember what was said earlier in the chat |
| **OCR not supported** | Scanned PDFs (images of text) cannot be read — only text-layer PDFs work with PyPDFLoader |
| **GPT-3.5-turbo only** | The LLM is hardcoded to `openai/gpt-3.5-turbo` in `llm_response.py` — changing the model requires editing the code |
| **Local FAISS only** | The vector database is saved locally to `vector_db/`. It resets when you re-process PDFs |
| **Single session context** | The app only answers from PDFs uploaded in the current processing run. Previously indexed PDFs are overwritten |
| **config.py and utils.py are empty** | These files are placeholders for future expansion |
| **No streaming** | Responses from the LLM are returned all at once — there is no token-by-token streaming in the Streamlit UI |

---

## 🗺️ Roadmap

- [x] PDF upload and processing via Streamlit sidebar
- [x] Text chunking with RecursiveCharacterTextSplitter
- [x] Local HuggingFace embeddings (all-MiniLM-L6-v2)
- [x] FAISS vector store with local persistence
- [x] Semantic retrieval (top-4 chunks)
- [x] RAG prompt template with grounding rules
- [x] GPT-3.5-turbo responses via OpenRouter API
- [x] Persistent chat history (JSON)
- [x] Clear chat functionality
- [x] Download chat as .txt
- [x] Export chat as PDF (ReportLab)
- [x] Animated toast warnings for user errors
- [x] Dark-mode custom UI with styled chat bubbles
- [ ] Add streaming LLM responses (token-by-token display)
- [ ] Add source page citations directly in the chat response
- [ ] Support for scanned PDFs via OCR (pytesseract or similar)
- [ ] Allow switching LLM models from the sidebar (GPT-4, Claude, Mistral, etc.)
- [ ] Add conversation memory so the LLM remembers previous questions
- [ ] Support uploading other file types (DOCX, TXT, CSV)
- [x] Deploy on Streamlit Community Cloud
- [ ] Add a "which page does this come from?" source attribution toggle

---

<div align="center">

Built with ❤️ using Python, LangChain, FAISS, HuggingFace, and Streamlit.

</div>
