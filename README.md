# Document Intelligence Assistant (RAG over PDFs)

A retrieval-augmented question-answering app: upload a PDF, ask questions about it, get
answers grounded strictly in the document's own content — with the retrieved source chunks
shown alongside the answer so you can verify grounding.

## How it works

1. **Ingestion** — the uploaded PDF is parsed (`pypdf`), split into overlapping chunks
   (`RecursiveCharacterTextSplitter`, 400 chars / 100 overlap) to preserve context across
   chunk boundaries.
2. **Indexing** — chunks are embedded with `sentence-transformers/all-MiniLM-L6-v2` and
   indexed in a **FAISS** vector store for similarity search (top-5 retrieval).
3. **Generation** — retrieved chunks are injected into a strict grounding prompt and sent to
   **Gemma-2-2B-it** via the Hugging Face Inference API, wrapped as a custom LangChain `LLM`
   so it composes with LangChain's `PromptTemplate` / `StrOutputParser` runnable chain.
4. **Grounding guardrail** — the prompt explicitly instructs the model to answer only from
   the provided context and to say so plainly when the answer isn't in the document, instead
   of guessing.

Retriever and LLM are cached per-session (`st.cache_resource`) so re-querying the same PDF
doesn't re-embed or re-load the model.

## Tech stack

Streamlit, LangChain (retrieval chain + custom LLM wrapper), FAISS, Hugging Face
`sentence-transformers` (embeddings) + Inference API (Gemma-2-2B-it), pypdf.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # set HF_TOKEN
```

## Run

```bash
streamlit run run.py
```

Upload a PDF, wait for indexing, and ask a question. Expand "Show retrieved document chunks"
to see exactly which passages the answer was grounded in.

## What this demonstrates

A minimal but correct RAG pipeline: chunking strategy, embedding-based retrieval, a
grounding-constrained prompt, and a custom LangChain LLM adapter around a hosted inference
API — the core building blocks behind most production document-QA systems.
