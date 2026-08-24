# Simple RAG chatbot with Streamlit, FAISS, LangChain, and Gemma 2B

import io
import os
from typing import Optional, List, Any

import streamlit as st
from pypdf import PdfReader

# Text splitter (supports old/new LangChain layouts)
try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter  # <0.2
except ModuleNotFoundError:
    from langchain_text_splitters import RecursiveCharacterTextSplitter  # >=0.2

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# LangChain Core (v1+)
from langchain_core.language_models.llms import LLM
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough

from pydantic import BaseModel, Field
from huggingface_hub import InferenceClient

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Hugging Face inference token — set HF_TOKEN in the environment (see .env.example).
HF_TOKEN = os.getenv("HF_TOKEN", "")


class GemmaWrapper(LLM):
    """Custom wrapper to use Gemma HF API as a LangChain LLM."""
    client: Any = Field(...)
    max_tokens: int = 400

    @property
    def _llm_type(self) -> str:
        return "gemma_hf_api"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        response = self.client.chat_completion(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=self.max_tokens,
            temperature=0.2,
        )
        return response.choices[0].message["content"]


rag_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=(
        "You are a helpful assistant that answers ONLY using the provided context.\n"
        "If the answer is not clearly in the context, reply exactly with:\n"
        "\"The document does not contain the answer.\"\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}\n\n"
        "Answer in clear bullet points."
    ),
)

splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=100)
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


def load_pdf_text(pdf_bytes: bytes) -> str:
    reader = PdfReader(io.BytesIO(pdf_bytes))
    all_text = ""
    for page in reader.pages:
        t = page.extract_text()
        if t:
            all_text += t + "\n"
    return all_text


@st.cache_resource(show_spinner=False)
def prepare_retriever(pdf_bytes: bytes):
    text = load_pdf_text(pdf_bytes)
    documents = splitter.create_documents([text])
    vectorstore = FAISS.from_documents(documents, embedding_model)
    return vectorstore.as_retriever(search_kwargs={"k": 5})


@st.cache_resource(show_spinner=False)
def build_gemma_llm(token: str) -> GemmaWrapper:
    client = InferenceClient(model="google/gemma-2-2b-it", token=token)
    return GemmaWrapper(client=client)


def format_docs(docs) -> str:
    return "\n\n".join(d.page_content for d in docs)


def build_qa_chain(retriever):
    llm = build_gemma_llm(HF_TOKEN)
    return (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | rag_prompt
        | llm
        | StrOutputParser()
    )


def main():
    st.title("Document Intelligence Assistant")
    st.write(
        "Upload a PDF knowledge base and ask precise questions to instantly surface the answers."
    )

    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])
    user_query = st.text_input(
        "What would you like to know?",
        placeholder="Ask about requirements, timelines, checklists, etc.",
        disabled=uploaded_file is None,
    )

    if uploaded_file is None:
        st.info("Upload a PDF to begin.")
        return

    pdf_bytes = uploaded_file.getvalue()
    with st.spinner("Indexing PDF..."):
        retriever = prepare_retriever(pdf_bytes)
        qa_chain = build_qa_chain(retriever)

    if user_query:
        with st.spinner("Thinking..."):
            answer = qa_chain.invoke(user_query)

        st.subheader("Answer")
        st.write(answer)

        with st.expander("Show retrieved document chunks"):
            docs = retriever.invoke(user_query)
            for i, d in enumerate(docs, 1):
                st.write(f"**Chunk {i}:**")
                st.write(d.page_content)
                st.write("---")


if __name__ == "__main__":
    main()
