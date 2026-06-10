# RAG PDF Chatbot

This is a chatbot application which makes use of Retrieval-Augmented Generation(RAG). This app allows users to upload multiple pdf files, ask questions related to the content of the pdfs and recieve answers from the pdf text.
The application has another option where there is a general bot mode where users can ask normal conversational questions. The answers are generated using Hugging Face models through LangChain. 


---

## Features

- **PDF Q&A mode** — upload PDFs, ask questions, get answers
- **General conversation mode** — Question about a topic and recieve answers about the question and related ones.
- **Separate chat histories** for each mode there is a separate clear chat option

---

## How It Works

### PDF Reader
The pdf reader extracts text from the uploaded PDF files in a readable format where it leaves behind the formats. If a PDF is scanned with Image based text it may not work.

### Semantic Chunking
After extracting the PDF text the text is divided into smaller chunks based on logical division using RecursiveCharacterTextSplitter to maintain accuracy. 

```python
chunk_size    = 1000   # characters per chunk
chunk_overlap = 200    # characters shared between neighbouring chunks
```
This means each chunk contains 1000 characters, and 200 characters are repeated between neighbouring chunks.


### Embeddings & Vector Store
After chunking each text chunk is converted into a  numerical representation using sentence transformer:

```python
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
```

Vectors are stored in a **FAISS** (Facebook AI Similarity Search) index held in Streamlit session state, enabling fast in-memory similarity search with no database setup required.

```python
vector_store = FAISS.from_texts(texts=chunks, embedding=embeddings)
```

### Semantic Search
 When the user asks a question the app performs a semantic search where it searches for chunks that mean similar to the user's question.

```python
retriever = st.session_state.vector_store.as_retriever(
    search_kwargs={"k": 3}
)
```

Here `k=3` means that the top 3 most relevant chunks are retrieved for each question.


### LLM Integration
 The Language model is integrated using Hugging Face through LangChain.

```python
hf_pipeline = pipeline(
    task="text-generation",
    model="HuggingFaceTB/SmolLM2-135M-Instruct",
    max_new_tokens=128,
    do_sample=False,
    return_full_text=False,
)
```
The model generates answers based on retrieved PDF context.
For PDF questions, the prompt instructs the model to answer only from the PDF context and if not found the model returns

> *"Answer is not available in the PDF context."*

### Chat History
Conversation history is stored in Streamlit `session_state`, with separate lists for PDF Q&A and General Conversation so the two modes never bleed into each other.

---

## Project Structure

```
rag-pdf-chatbot/
├── app.py               # Main Streamlit application
├── requirements.txt     # Python dependencies
├── .env.example         # Template for environment variables
├── .gitignore
└── README.md
```

---
## Setup

### 1. Clone the repository

```bash
git clone https://github.com/haniya02/Multi-PDF-and-Conversational-bot.git

```

### 2. Create and activate a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate        # macOS / Linux
.venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note on `torch` and `transformers`:** torch and transformers are not strictly pinned here because their best versions depend on the user’s Python version, operating system, and CPU/GPU setup. Hugging Face Transformers requires Python 3.10+ and PyTorch 2.4+ for current versions.

### 4. Configure environment variables

```bash
cp .env.example .env
```

Open `.env` and fill in your tokens (only needed if you use gated or private Hugging Face models):

```
HUGGINGFACEHUB_API_TOKEN=your_hugging_face_key_here
HF_TOKEN=your_hugging_face_key_here
```

The default model (`HuggingFaceTB/SmolLM2-135M-Instruct`) is public and requires no token.

### 5. Run the app

```bash
streamlit run app.py
```

---

## Imports Reference

```python
import streamlit as st
from pypdf import PdfReader

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFacePipeline

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from transformers import pipeline

```

---

## Recommendations


**Swap the LLM for a stronger model**
The LLM model `SmolLM2-135M-Instruct` is fast but not capable enough. Better open-source options that still runs locally are `microsoft/Phi-3-mini-4k-instruct`, `mistralai/Mistral-7B-Instruct-v0.2` if you have a GPU.

**Chunk by semantic similarity instead of character count**
`langchain_experimental.text_splitter.SemanticChunker` splits on embedding similarity rather than fixed sizes, which tends to produce more coherent chunks.
