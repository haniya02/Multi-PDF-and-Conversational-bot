import re
import streamlit as st
from pypdf import PdfReader

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFacePipeline

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from transformers import pipeline


# ============================================================
# PDF TEXT FUNCTIONS
# ============================================================

def get_pdf_text(pdf_docs):
    text = ""

    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)

        for page in pdf_reader.pages:
            page_text = page.extract_text()

            if page_text:
                text += page_text + "\n"

    return text


def get_chunks_text(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )

    return text_splitter.split_text(text)


@st.cache_resource
def get_embeddings():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    return embeddings


def get_vectorstore(chunks):
    embeddings = get_embeddings()

    vector_store = FAISS.from_texts(
        texts=chunks,
        embedding=embeddings
    )

    return vector_store


# ============================================================
# LIGHTWEIGHT HUGGING FACE MODEL
# ============================================================

@st.cache_resource
def get_llm():
    hf_pipeline = pipeline(
        task="text-generation",
        model="HuggingFaceTB/SmolLM2-135M-Instruct",
        max_new_tokens=50,
        do_sample=False,
        return_full_text=False,
        repetition_penalty=1.2,
        no_repeat_ngram_size=4
    )

    llm = HuggingFacePipeline(
        pipeline=hf_pipeline
    )

    return llm


# ============================================================
# CLEAN MODEL OUTPUT
# ============================================================

def clean_model_answer(answer):
    answer = answer.strip()

    # Remove common prefixes at the beginning
    answer = re.sub(
        r"^(assistant answer|assistant|answer|ai|response)\s*:\s*",
        "",
        answer,
        flags=re.IGNORECASE
    ).strip()

    # Stop model if it starts inventing extra Q/A examples
    stop_patterns = [
        r"\n\s*Question\s*:",
        r"\n\s*Q\s*:",
        r"\n\s*User\s*:",
        r"\n\s*Human\s*:",
        r"\n\s*Assistant\s*:",
        r"\n\s*Answer\s*:",
        r"\n\s*AI\s*:",
        r"\n\s*Response\s*:"
    ]

    cut_positions = []

    for pattern in stop_patterns:
        match = re.search(pattern, answer, flags=re.IGNORECASE)

        if match:
            cut_positions.append(match.start())

    if cut_positions:
        answer = answer[:min(cut_positions)].strip()

    return answer


# ============================================================
# CHAINS
# ============================================================

@st.cache_resource
def build_pdf_chain():
    llm = get_llm()

    prompt = PromptTemplate.from_template(
        """
You are a helpful PDF question-answering assistant.

Use only the PDF context below to answer the user's question.

Rules:
- Answer only the user's latest question.
- Do not create extra questions.
- Do not write repeated Question/Answer examples.
- Do not continue the conversation by yourself.
- If the answer is not in the PDF context, say:
"Answer is not available in the PDF context."

Previous PDF chat:
{chat_history}

PDF context:
{context}

User question:
{question}

Response:
"""
    )

    chain = prompt | llm | StrOutputParser()

    return chain


@st.cache_resource
def build_conversation_chain():
    llm = get_llm()

    prompt = PromptTemplate.from_template(
        """
You are a friendly general chatbot.

Answer only the user's latest question.

Rules:
- Do not make up new questions.
- Do not write "Question:".
- Do not write "Answer:".
- Do not write fake examples.
- Do not continue the conversation by yourself.
- Stop after answering the user's question.
- Keep the response brief and clear.
- If the user asks for live/current information such as weather, news, stock prices, or today's events, say:
"I do not have access to live real-time data."

Previous conversation:
{chat_history}

User message:
{question}

Response:
"""
    )

    chain = prompt | llm | StrOutputParser()

    return chain


# ============================================================
# CHAT HELPERS
# ============================================================

def make_chat_history_text(history, max_messages=6):
    chat_history_text = ""

    recent_history = history[-max_messages:]

    for message in recent_history:
        role = message["role"]
        content = message["content"]

        if role == "AI":
            content = clean_model_answer(content)

        chat_history_text += f"{role}: {content}\n"

    return chat_history_text


def display_chat_history(history):
    for message in history:
        if message["role"] == "User":
            with st.chat_message("user"):
                st.write(message["content"])
        else:
            with st.chat_message("assistant"):
                st.write(message["content"])


# ============================================================
# PDF QUESTION ANSWERING
# ============================================================

def generate_pdf_output(user_question):
    if st.session_state.vector_store is None:
        st.warning("Please upload and process PDF files first.")
        return

    if "answer_chain" not in st.session_state:
        with st.spinner("Loading PDF question-answering model..."):
            st.session_state.answer_chain = build_pdf_chain()

    retriever = st.session_state.vector_store.as_retriever(
        search_kwargs={"k": 3}
    )

    docs = retriever.invoke(user_question)

    context = "\n\n".join(
        doc.page_content for doc in docs
    )

    chat_history_text = make_chat_history_text(
        st.session_state.pdf_chat_history
    )

    answer = st.session_state.answer_chain.invoke(
        {
            "context": context,
            "question": user_question,
            "chat_history": chat_history_text
        }
    )

    answer = clean_model_answer(answer)

    st.session_state.pdf_chat_history.append(
        {
            "role": "User",
            "content": user_question
        }
    )

    st.session_state.pdf_chat_history.append(
        {
            "role": "AI",
            "content": answer
        }
    )


# ============================================================
# GENERAL CONVERSATION
# ============================================================

def generate_general_output(user_question):
    if "conversation_chain" not in st.session_state:
        with st.spinner("Loading chatbot model..."):
            st.session_state.conversation_chain = build_conversation_chain()

    chat_history_text = make_chat_history_text(
        st.session_state.general_chat_history
    )

    answer = st.session_state.conversation_chain.invoke(
        {
            "question": user_question,
            "chat_history": chat_history_text
        }
    )

    answer = clean_model_answer(answer)

    st.session_state.general_chat_history.append(
        {
            "role": "User",
            "content": user_question
        }
    )

    st.session_state.general_chat_history.append(
        {
            "role": "AI",
            "content": answer
        }
    )


# ============================================================
# MAIN APP
# ============================================================

def main():
    st.set_page_config(
        page_title="PDF RAG Chatbot",
        layout="wide"
    )

    st.header("PDF Q&A + General Chatbot")

    # Lightweight session initialization only.
    # The model is not loaded until the first question is asked.

    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None

    if "pdf_chat_history" not in st.session_state:
        st.session_state.pdf_chat_history = []

    if "general_chat_history" not in st.session_state:
        st.session_state.general_chat_history = []

    with st.sidebar:
        st.header("Choose Mode")

        mode = st.radio(
            "Select an option",
            [
                "Read PDF and Answer Questions",
                "General Conversation"
            ]
        )

        st.divider()

        if mode == "Read PDF and Answer Questions":
            st.header("Upload PDF files")

            pdf_docs = st.file_uploader(
                "Choose PDF files",
                type=["pdf"],
                accept_multiple_files=True
            )

            if st.button("Upload and Process"):
                if not pdf_docs:
                    st.warning("Please upload at least one PDF file.")
                    return

                with st.spinner("Reading and processing PDFs..."):
                    raw_text = get_pdf_text(pdf_docs)

                    if not raw_text.strip():
                        st.warning("No extractable text found. The PDF may be scanned.")
                        return

                    chunks = get_chunks_text(raw_text)

                    if not chunks:
                        st.warning("No text chunks were created.")
                        return

                    st.session_state.vector_store = get_vectorstore(chunks)
                    st.session_state.pdf_chat_history = []

                    st.success("PDFs processed successfully. You can now ask PDF questions.")

            if st.button("Clear PDF Chat"):
                st.session_state.pdf_chat_history = []
                st.rerun()

        else:
            st.header("Conversation")

            if st.button("Clear Conversation"):
                st.session_state.general_chat_history = []
                st.rerun()

        st.divider()

        if st.button("Clear All Cache"):
            st.session_state.clear()
            st.cache_resource.clear()
            st.rerun()

    if mode == "Read PDF and Answer Questions":
        st.subheader("PDF Q&A Mode")

        display_chat_history(
            st.session_state.pdf_chat_history
        )

        user_question = st.chat_input(
            "Ask a question related to the uploaded PDF"
        )

        if user_question:
            generate_pdf_output(user_question)
            st.rerun()

    else:
        st.subheader("General Conversation Mode")

        st.info(
            "You can ask general questions. This local Hugging Face model does not have live real-time data."
        )

        display_chat_history(
            st.session_state.general_chat_history
        )

        user_question = st.chat_input(
            "Ask anything..."
        )

        if user_question:
            generate_general_output(user_question)
            st.rerun()


if __name__ == "__main__":
    main()