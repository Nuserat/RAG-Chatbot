import os
import time

import streamlit as st

from langchain_core.prompts import (
    PromptTemplate
)

from langchain_huggingface import (
    HuggingFaceEmbeddings
)

from langchain_community.vectorstores import (
    FAISS
)

from langchain_groq import ChatGroq


# ============================================================
# CONFIGURATION
# ============================================================

INDEXES = {

    "Fixed 250 / 25":
        "vectorstore/fixed_250",

    "Fixed 500 / 50":
        "vectorstore/fixed_500",

    "Fixed 750 / 75":
        "vectorstore/fixed_750",

    "Fixed 1000 / 100":
        "vectorstore/fixed_1000",

    "Fixed 500 / 0":
        "vectorstore/fixed_500_overlap_0",

    "Fixed 500 / 25":
        "vectorstore/fixed_500_overlap_25",

    "Fixed 500 / 50 (Baseline)":
        "vectorstore/fixed_500_overlap_50",

    "Fixed 500 / 100":
        "vectorstore/fixed_500_overlap_100",

    "Sentence / 500":
        "vectorstore/sentence_500",

    "Paragraph / 500":
        "vectorstore/paragraph_500",

    "Heading-aware / 500":
        "vectorstore/heading_500",
}


EMBEDDING_MODEL = (
    "sentence-transformers/all-MiniLM-L6-v2"
)


# ============================================================
# PAGE
# ============================================================

st.set_page_config(
    page_title="CSE505 Academic RAG",
    page_icon="🎓",
    layout="wide",
)


# ============================================================
# CSS
# ============================================================

st.markdown(
    """
<style>

.main-title {
    text-align: center;
    font-size: 2.4rem;
    font-weight: 700;
}

.subtitle {
    text-align: center;
    color: #666;
    margin-bottom: 30px;
}

.metric-card {
    padding: 15px;
    border-radius: 10px;
    border: 1px solid #ddd;
}

</style>
""",
    unsafe_allow_html=True,
)


st.markdown(
    '<div class="main-title">'
    '🎓 Academic RAG '
    '</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="subtitle">'
    'CSE505 Project 11 — '
    'Document Chunking as Database Physical Design'
    '</div>',
    unsafe_allow_html=True,
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header(
        "⚙️ Experimental Configuration"
    )

    selected_strategy = st.selectbox(
        "Chunking strategy",
        list(INDEXES.keys())
    )

    k = st.slider(
        "Top-K retrieval",
        min_value=1,
        max_value=10,
        value=3
    )

    st.divider()

    st.markdown(
        """
### Research Questions

**RQ1:** Does smaller chunk size
improve evidence retrieval?

**RQ2:** Does chunk overlap improve
answer quality?

**RQ3:** What is the trade-off between
number of chunks and retrieval accuracy?

**RQ4:** Which strategy gives the best
quality/storage/latency balance?
"""
    )


# ============================================================
# LOAD EMBEDDINGS
# ============================================================

@st.cache_resource
def load_embeddings():

    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )


# ============================================================
# LOAD DATABASE
# ============================================================

@st.cache_resource
def load_database(
    path
):

    if not os.path.exists(path):

        return None

    embeddings = load_embeddings()

    return FAISS.load_local(
        path,
        embeddings,
        allow_dangerous_deserialization=True,
    )


# ============================================================
# LOAD LLM
# ============================================================

@st.cache_resource
def load_llm():

    api_key = os.getenv(
        "GROQ_API_KEY"
    )

    if not api_key:

        return None

    return ChatGroq(
        api_key=api_key,
        model="llama-3.3-70b-versatile",
        temperature=0,
        max_tokens=500,
    )


# ============================================================
# DATABASE
# ============================================================

db_path = INDEXES[
    selected_strategy
]

db = load_database(
    db_path
)

if db is None:

    st.error(
        f"FAISS index not found:\n"
        f"{db_path}\n\n"
        "Run:\n"
        "`python -m src.build_indexes`"
    )

    st.stop()


# ============================================================
# DATABASE INFORMATION
# ============================================================

st.subheader(
    "📊 Database Physical Design"
)

col1, col2, col3 = st.columns(3)

with col1:

    st.metric(
        "Indexed vectors",
        db.index.ntotal
    )

with col2:

    st.metric(
        "Chunking strategy",
        selected_strategy
    )

with col3:

    st.metric(
        "Top-K",
        k
    )


# ============================================================
# QUESTION
# ============================================================

question = st.chat_input(
    "Ask an academic question..."
)


if question:

    # --------------------------------------------------------
    # RETRIEVAL
    # --------------------------------------------------------

    start = time.perf_counter()

    docs = db.similarity_search(
        question,
        k=k
    )

    latency = (
        time.perf_counter()
        - start
    ) * 1000

    # --------------------------------------------------------
    # CONTEXT
    # --------------------------------------------------------

    context_parts = []

    for i, doc in enumerate(docs):

        source = doc.metadata.get(
            "source",
            "Unknown"
        )

        page = doc.metadata.get(
            "page_number",
            doc.metadata.get(
                "page",
                "Unknown"
            )
        )

        context_parts.append(
            f"""
[EVIDENCE {i + 1}]
Source: {source}
Page: {page}

{doc.page_content}
"""
        )

    context = "\n".join(
        context_parts
    )

    # --------------------------------------------------------
    # LLM
    # --------------------------------------------------------

    llm = load_llm()

    if llm is None:

        st.error(
            "GROQ_API_KEY is not configured."
        )

        st.stop()

    prompt = f"""
You are an Academic RAG assistant.

Answer ONLY using the retrieved
official academic context.

Do not use outside knowledge.

If the answer cannot be supported
by the context, say:

"Information not available in the retrieved documents."

Question:
{question}

Retrieved Context:
{context}

Give a concise answer and mention
the source/page when possible.
"""

    with st.chat_message(
        "user"
    ):

        st.write(question)

    with st.chat_message(
        "assistant"
    ):

        with st.spinner(
            "Retrieving and generating..."
        ):

            response = llm.invoke(
                prompt
            )

        answer = response.content

        st.write(answer)

        st.divider()

        st.caption(
            f"Retrieval latency: "
            f"{latency:.2f} ms"
        )

        st.subheader(
            "📚 Retrieved Evidence"
        )

        for i, doc in enumerate(docs):

            source = doc.metadata.get(
                "source",
                "Unknown"
            )

            page = doc.metadata.get(
                "page_number",
                doc.metadata.get(
                    "page",
                    "Unknown"
                )
            )

            with st.expander(
                f"Evidence {i + 1} — "
                f"{source} — Page {page}"
            ):

                st.write(
                    doc.page_content
                )
