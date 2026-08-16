import os
import time
from pathlib import Path

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq


# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv()


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent


# ============================================================
# PROJECT 11 EXPERIMENTS
# ============================================================

INDEXES = {
    "Fixed 250": "vectorstore/fixed_250",
    "Fixed 500": "vectorstore/fixed_500",
    "Fixed 750": "vectorstore/fixed_750",
    "Fixed 1000": "vectorstore/fixed_1000",

    "Fixed 500 / Overlap 0": "vectorstore/fixed_500_overlap_0",
    "Fixed 500 / Overlap 25":"vectorstore/fixed_500_overlap_25",
    "Fixed 500 / Overlap 50": "vectorstore/fixed_500_overlap_50",
    "Fixed 500 / Overlap 100":"vectorstore/fixed_500_overlap_100",

    "Sentence":"vectorstore/sentence_500",
    "Paragraph":"vectorstore/paragraph_500",
    "Heading-aware":"vectorstore/heading_500",
}


# ============================================================
# MODELS
# ============================================================

EMBEDDING_MODEL = (
    "sentence-transformers/all-MiniLM-L6-v2"
)

LLM_MODEL = "llama-3.3-70b-versatile"


# ============================================================
# PAGE CONFIGURATION
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
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        color: #666;
        margin-bottom: 30px;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">'
    '🎓 Academic RAG'
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
        list(INDEXES.keys()),
        index=1,
    )

    k = st.slider(
        "Top-K retrieval",
        min_value=1,
        max_value=10,
        value=3,
    )

    st.divider()

    st.markdown(
        """
        ### Research Questions

        **RQ1:** Does smaller chunk size
        improve evidence retrieval?

        **RQ2:** How does chunk overlap affect 
        retrieval performance and indexing cost?

        **RQ3:** What is the trade-off between
        number of chunks and retrieval accuracy?

        **RQ4:** Which strategy gives the best
        quality/storage/latency balance?
        """
    )

# ============================================================
# RESEARCH EVALUATION
# ============================================================

st.divider()

st.header("📊 Research Evaluation")

RESULTS_FILE = Path(
    "results/retrieval_results.csv"
)

if not RESULTS_FILE.exists():

    st.warning(
        "retrieval_results.csv not found. "
        "Run the retrieval benchmark first:"
    )

    st.code(
        "python -m src.benchmark"
    )

else:

    results_df = pd.read_csv(
        RESULTS_FILE
    )

    st.subheader(
        "Retrieval Performance"
    )

    # --------------------------------------------------------
    # RECALL @ K
    # --------------------------------------------------------

    recall_df = (
        results_df
        .groupby("k")["evidence_found"]
        .mean()
        .reset_index()
    )

    recall_df["recall"] = (
        recall_df["evidence_found"]
    )

    recall_values = {}

    for _, row in recall_df.iterrows():

        recall_values[
            int(row["k"])
        ] = row["recall"]

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Recall@1",
            f"{recall_values.get(1, 0) * 100:.2f}%"
        )

    with col2:

        st.metric(
            "Recall@3",
            f"{recall_values.get(3, 0) * 100:.2f}%"
        )

    with col3:

        st.metric(
            "Recall@5",
            f"{recall_values.get(5, 0) * 100:.2f}%"
        )

    # --------------------------------------------------------
    # AVERAGE RETRIEVAL LATENCY
    # --------------------------------------------------------

    avg_latency = (
        results_df[
            "retrieval_latency_ms"
        ].mean()
    )

    st.metric(
        "Average Retrieval Latency",
        f"{avg_latency:.2f} ms"
    )

    # --------------------------------------------------------
    # PER-STRATEGY RECALL
    # --------------------------------------------------------

    strategy_recall = (
        results_df
        .groupby(
            ["index_name", "k"]
        )["evidence_found"]
        .mean()
        .reset_index()
    )

    recall_pivot = (
        strategy_recall
        .pivot(
            index="index_name",
            columns="k",
            values="evidence_found"
        )
        .reset_index()
    )

    recall_pivot.columns.name = None

    rename_map = {
        1: "Recall@1",
        3: "Recall@3",
        5: "Recall@5",
    }

    recall_pivot = recall_pivot.rename(
        columns=rename_map
    )

    st.subheader(
        "Recall by Chunking Strategy"
    )

    st.dataframe(
        recall_pivot,
        use_container_width=True
    )

    # --------------------------------------------------------
    # AVERAGE LATENCY BY STRATEGY
    # --------------------------------------------------------

    latency_df = (
        results_df
        .groupby("index_name")
        ["retrieval_latency_ms"]
        .mean()
        .reset_index()
    )

    latency_df = latency_df.rename(
        columns={
            "retrieval_latency_ms":
                "Average Latency (ms)"
        }
    )

    st.subheader(
        "Average Retrieval Latency"
    )

    st.dataframe(
        latency_df,
        use_container_width=True
    )

    # --------------------------------------------------------
    # NUMBER OF CHUNKS
    # --------------------------------------------------------

    chunk_records = []

    for index_name in results_df[
        "index_name"
    ].unique():

        metadata_file = Path(
            "vectorstore"
        ) / index_name / "metadata.json"

        if metadata_file.exists():

            import json

            with open(
                metadata_file,
                "r",
                encoding="utf-8"
            ) as f:

                metadata = json.load(f)

            chunk_records.append({

                "index_name":
                    index_name,

                "number_of_chunks":
                    metadata.get(
                        "num_vectors",
                        0
                    ),

                "storage_mb":
                    metadata.get(
                        "total_storage_mb",
                        0
                    ),

                "ingestion_time_sec":
                    metadata.get(
                        "ingestion_time_sec",
                        0
                    ),
            })

    physical_df = pd.DataFrame(
        chunk_records
    )

    if not physical_df.empty:

        st.subheader(
            "Database Physical Design"
        )

        st.dataframe(
            physical_df,
            use_container_width=True
        )

        # ----------------------------------------------------
        # MERGE RECALL + CHUNK COUNT
        # ----------------------------------------------------

        recall_3 = (
            results_df[
                results_df["k"] == 3
            ]
            .groupby("index_name")
            ["evidence_found"]
            .mean()
            .reset_index()
        )

        recall_3 = recall_3.rename(
            columns={
                "evidence_found":
                    "Recall@3"
            }
        )

        analysis_df = physical_df.merge(
            recall_3,
            on="index_name",
            how="left"
        )

        # ----------------------------------------------------
        # CHUNKS VS RECALL@3
        # ----------------------------------------------------

        st.subheader(
            "📈 Number of Chunks vs Recall@3"
        )

        fig, ax = plt.subplots()

        ax.scatter(
            analysis_df[
                "number_of_chunks"
            ],
            analysis_df[
                "Recall@3"
            ]
        )

        for _, row in analysis_df.iterrows():

            ax.annotate(
                row["index_name"],
                (
                    row["number_of_chunks"],
                    row["Recall@3"]
                ),
                fontsize=8
            )

        ax.set_xlabel(
            "Number of Indexed Chunks"
        )

        ax.set_ylabel(
            "Recall@3"
        )

        ax.set_title(
            "Chunk Count vs Retrieval Accuracy"
        )

        ax.grid(
            True,
            alpha=0.3
        )

        st.pyplot(fig)

        # ----------------------------------------------------
        # CHUNKS VS RECALL@1
        # ----------------------------------------------------

        recall_1 = (
            results_df[
                results_df["k"] == 1
            ]
            .groupby("index_name")
            ["evidence_found"]
            .mean()
            .reset_index()
        )

        recall_1 = recall_1.rename(
            columns={
                "evidence_found":
                    "Recall@1"
            }
        )

        analysis_df = analysis_df.merge(
            recall_1,
            on="index_name",
            how="left"
        )

        st.subheader(
            "📈 Number of Chunks vs Recall@1"
        )

        fig, ax = plt.subplots()

        ax.scatter(
            analysis_df[
                "number_of_chunks"
            ],
            analysis_df[
                "Recall@1"
            ]
        )

        for _, row in analysis_df.iterrows():

            ax.annotate(
                row["index_name"],
                (
                    row["number_of_chunks"],
                    row["Recall@1"]
                ),
                fontsize=8
            )

        ax.set_xlabel(
            "Number of Indexed Chunks"
        )

        ax.set_ylabel(
            "Recall@1"
        )

        ax.set_title(
            "Chunk Count vs Recall@1"
        )

        ax.grid(
            True,
            alpha=0.3
        )

        st.pyplot(fig)
        
# ============================================================
# EMBEDDING MODEL
# ============================================================

@st.cache_resource
def load_embeddings():

    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )


# ============================================================
# FAISS DATABASE
# ============================================================

@st.cache_resource
def load_database(path):

    if not os.path.exists(path):

        return None

    embeddings = load_embeddings()

    return FAISS.load_local(
        path,
        embeddings,
        allow_dangerous_deserialization=True,
    )


# ============================================================
# GROQ LLM
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
        model=LLM_MODEL,
        temperature=0,
        max_tokens=250,
    )


# ============================================================
# SELECT DATABASE
# ============================================================

db_path = INDEXES[
    selected_strategy
]


# ============================================================
# CHECK INDEX
# ============================================================

db = load_database(
    db_path
)


if db is None:

    st.error(
        f"""
        FAISS index not found:

        `{db_path}`

        Run:

        `python -m src.build_indexes`
        """
    )

    st.stop()


# ============================================================
# DATABASE PHYSICAL DESIGN INFORMATION
# ============================================================

st.subheader(
    "📊 Database Physical Design"
)


col1, col2, col3 = st.columns(3)


with col1:

    st.metric(
        "Indexed vectors",
        db.index.ntotal,
    )


with col2:

    st.metric(
        "Chunking strategy",
        selected_strategy,
    )


with col3:

    st.metric(
        "Top-K",
        k,
    )


# ============================================================
# LOAD METADATA
# ============================================================

metadata_file = (
    Path(db_path) /
    "metadata.json"
)


if metadata_file.exists():

    import json

    with open(
        metadata_file,
        "r",
        encoding="utf-8",
    ) as f:

        index_metadata = json.load(f)

    col4, col5, col6 = st.columns(3)

    with col4:

        st.metric(
            "Storage",
            f"{index_metadata['total_storage_mb']:.4f} MB",
        )

    with col5:

        st.metric(
            "Ingestion",
            f"{index_metadata['ingestion_time_sec']:.2f} sec",
        )

    with col6:

        st.metric(
            "Vectors",
            index_metadata["num_vectors"],
        )


# ============================================================
# CHAT HISTORY
# ============================================================

if "messages" not in st.session_state:

    st.session_state.messages = []


for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        st.write(
            message["content"]
        )


# ============================================================
# QUESTION
# ============================================================

question = st.chat_input(
    "Ask an academic question..."
)


if question:

    # ========================================================
    # DISPLAY USER QUESTION
    # ========================================================

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question,
        }
    )

    with st.chat_message("user"):

        st.write(question)


    # ========================================================
    # RETRIEVAL
    # ========================================================

    retrieval_start = time.perf_counter()

    docs = db.similarity_search(
        question,
        k=k,
    )

    retrieval_latency = (
        time.perf_counter()
        - retrieval_start
    ) * 1000


    # ========================================================
    # BUILD CONTEXT
    # ========================================================

    context_parts = []


    for i, doc in enumerate(docs):

        source = doc.metadata.get(
            "source",
            doc.metadata.get(
                "source_file",
                "Unknown",
            ),
        )

        page = doc.metadata.get(
            "page_number",
            doc.metadata.get(
                "page",
                "Unknown",
            ),
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


    # ========================================================
    # LOAD LLM
    # ========================================================

    llm = load_llm()


    if llm is None:

        st.error(
            """
            GROQ_API_KEY is not configured.

            Make sure your `.env` contains:

            GROQ_API_KEY=your_actual_key
            """
        )

        st.stop()


    # ========================================================
    # PROMPT
    # ========================================================

    prompt = f"""
You are an Academic RAG assistant for
East West University.

Answer ONLY using the retrieved
official academic context.

Do NOT use outside knowledge.

If the answer cannot be supported
by the retrieved context, say:

"Information not available in the
retrieved documents."

Question:

{question}

Retrieved Context:

{context}

Give a concise and accurate answer.

Mention the source and page when
possible.
"""


    # ========================================================
    # GENERATE ANSWER
    # ========================================================

    generation_start = time.perf_counter()


    with st.chat_message(
        "assistant"
    ):

        with st.spinner(
            "Retrieving and generating..."
        ):

            response = llm.invoke(
                prompt
            )


        generation_latency = (
            time.perf_counter()
            - generation_start
        ) * 1000


        answer = response.content


        st.write(
            answer
        )


        # ====================================================
        # EXPERIMENT METRICS
        # ====================================================

        st.divider()

        st.subheader(
            "📈 Experiment Metrics"
        )


        metric1, metric2, metric3 = st.columns(3)


        with metric1:

            st.metric(
                "Retrieval latency",
                f"{retrieval_latency:.2f} ms",
            )


        with metric2:

            st.metric(
                "Retrieved chunks",
                len(docs),
            )


        with metric3:

            st.metric(
                "Generation latency",
                f"{generation_latency:.2f} ms",
            )


        # ====================================================
        # RETRIEVED EVIDENCE
        # ====================================================

        st.subheader(
            "📚 Retrieved Evidence"
        )


        for i, doc in enumerate(docs):

            source = doc.metadata.get(
                "source",
                doc.metadata.get(
                    "source_file",
                    "Unknown",
                ),
            )

            page = doc.metadata.get(
                "page_number",
                doc.metadata.get(
                    "page",
                    "Unknown",
                ),
            )


            with st.expander(
                f"Evidence {i + 1} — "
                f"{source} — Page {page}"
            ):

                st.write(
                    doc.page_content
                )


    # ========================================================
    # SAVE ASSISTANT RESPONSE
    # ========================================================

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
        }
    )
