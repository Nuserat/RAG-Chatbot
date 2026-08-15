from pathlib import Path
from typing import List

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document

from pathlib import Path


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Your PDFs are inside:
# RAG-Chatbot/Doc/
DATA_DIR = PROJECT_ROOT / "Doc"

# FAISS indexes
VECTORSTORE_DIR = PROJECT_ROOT / "vectorstore"

# Backward-compatible alias
INDEX_DIR = VECTORSTORE_DIR

# Experimental results
RESULTS_DIR = PROJECT_ROOT / "results"

# Logs
LOG_DIR = PROJECT_ROOT / "logs"

EVALUATION_DIR = PROJECT_ROOT / "evaluation"


for directory in [
    INDEX_DIR,
    RESULTS_DIR,
    EVALUATION_DIR,
    LOG_DIR,
]:
    directory.mkdir(
        parents=True,
        exist_ok=True
    )



# ============================================================
# RETRIEVAL
# ============================================================

TOP_K = 3



# ============================================================
# BASELINE
# ============================================================

# Your existing chatbot configuration
BASELINE_CHUNK_SIZE = 500

BASELINE_CHUNK_OVERLAP = 50


# ============================================================
# EVALUATION DIRECTORY
# ============================================================

EVALUATION_DIR = PROJECT_ROOT / "evaluation"

EVALUATION_DIR.mkdir(
    parents=True,
    exist_ok=True
)
# ============================================================
# PROJECT 11 EXPERIMENTS
# ============================================================

EXPERIMENTS = {

    # --------------------------------------------------------
    # RQ1
    # Effect of chunk size
    # --------------------------------------------------------

    "fixed_250": {
        "strategy": "fixed",
        "chunk_size": 250,
        "chunk_overlap": 25,
    },

    "fixed_500": {
        "strategy": "fixed",
        "chunk_size": 500,
        "chunk_overlap": 50,
    },

    "fixed_750": {
        "strategy": "fixed",
        "chunk_size": 750,
        "chunk_overlap": 75,
    },

    "fixed_1000": {
        "strategy": "fixed",
        "chunk_size": 1000,
        "chunk_overlap": 100,
    },


    # --------------------------------------------------------
    # RQ2
    # Effect of overlap
    # --------------------------------------------------------

    "fixed_500_overlap_0": {
        "strategy": "fixed",
        "chunk_size": 500,
        "chunk_overlap": 0,
    },

    "fixed_500_overlap_25": {
        "strategy": "fixed",
        "chunk_size": 500,
        "chunk_overlap": 25,
    },

    "fixed_500_overlap_50": {
        "strategy": "fixed",
        "chunk_size": 500,
        "chunk_overlap": 50,
    },

    "fixed_500_overlap_100": {
        "strategy": "fixed",
        "chunk_size": 500,
        "chunk_overlap": 100,
    },


    # --------------------------------------------------------
    # RQ4
    # Chunking strategy comparison
    # --------------------------------------------------------

    "sentence_500": {
        "strategy": "sentence",
        "chunk_size": 500,
        "chunk_overlap": 50,
    },

    "paragraph_500": {
        "strategy": "paragraph",
        "chunk_size": 500,
        "chunk_overlap": 50,
    },

    "heading_500": {
        "strategy": "heading",
        "chunk_size": 500,
        "chunk_overlap": 50,
    },
}

# ============================================================
# RETRIEVAL EVALUATION
# ============================================================

# Values of K used for Recall@K, Precision@K, etc.
K_VALUES = [1, 3, 5]

# ============================================================
# EMBEDDING MODEL
# ============================================================

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# ============================================================
# LLM
# ============================================================

LLM_MODEL = "llama-3.3-70b-versatile"
# Backward-compatible alias
GROQ_MODEL = LLM_MODEL
LLM_TEMPERATURE = 0.3

LLM_MAX_TOKENS = 800

# ============================================================
# DIRECTORY CREATION
# ============================================================

for directory in [
    DATA_DIR,
    INDEX_DIR,
    RESULTS_DIR,
    LOG_DIR,
]:
    directory.mkdir(parents=True, exist_ok=True)

class AcademicDocumentLoader:

    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)

    def load(self) -> List[Document]:

        documents = []

        pdf_files = sorted(
            self.data_dir.rglob("*.pdf")
        )

        if not pdf_files:
            raise FileNotFoundError(
                f"No PDF files found inside {self.data_dir}"
            )

        print(f"\nFound {len(pdf_files)} PDF files.")

        for pdf_path in pdf_files:

            print(f"Loading: {pdf_path}")

            try:

                loader = PyPDFLoader(
                    str(pdf_path)
                )

                pages = loader.load()

                for page in pages:

                    page.metadata["source"] = str(
                        pdf_path.relative_to(self.data_dir)
                    )

                    page.metadata["file_name"] = (
                        pdf_path.name
                    )

                    page_number = page.metadata.get(
                        "page", 0
                    )

                    page.metadata["page_number"] = (
                        int(page_number) + 1
                    )

                    documents.append(page)

            except Exception as e:

                print(
                    f"ERROR loading {pdf_path}: {e}"
                )

        print(
            f"\nTotal pages loaded: {len(documents)}"
        )

        return documents