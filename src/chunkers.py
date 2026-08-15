from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
import re
from typing import List, Dict
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    SentenceTransformersTokenTextSplitter,
)


# ============================================================
# FIXED-SIZE CHUNKING
# ============================================================

def fixed_chunks(
    documents: List[Document],
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> List[Document]:

    if chunk_overlap >= chunk_size:
        raise ValueError(
            "chunk_overlap must be smaller than chunk_size"
        )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=[
            "\n\n",
            "\n",
            ". ",
            " ",
            "",
        ],
    )

    return splitter.split_documents(documents)


# ============================================================
# SENTENCE-BASED CHUNKING
# ============================================================

def sentence_chunks(
    documents: List[Document],
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> List[Document]:

    chunks = []

    for document in documents:

        text = document.page_content.strip()

        if not text:
            continue

        sentences = re.split(
            r"(?<=[.!?])\s+",
            text,
        )

        current_sentences = []
        current_length = 0

        for sentence in sentences:

            sentence = sentence.strip()

            if not sentence:
                continue

            sentence_length = len(sentence)

            # If adding this sentence exceeds the target
            if (
                current_length + sentence_length
                > chunk_size
                and current_sentences
            ):

                chunk_text = " ".join(
                    current_sentences
                )

                chunks.append(
                    Document(
                        page_content=chunk_text,
                        metadata={
                            **document.metadata,
                            "chunk_strategy": "sentence",
                        },
                    )
                )

                # Keep overlapping sentences
                overlap_sentences = []

                overlap_length = 0

                for previous in reversed(
                    current_sentences
                ):

                    if (
                        overlap_length
                        + len(previous)
                        <= chunk_overlap
                    ):

                        overlap_sentences.insert(
                            0,
                            previous,
                        )

                        overlap_length += len(previous)

                    else:
                        break

                current_sentences = (
                    overlap_sentences
                )

                current_length = sum(
                    len(x)
                    for x in current_sentences
                )

            current_sentences.append(
                sentence
            )

            current_length += sentence_length

        # Remaining sentences
        if current_sentences:

            chunk_text = " ".join(
                current_sentences
            )

            chunks.append(
                Document(
                    page_content=chunk_text,
                    metadata={
                        **document.metadata,
                        "chunk_strategy": "sentence",
                    },
                )
            )

    return chunks


# ============================================================
# PARAGRAPH-BASED CHUNKING
# ============================================================

def paragraph_chunks(
    documents: List[Document],
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> List[Document]:

    chunks = []

    for document in documents:

        paragraphs = re.split(
            r"\n\s*\n",
            document.page_content,
        )

        current = []
        current_length = 0

        for paragraph in paragraphs:

            paragraph = paragraph.strip()

            if not paragraph:
                continue

            paragraph_length = len(paragraph)

            if (
                current_length
                + paragraph_length
                > chunk_size
                and current
            ):

                text = "\n\n".join(current)

                chunks.append(
                    Document(
                        page_content=text,
                        metadata={
                            **document.metadata,
                            "chunk_strategy": "paragraph",
                        },
                    )
                )

                # Basic overlap using previous paragraph
                if chunk_overlap > 0:
                    current = [current[-1]]
                    current_length = len(
                        current[0]
                    )
                else:
                    current = []
                    current_length = 0

            current.append(paragraph)
            current_length += paragraph_length

        if current:

            text = "\n\n".join(current)

            chunks.append(
                Document(
                    page_content=text,
                    metadata={
                        **document.metadata,
                        "chunk_strategy": "paragraph",
                    },
                )
            )

    return chunks


# ============================================================
# HEADING-AWARE CHUNKING
# ============================================================

def heading_chunks(
    documents: List[Document],
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> List[Document]:

    chunks = []

    heading_pattern = re.compile(
        r"^(?:"
        r"\d+(?:\.\d+)*\s+.+"
        r"|"
        r"[A-Z][A-Z\s\-]{4,}"
        r"|"
        r"(?:Chapter|CHAPTER|Section|SECTION)\s+.+"
        r")$"
    )

    for document in documents:

        lines = document.page_content.splitlines()

        sections = []

        current_heading = None
        current_content = []

        for line in lines:

            line = line.strip()

            if not line:
                continue

            is_heading = bool(
                heading_pattern.match(line)
            )

            if is_heading:

                if current_content:

                    sections.append(
                        (
                            current_heading,
                            "\n".join(
                                current_content
                            ),
                        )
                    )

                current_heading = line
                current_content = []

            else:

                current_content.append(line)

        if current_content:

            sections.append(
                (
                    current_heading,
                    "\n".join(
                        current_content
                    ),
                )
            )

        # Split large sections while preserving heading
        for heading, content in sections:

            prefix = (
                f"{heading}\n\n"
                if heading
                else ""
            )

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )

            section_docs = splitter.create_documents(
                [content]
            )

            for section_doc in section_docs:

                chunks.append(
                    Document(
                        page_content=(
                            prefix
                            + section_doc.page_content
                        ),
                        metadata={
                            **document.metadata,
                            "chunk_strategy": "heading",
                            "heading": heading,
                        },
                    )
                )

    return chunks


# ============================================================
# SLIDING WINDOW
# ============================================================

def sliding_window_chunks(
    documents: List[Document],
    chunk_size: int = 500,
    chunk_overlap: int = 100,
) -> List[Document]:

    return fixed_chunks(
        documents,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )


# ============================================================
# MAIN CHUNKING DISPATCHER
# ============================================================

def create_chunks(
    documents,
    strategy,
    chunk_size=500,
    chunk_overlap=50,
):
    """
    Create chunks using one of the Project 11 strategies.

    strategy can be either:

        "fixed"

    OR an experiment configuration dictionary:

        {
            "strategy": "fixed",
            "chunk_size": 500,
            "chunk_overlap": 50
        }
    """

    # ========================================================
    # HANDLE EXPERIMENT DICTIONARY
    # ========================================================

    if isinstance(strategy, dict):

        experiment = strategy

        strategy_name = experiment.get(
            "strategy",
            "fixed"
        )

        chunk_size = experiment.get(
            "chunk_size",
            chunk_size
        )

        chunk_overlap = experiment.get(
            "chunk_overlap",
            chunk_overlap
        )

    # ========================================================
    # HANDLE STRING
    # ========================================================

    elif isinstance(strategy, str):

        strategy_name = strategy

    else:

        raise TypeError(
            "strategy must be either a string "
            "or an experiment dictionary"
        )

    # ========================================================
    # NORMALIZE
    # ========================================================

    strategy_name = strategy_name.lower().strip()

    # ========================================================
    # CREATE CHUNKS
    # ========================================================

    if strategy_name == "fixed":

        chunks = fixed_chunks(
            documents,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    elif strategy_name == "sentence":

        chunks = sentence_chunks(
            documents,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    elif strategy_name == "paragraph":

        chunks = paragraph_chunks(
            documents,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    elif strategy_name == "heading":

        chunks = heading_chunks(
            documents,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    elif strategy_name in (
        "sliding",
        "sliding_window",
    ):

        chunks = sliding_window_chunks(
            documents,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    else:

        raise ValueError(
            f"Unknown chunking strategy: "
            f"{strategy_name}"
        )

    # ========================================================
    # ADD EXPERIMENT METADATA
    # ========================================================

    for index, chunk in enumerate(chunks):

        chunk.metadata["chunk_id"] = index

        chunk.metadata["strategy"] = strategy_name

        chunk.metadata["chunk_size"] = chunk_size

        chunk.metadata["chunk_overlap"] = (
            chunk_overlap
        )

    return chunks

    strategy = strategy.lower().strip()

    if strategy == "fixed":

        chunks = fixed_chunks(
            documents,
            chunk_size,
            chunk_overlap,
        )

    elif strategy == "sentence":

        chunks = sentence_chunks(
            documents,
            chunk_size,
            chunk_overlap,
        )

    elif strategy == "paragraph":

        chunks = paragraph_chunks(
            documents,
            chunk_size,
            chunk_overlap,
        )

    elif strategy == "heading":

        chunks = heading_chunks(
            documents,
            chunk_size,
            chunk_overlap,
        )

    elif strategy in {
        "sliding",
        "sliding_window",
    }:

        chunks = sliding_window_chunks(
            documents,
            chunk_size,
            chunk_overlap,
        )

    else:

        raise ValueError(
            f"Unknown chunking strategy: {strategy}\n"
            f"Supported strategies: "
            f"fixed, sentence, paragraph, heading, "
            f"sliding_window"
        )

    # Add common metadata
    for index, chunk in enumerate(chunks):

        chunk.metadata["chunk_id"] = index

        chunk.metadata["chunk_size"] = chunk_size

        chunk.metadata["chunk_overlap"] = (
            chunk_overlap
        )

        chunk.metadata["strategy"] = strategy

    return chunks
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