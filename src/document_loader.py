from pathlib import Path
from typing import List

import re

from langchain_core.documents import Document
import fitz  # PyMuPDF


class AcademicDocumentLoader:

    def __init__(self, data_dir):

        self.data_dir = Path(data_dir)

    # =========================================================
    # TEXT CLEANING
    # =========================================================

    def clean_text(self, text: str) -> str:

        if not text:
            return ""

        # Normalize non-breaking spaces
        text = text.replace("\xa0", " ")

        # Remove excessive spaces
        text = re.sub(r"[ \t]+", " ", text)

        # Normalize excessive blank lines
        text = re.sub(r"\n{3,}", "\n\n", text)

        # Remove spaces at beginning/end of lines
        text = re.sub(
            r"[ \t]+\n",
            "\n",
            text
        )

        text = re.sub(
            r"\n[ \t]+",
            "\n",
            text
        )

        return text.strip()

    # =========================================================
    # LOAD PDFS
    # =========================================================

    def load(self) -> List[Document]:

        documents = []

        pdf_files = sorted(
            self.data_dir.rglob("*.pdf")
        )

        if not pdf_files:

            raise FileNotFoundError(
                f"No PDF files found inside {self.data_dir}"
            )

        print(
            f"\nFound {len(pdf_files)} PDF files."
        )

        for pdf_path in pdf_files:

            print(
                f"Loading: {pdf_path}"
            )

            try:

                pdf = fitz.open(
                    pdf_path
                )

                for page_index in range(
                    len(pdf)
                ):

                    page = pdf[
                        page_index
                    ]

                    # -------------------------------------------------
                    # Extract text
                    # -------------------------------------------------

                    text = page.get_text(
                        "text",
                        sort=True
                    )

                    text = self.clean_text(
                        text
                    )

                    if not text:
                        continue

                    # -------------------------------------------------
                    # Metadata
                    # -------------------------------------------------

                    metadata = {

                        "source": str(
                            pdf_path
                            .relative_to(
                                self.data_dir
                            )
                        ),

                        "source_file":
                            pdf_path.name,

                        "file_name":
                            pdf_path.name,

                        # 1-based page number
                        "page_number":
                            page_index + 1,

                        # Keep original 0-based page too
                        "page":
                            page_index,

                        "total_pages":
                            len(pdf),

                    }

                    documents.append(
                        Document(
                            page_content=text,
                            metadata=metadata
                        )
                    )

                pdf.close()

            except Exception as e:

                print(
                    f"ERROR loading "
                    f"{pdf_path}: {e}"
                )

        print(
            f"\nTotal pages loaded: "
            f"{len(documents)}"
        )

        return documents