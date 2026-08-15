import re
from pathlib import Path
from typing import List

from langchain_community.document_loaders import (
    PyPDFLoader
)

from langchain_core.documents import Document


class AcademicDocumentLoader:

    def __init__(self, data_dir):

        self.data_dir = Path(data_dir)

    @staticmethod
    def clean_text(text: str) -> str:

        if not text:
            return ""

        text = text.replace(
            "\xa0",
            " "
        )

        # Normalize spaces
        text = re.sub(
            r"[ \t]+",
            " ",
            text
        )

        # Normalize line breaks
        text = re.sub(
            r" *\n *",
            "\n",
            text
        )

        # Remove excessive blank lines
        text = re.sub(
            r"\n{3,}",
            "\n\n",
            text
        )

        return text.strip()

    def load(self) -> List[Document]:

        documents = []

        pdf_files = sorted(
            self.data_dir.rglob("*.pdf")
        )

        if not pdf_files:

            raise FileNotFoundError(
                f"No PDF files found inside "
                f"{self.data_dir}"
            )

        print(
            f"\nFound {len(pdf_files)} PDF files."
        )

        for pdf_path in pdf_files:

            print(
                f"Loading: {pdf_path}"
            )

            try:

                loader = PyPDFLoader(
                    str(pdf_path)
                )

                pages = loader.load()

                for page in pages:

                    # ----------------------------------------
                    # CLEAN TEXT
                    # ----------------------------------------

                    page.page_content = (
                        self.clean_text(
                            page.page_content
                        )
                    )

                    # ----------------------------------------
                    # METADATA
                    # ----------------------------------------

                    page.metadata[
                        "source"
                    ] = str(
                        pdf_path.relative_to(
                            self.data_dir
                        )
                    )

                    page.metadata[
                        "file_name"
                    ] = pdf_path.name

                    original_page = (
                        page.metadata.get(
                            "page",
                            0
                        )
                    )

                    page.metadata[
                        "page_number"
                    ] = int(
                        original_page
                    ) + 1

                    documents.append(
                        page
                    )

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
