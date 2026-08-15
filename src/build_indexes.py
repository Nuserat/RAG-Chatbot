import csv
import time
from pathlib import Path
from src.config import EXPERIMENTS
from src.document_loader import AcademicDocumentLoader

from langchain_huggingface import (
    HuggingFaceEmbeddings
)

from src.config import (
    DATA_DIR,
    RESULTS_DIR,
    EXPERIMENTS,
    EMBEDDING_MODEL,
)

from src.document_loader import (
    AcademicDocumentLoader
)

from src.chunkers import (
    create_chunks
)

from src.vector_index import (
    FAISSIndexManager
)


def calculate_chunk_statistics(chunks):

    if not chunks:

        return {
            "avg_chunk_chars": 0,
            "min_chunk_chars": 0,
            "max_chunk_chars": 0,
            "total_chunk_chars": 0,
            "redundancy_ratio": 0,
        }

    lengths = [
        len(doc.page_content)
        for doc in chunks
    ]

    total_chunk_chars = sum(
        lengths
    )

    return {

        "avg_chunk_chars":
            round(
                sum(lengths)
                / len(lengths),
                2
            ),

        "min_chunk_chars":
            min(lengths),

        "max_chunk_chars":
            max(lengths),

        "total_chunk_chars":
            total_chunk_chars,
    }


def main():

    print("=" * 70)
    print(
        "CSE505 PROJECT 11"
    )
    print(
        "Document Chunking as Database "
        "Physical-Design Decision"
    )
    print("=" * 70)

    # --------------------------------------------------------
    # LOAD DOCUMENTS
    # --------------------------------------------------------

    loader = AcademicDocumentLoader(
        DATA_DIR
    )

    documents = loader.load()

    # --------------------------------------------------------
    # EMBEDDING MODEL
    # --------------------------------------------------------

    embedding_model = (
        HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL
        )
    )

    # --------------------------------------------------------
    # INDEX MANAGER
    # --------------------------------------------------------

    index_manager = (
        FAISSIndexManager()
    )

    results = []

    # --------------------------------------------------------
    # BUILD EVERY EXPERIMENT
    # --------------------------------------------------------

    for index_name, config in EXPERIMENTS.items():

        print("\n")
        print("=" * 70)
        print(
            f"EXPERIMENT: {index_name}"
        )
        print("=" * 70)

        start_time = time.perf_counter()

        try:

            chunk_result = create_chunks(
                documents,
                config,
                embedding_model
            )

            # Parent-child returns:
            # parents, children

            if config["strategy"] == "parent_child":

                parents, chunks = (
                    chunk_result
                )

                print(
                    f"Parents: "
                    f"{len(parents)}"
                )

                print(
                    f"Children: "
                    f"{len(chunks)}"
                )

            else:

                chunks = chunk_result

                parents = []

            chunk_stats = (
                calculate_chunk_statistics(
                    chunks
                )
            )

            print(
                f"Chunks created: "
                f"{len(chunks)}"
            )

            # ------------------------------------------------
            # BUILD FAISS
            # ------------------------------------------------

            db, index_metrics = (
                index_manager.build_index(
                    chunks,
                    index_name
                )
            )

            total_time = (
                time.perf_counter()
                - start_time
            )

            row = {

                "index_name":
                    index_name,

                "strategy":
                    config["strategy"],

                "chunk_count":
                    len(chunks),

                "parent_count":
                    len(parents),

                **chunk_stats,

                **index_metrics,

                "total_pipeline_time_sec":
                    round(
                        total_time,
                        4
                    ),
            }

            results.append(row)

            print(
                "\n✓ Experiment completed."
            )

        except Exception as e:

            print(
                f"\n✗ Experiment failed: "
                f"{e}"
            )

    # --------------------------------------------------------
    # SAVE RESULTS
    # --------------------------------------------------------

    output_file = (
        RESULTS_DIR /
        "experiment_results.csv"
    )

    if results:

        fieldnames = list(
            results[0].keys()
        )

        with open(
            output_file,
            "w",
            newline="",
            encoding="utf-8",
        ) as f:

            writer = csv.DictWriter(
                f,
                fieldnames=fieldnames
            )

            writer.writeheader()

            writer.writerows(
                results
            )

    print("\n")
    print("=" * 70)
    print(
        "ALL INDEXES BUILT"
    )
    print("=" * 70)

    print(
        f"Results saved to: "
        f"{output_file}"
    )


if __name__ == "__main__":

    main()