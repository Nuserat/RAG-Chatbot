import json
import time
import csv
from pathlib import Path

from src.config import (
    EVALUATION_DIR,
    RESULTS_DIR,
    EXPERIMENTS,
    K_VALUES,
)

from src.vector_index import (
    FAISSIndexManager
)


def load_questions():

    with open(
        EVALUATION_DIR /
        "questions.json",
        "r",
        encoding="utf-8",
    ) as f:

        return json.load(f)


def normalize_source(source):

    return (
        Path(str(source))
        .name
        .lower()
        .strip()
    )


def evidence_found(
    docs,
    question,
):

    gold_source = normalize_source(
        question["source"]
    )

    gold_pages = set(
        int(p)
        for p in question["pages"]
    )

    for doc in docs:

        source = normalize_source(
            doc.metadata.get(
                "source",
                ""
            )
        )

        page = doc.metadata.get(
            "page_number",
            doc.metadata.get(
                "page",
                None
            )
        )

        try:

            page = int(page)

        except Exception:

            continue

        if (
            source == gold_source
            and page in gold_pages
        ):

            return True

    return False


def benchmark_index(
    db,
    questions,
    index_name,
):

    rows = []

    for question in questions:

        question_id = question["id"]

        query = question["question"]

        for k in K_VALUES:

            start = time.perf_counter()

            docs = (
                db.similarity_search(
                    query,
                    k=k
                )
            )

            latency = (
                time.perf_counter()
                - start
            )

            found = evidence_found(
                docs,
                question
            )

            rows.append({

                "index_name":
                    index_name,

                "question_id":
                    question_id,

                "k":
                    k,

                "retrieval_latency_ms":
                    round(
                        latency * 1000,
                        4
                    ),

                "evidence_found":
                    int(found),

                "retrieved_chunks":
                    len(docs),
            })

    return rows


def main():

    print("=" * 70)
    print(
        "RETRIEVAL BENCHMARK"
    )
    print("=" * 70)

    questions = load_questions()

    manager = (
        FAISSIndexManager()
    )

    all_results = []

    for index_name in EXPERIMENTS:

        print(
            f"\nBenchmarking: "
            f"{index_name}"
        )

        try:

            db = manager.load_index(
                index_name
            )

            results = benchmark_index(
                db,
                questions,
                index_name
            )

            all_results.extend(
                results
            )

        except Exception as e:

            print(
                f"ERROR: {e}"
            )

    output_file = (
        RESULTS_DIR /
        "retrieval_results.csv"
    )

    with open(
        output_file,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        if all_results:

            writer = csv.DictWriter(
                f,
                fieldnames=list(
                    all_results[0].keys()
                )
            )

            writer.writeheader()

            writer.writerows(
                all_results
            )

    print(
        f"\nSaved: {output_file}"
    )


if __name__ == "__main__":

    main()