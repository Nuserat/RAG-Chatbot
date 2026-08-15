import pandas as pd

from src.config import RESULTS_DIR


def main():

    file = (
        RESULTS_DIR /
        "retrieval_results.csv"
    )

    df = pd.read_csv(file)

    # --------------------------------------------------------
    # Retrieval Recall
    # --------------------------------------------------------

    recall = (
        df.groupby(
            ["index_name", "k"]
        )["evidence_found"]
        .mean()
        .reset_index()
    )

    recall[
        "evidence_recall_percent"
    ] = (
        recall["evidence_found"]
        * 100
    )

    # --------------------------------------------------------
    # Latency
    # --------------------------------------------------------

    latency = (
        df.groupby(
            ["index_name", "k"]
        )[
            "retrieval_latency_ms"
        ]
        .agg(
            [
                "mean",
                "median",
                lambda x:
                    x.quantile(0.95),
            ]
        )
        .reset_index()
    )

    latency.columns = [
        "index_name",
        "k",
        "mean_latency_ms",
        "median_latency_ms",
        "p95_latency_ms",
    ]

    # --------------------------------------------------------
    # Merge
    # --------------------------------------------------------

    final = recall.merge(
        latency,
        on=[
            "index_name",
            "k"
        ]
    )

    output = (
        RESULTS_DIR /
        "retrieval_summary.csv"
    )

    final.to_csv(
        output,
        index=False
    )

    print("\nRetrieval Summary\n")

    print(
        final.to_string(
            index=False
        )
    )

    print(
        f"\nSaved: {output}"
    )


if __name__ == "__main__":

    main()