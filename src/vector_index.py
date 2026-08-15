import json
import os
import time
from pathlib import Path

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from src.config import (
    EMBEDDING_MODEL,
    VECTORSTORE_DIR,
)


class FAISSIndexManager:

    def __init__(self):

        print(
            f"\nLoading embedding model: "
            f"{EMBEDDING_MODEL}"
        )

        self.embeddings = (
            HuggingFaceEmbeddings(
                model_name=EMBEDDING_MODEL
            )
        )

    # ========================================================
    # CREATE INDEX
    # ========================================================

    def build_index(
        self,
        documents,
        index_name,
    ):

        index_dir = (
            VECTORSTORE_DIR / index_name
        )

        index_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        print(
            f"\nBuilding FAISS index: "
            f"{index_name}"
        )

        start_time = time.perf_counter()

        db = FAISS.from_documents(
            documents,
            self.embeddings
        )

        build_time = (
            time.perf_counter()
            - start_time
        )

        db.save_local(
            str(index_dir)
        )

        faiss_file = (
            index_dir / "index.faiss"
        )

        pkl_file = (
            index_dir / "index.pkl"
        )

        faiss_size = (
            faiss_file.stat().st_size
            if faiss_file.exists()
            else 0
        )

        pkl_size = (
            pkl_file.stat().st_size
            if pkl_file.exists()
            else 0
        )

        total_size = (
            faiss_size + pkl_size
        )

        metrics = {

            "index_name":
                index_name,

            "num_vectors":
                int(db.index.ntotal),

            "faiss_size_bytes":
                faiss_size,

            "pkl_size_bytes":
                pkl_size,

            "total_storage_bytes":
                total_size,

            "total_storage_mb":
                round(
                    total_size / (
                        1024 * 1024
                    ),
                    4,
                ),

            "ingestion_time_sec":
                round(
                    build_time,
                    4,
                ),
        }

        with open(
            index_dir / "metadata.json",
            "w",
            encoding="utf-8",
        ) as f:

            json.dump(
                metrics,
                f,
                indent=2
            )

        print(
            f"Vectors: "
            f"{metrics['num_vectors']}"
        )

        print(
            f"Storage: "
            f"{metrics['total_storage_mb']} MB"
        )

        print(
            f"Ingestion: "
            f"{metrics['ingestion_time_sec']} sec"
        )

        return db, metrics

    # ========================================================
    # LOAD INDEX
    # ========================================================

    def load_index(
        self,
        index_name,
    ):

        index_dir = (
            VECTORSTORE_DIR / index_name
        )

        if not index_dir.exists():

            raise FileNotFoundError(
                f"Index not found: "
                f"{index_dir}"
            )

        db = FAISS.load_local(
            str(index_dir),
            self.embeddings,
            allow_dangerous_deserialization=True,
        )

        return db