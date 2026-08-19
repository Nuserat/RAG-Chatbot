import json
import csv
import time

from pathlib import Path

from langchain_groq import ChatGroq

from src.config import (
    EVALUATION_DIR,
    RESULTS_DIR,
    EXPERIMENTS,
    GROQ_MODEL,
)

from src.vector_index import (
    FAISSIndexManager
)
from dotenv import load_dotenv

load_dotenv()



SYSTEM_PROMPT = """
You are an academic RAG evaluation assistant.

Answer ONLY using the retrieved context.

Do not use outside knowledge.

If the retrieved context does not contain
enough evidence, say:

"Information not available in the retrieved documents."

Your answer must be concise and factual.
"""


def load_questions():

    with open(
        EVALUATION_DIR /
        "questions.json",
        "r",
        encoding="utf-8",
    ) as f:

        return json.load(f)


def build_context(docs):

    parts = []

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

        parts.append(
            f"""
[EVIDENCE {i + 1}]
Source: {source}
Page: {page}

{doc.page_content}
"""
        )

    return "\n".join(parts)


def evaluate_with_llm(
    llm,
    question,
    context,
    reference_answer,
):

    prompt = f"""
{SYSTEM_PROMPT}

QUESTION:
{question}

REFERENCE ANSWER:
{reference_answer}

RETRIEVED CONTEXT:
{context}

Evaluate the quality of the answer that can
be generated from the retrieved evidence.

Return ONLY valid JSON:

{{
    "answer": "...",
    "correctness": 1,
    "faithfulness": 1,
    "completeness": 1,
    "evidence_supported": 1
}}

Scores must be integers from 1 to 5.

Definitions:

correctness:
1 = incorrect
5 = fully correct

faithfulness:
1 = unsupported by evidence
5 = fully supported

completeness:
1 = missing most information
5 = complete

evidence_supported:
1 = evidence insufficient
5 = evidence strongly supports answer
"""

    response = llm.invoke(
        prompt
    )

    return response.content

def parse_llm_json(text):
    """
    Safely parse JSON returned by the LLM.

    Handles:
    1. Normal JSON
    2. ```json ... ``` responses
    3. JSON embedded in surrounding text
    """

    if not text:
        raise ValueError(
            "LLM returned an empty response."
        )

    text = text.strip()

    # --------------------------------------------------
    # Case 1: direct JSON
    # --------------------------------------------------

    try:
        return json.loads(text)

    except json.JSONDecodeError:
        pass

    # --------------------------------------------------
    # Case 2: Markdown code block
    # --------------------------------------------------

    if "```" in text:

        cleaned = text

        cleaned = cleaned.replace(
            "```json",
            ""
        )

        cleaned = cleaned.replace(
            "```JSON",
            ""
        )

        cleaned = cleaned.replace(
            "```",
            ""
        )

        cleaned = cleaned.strip()

        try:
            return json.loads(cleaned)

        except json.JSONDecodeError:
            pass

    # --------------------------------------------------
    # Case 3: JSON embedded in text
    # --------------------------------------------------

    start = text.find("{")
    end = text.rfind("}")

    if start != -1 and end != -1:

        candidate = text[
            start:end + 1
        ]

        try:
            return json.loads(
                candidate
            )

        except json.JSONDecodeError:
            pass

    # --------------------------------------------------
    # Failed
    # --------------------------------------------------

    raise ValueError(
        "Could not parse LLM response as JSON.\n"
        f"LLM response:\n{text}"
    )

def main():

    import os

    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise ValueError(
            "Set GROQ_API_KEY environment variable." )

    llm = ChatGroq(
        api_key=api_key,
        model=GROQ_MODEL,
        temperature=0,
        max_tokens=500,
    )

    manager = (
        FAISSIndexManager()
    )

    questions = load_questions()

    questions = questions[:2]

    results = []

    for index_name in EXPERIMENTS:

        print(
            f"\nEvaluating: "
            f"{index_name}"
        )

        db = manager.load_index(
            index_name
        )

        for question in questions:
            for k in [1, 3, 5]:
                start = time.perf_counter()
                docs = (
                    db.similarity_search(
                    question["question"],
                    k=k
                )
            )

            retrieval_time = (
                time.perf_counter()
                - start
            )

            context = build_context(
                docs
            )

            try:

                evaluation_text = (
                    evaluate_with_llm(
                        llm,
                        question[
                            "question"
                        ],
                        context,
                        question[
                            "reference_answer"
                        ],
                    )
                )


                evaluation = parse_llm_json( evaluation_text)


            except Exception as e:

                print(
                    f"Evaluation error: {e}"
                )

                evaluation = {
                    "answer": "",
                    "correctness": None,
                    "faithfulness": None,
                    "completeness": None,
                    "evidence_supported": None,
                }

            results.append({

                "index_name":
                    index_name,

                "question_id":
                    question["id"],

                "k":
                    k,

                "retrieval_latency_ms":
                    retrieval_time * 1000,

                "answer":
                    evaluation.get(
                        "answer",
                        ""
                    ),

                "correctness":
                    evaluation.get(
                        "correctness",
                        0
                    ),

                "faithfulness":
                    evaluation.get(
                        "faithfulness",
                        0
                    ),

                "completeness":
                    evaluation.get(
                        "completeness",
                        0
                    ),

                "evidence_supported":
                    evaluation.get(
                        "evidence_supported",
                        0
                    ),
            })

    output = (
        RESULTS_DIR /
        "answer_results.csv"
    )

    with open(
        output,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=list(
                results[0].keys()
            )
        )

        writer.writeheader()

        writer.writerows(
            results
        )

    print(
        f"\nSaved: {output}"
    )
    


if __name__ == "__main__":

    main()
