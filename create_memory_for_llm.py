import os

from typing import List

from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import (
    RunnablePassthrough
)
from langchain_core.output_parsers import (
    StrOutputParser
)

from langchain_huggingface import (
    HuggingFaceEmbeddings
)

from langchain_community.vectorstores import (
    FAISS
)

from langchain_groq import ChatGroq


class AcademicAssistant:

    def __init__(
        self,
        groq_api_key: str,
        db_path: str,
        model: str = "openai/gpt-oss-120b",
    ):

        if not groq_api_key:

            raise ValueError(
                "Groq API key is required"
            )

        self.groq_api_key = (
            groq_api_key
        )

        self.db_path = db_path

        self.model = model

        self.embeddings = None
        self.vectorstore = None
        self.retriever = None
        self.llm = None
        self.chain = None

        self.system_prompt = """
You are an AI-powered Academic Assistant.

Answer questions ONLY using the
provided official academic context.

Do not use outside knowledge.

If the information cannot be supported
by the retrieved documents, say:

"Information not available in the retrieved documents."

Always identify the source/page when possible.
"""

        self.prompt = PromptTemplate(
            template="""
{system_prompt}

Context:
{context}

Question:
{question}

Answer ONLY from the context.
""",
            input_variables=[
                "system_prompt",
                "context",
                "question",
            ],
        )

    # ========================================================
    # EMBEDDINGS
    # ========================================================

    def load_embeddings(self):

        self.embeddings = (
            HuggingFaceEmbeddings(
                model_name=
                "sentence-transformers/all-MiniLM-L6-v2"
            )
        )

    # ========================================================
    # VECTORSTORE
    # ========================================================

    def load_vectorstore(self):

        if not os.path.exists(
            self.db_path
        ):

            raise FileNotFoundError(
                f"Vectorstore not found: "
                f"{self.db_path}"
            )

        self.vectorstore = (
            FAISS.load_local(
                self.db_path,
                self.embeddings,
                allow_dangerous_deserialization=True,
            )
        )

        self.retriever = (
            self.vectorstore
            .as_retriever(
                search_kwargs={
                    "k": 3
                }
            )
        )

    # ========================================================
    # LLM
    # ========================================================

    def load_llm(self):

        self.llm = ChatGroq(
            api_key=self.groq_api_key,
            model=self.model,
            temperature=0,
            max_tokens=250,
        )

    # ========================================================
    # FORMAT CONTEXT
    # ========================================================

    def format_docs(
        self,
        docs: List
    ):

        output = []

        for doc in docs:

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

            output.append(
                f"""
[Source: {source}]
[Page: {page}]

{doc.page_content}
"""
            )

        return "\n\n".join(
            output
        )

    # ========================================================
    # CHAIN
    # ========================================================

    def build_chain(self):

        self.chain = (

            {
                "context":
                    self.retriever
                    | self.format_docs,

                "question":
                    RunnablePassthrough(),

                "system_prompt":
                    lambda _: (
                        self.system_prompt
                    ),
            }

            | self.prompt

            | self.llm

            | StrOutputParser()
        )

    # ========================================================
    # INITIALIZE
    # ========================================================

    def initialize(self):

        self.load_embeddings()

        self.load_vectorstore()

        self.load_llm()

        self.build_chain()

    # ========================================================
    # QUERY
    # ========================================================

    def query(
        self,
        question
    ):

        return self.chain.invoke(
            question
        )


if __name__ == "__main__":

    GROQ_API_KEY = os.getenv(
        "GROQ_API_KEY"
    )

    INDEX = (
        "vectorstore/fixed_500"
    )

    assistant = AcademicAssistant(
        GROQ_API_KEY,
        INDEX
    )

    assistant.initialize()

    print(
        "\nAcademic RAG Assistant"
    )

    while True:

        question = input(
            "\nQuestion: "
        ).strip()

        if question.lower() in {
            "exit",
            "quit",
            "q",
        }:

            break

        answer = assistant.query(
            question
        )

        print(
            "\nAnswer:\n",
            answer
        )
