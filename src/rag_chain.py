"""Defines the retrieval-augmented generation chain for the chatbot."""

import logging
from dataclasses import dataclass
from typing import List, Optional

from langchain.schema import Document
from langchain.schema.runnable import RunnablePassthrough
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    PromptTemplate,
    SystemMessagePromptTemplate,
)

from .config import HUMAN_PROMPT_TEMPLATE, SYSTEM_PROMPT_TEMPLATE, TEMPERATURE

logger = logging.getLogger(__name__)


@dataclass
class RAGChainConfig:
    """Configuration for the RAG chain."""

    chat_model: str
    api_key: str
    top_k: int


class ReviewRAGChain:
    """Handles RAG operations for answering user queries."""

    def __init__(self, vector_store: Chroma, config: RAGChainConfig) -> None:
        self.vector_store = vector_store
        self.config = config
        self.prompt = self._build_prompt_template()
        self.chat_model = ChatGoogleGenerativeAI(
            model=self.config.chat_model,
            temperature=TEMPERATURE,
            google_api_key=self.config.api_key,
        )

        self.retriever = self.vector_store.as_retriever(k=self.config.top_k)
        self.chain = (
            {"context": self.retriever, "question": RunnablePassthrough()}
            | self.prompt
            | self.chat_model
            | StrOutputParser()
        )
        logger.info("RAG chain initialized with model %s", self.config.chat_model)

    @staticmethod
    def _build_prompt_template() -> ChatPromptTemplate:
        system_prompt = SystemMessagePromptTemplate(
            prompt=PromptTemplate(
                input_variables=["context"],
                template=SYSTEM_PROMPT_TEMPLATE,
            )
        )
        human_prompt = HumanMessagePromptTemplate(
            prompt=PromptTemplate(
                input_variables=["question"],
                template=HUMAN_PROMPT_TEMPLATE,
            )
        )
        return ChatPromptTemplate.from_messages([system_prompt, human_prompt])

    def answer_question(self, question: str) -> str:
        """Generate an answer for the given question."""
        logger.debug("Answering question: %s", question)
        return self.chain.invoke(question)

    def retrieve_relevant_documents(self, question: str, k: Optional[int] = None) -> List[Document]:
        """Retrieve relevant documents for a question."""
        k_value = k or self.config.top_k
        logger.debug("Retrieving %d documents for question: %s", k_value, question)
        return self.vector_store.similarity_search(question, k_value)
