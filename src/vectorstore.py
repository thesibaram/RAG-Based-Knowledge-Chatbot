"""Vector store management for the RAG chatbot."""

import logging
import shutil
from pathlib import Path
from typing import List, Optional

from langchain.schema import Document
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from .utils import ensure_directory

logger = logging.getLogger(__name__)


class VectorStoreManager:
    """Manages creation and retrieval of the vector store."""

    def __init__(
        self,
        persist_directory: Path,
        embedding_model: str,
        api_key: str,
    ) -> None:
        """Initialize the vector store manager."""
        self.persist_directory = persist_directory
        self.embedding_model = embedding_model
        self.api_key = api_key
        self.embedding_function = GoogleGenerativeAIEmbeddings(
            model=self.embedding_model,
            google_api_key=self.api_key,
        )

    def create_vector_store(
        self,
        documents: List[Document],
        recreate: bool = False,
    ) -> Chroma:
        """Create a new vector store from documents."""
        if recreate and self.persist_directory.exists():
            logger.info("Recreating vector store: removing existing directory")
            shutil.rmtree(self.persist_directory)

        ensure_directory(self.persist_directory)
        logger.info("Creating vector store at %s", self.persist_directory)
        vector_store = Chroma.from_documents(
            documents=documents,
            embedding=self.embedding_function,
            persist_directory=str(self.persist_directory),
        )
        logger.info("Vector store created successfully with %d documents", len(documents))
        return vector_store

    def load_vector_store(self) -> Optional[Chroma]:
        """Load the existing vector store."""
        if not self.persist_directory.exists():
            logger.warning("Vector store directory does not exist at %s", self.persist_directory)
            return None

        logger.info("Loading vector store from %s", self.persist_directory)
        return Chroma(
            persist_directory=str(self.persist_directory),
            embedding_function=self.embedding_function,
        )
