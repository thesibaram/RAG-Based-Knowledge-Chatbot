"""Batch embedding functionality for handling API rate limits."""

import logging
import time
from typing import List

from langchain.schema import Document
from langchain_chroma import Chroma

logger = logging.getLogger(__name__)


class BatchEmbeddingProcessor:
    """Processes documents in batches to avoid API rate limits."""

    def __init__(
        self,
        batch_size: int = 20,
        wait_time: int = 30,
    ) -> None:
        """
        Initialize the batch processor.

        Args:
            batch_size: Number of documents to process per batch.
            wait_time: Time to wait between batches in seconds.
        """
        self.batch_size = batch_size
        self.wait_time = wait_time

    def process_documents_in_batches(
        self,
        documents: List[Document],
        vector_store_manager,
    ) -> Chroma:
        """
        Process documents in batches to create a vector store.

        Args:
            documents: List of documents to embed and store.
            vector_store_manager: Instance of VectorStoreManager.

        Returns:
            Chroma vector store with all embedded documents.
        """
        num_batches = (len(documents) - 1) // self.batch_size + 1
        logger.info(f"Processing {len(documents)} documents in {num_batches} batches")

        vector_db = None

        for i in range(0, len(documents), self.batch_size):
            batch_docs = documents[i : i + self.batch_size]
            current_batch_num = i // self.batch_size + 1

            logger.info(f"Processing batch {current_batch_num}/{num_batches}...")

            if i == 0:
                vector_db = vector_store_manager.create_vector_store(
                    documents=batch_docs,
                    recreate=True,
                )
            else:
                if vector_db is None:
                    raise RuntimeError("Vector store is not initialized. Ensure recreate=True for the first batch.")
                vector_db.add_documents(documents=batch_docs)

            if current_batch_num < num_batches:
                logger.info(f"Batch {current_batch_num} processed. Waiting {self.wait_time} seconds...")
                time.sleep(self.wait_time)
            else:
                logger.info(f"Batch {current_batch_num} processed (final batch). Persisting store...")
                if vector_db is not None:
                    vector_db.persist()

        logger.info("All batches processed successfully")
        return vector_db
