"""Standalone script to build and save the vector database."""

import argparse
import logging

from src.config import (
    API_KEY_ENV_VAR,
    BATCH_SIZE,
    BATCH_WAIT_TIME,
    CHROMA_DB_PATH,
    EMBEDDING_MODEL,
    REVIEWS_CSV_PATH,
)
from src.data_loader import ReviewDataLoader
from src.embeddings import BatchEmbeddingProcessor
from src.utils import ensure_directory, get_api_key, setup_logging
from src.vectorstore import VectorStoreManager

logger = logging.getLogger(__name__)


def main():
    """Build the vector store from scratch."""
    parser = argparse.ArgumentParser(description="Build the vector database for the RAG chatbot")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Set the logging level",
    )
    args = parser.parse_args()

    setup_logging(getattr(logging, args.log_level))

    try:
        api_key = get_api_key(API_KEY_ENV_VAR)
        logger.info("Starting vector database build process")

        ensure_directory(CHROMA_DB_PATH.parent)

        data_loader = ReviewDataLoader(csv_path=REVIEWS_CSV_PATH)
        reviews = data_loader.load_reviews()

        vector_store_manager = VectorStoreManager(
            persist_directory=CHROMA_DB_PATH,
            embedding_model=EMBEDDING_MODEL,
            api_key=api_key,
        )

        batch_processor = BatchEmbeddingProcessor(
            batch_size=BATCH_SIZE,
            wait_time=BATCH_WAIT_TIME,
        )

        vector_db = batch_processor.process_documents_in_batches(
            documents=reviews,
            vector_store_manager=vector_store_manager,
        )

        logger.info(f"Vector database created successfully at {CHROMA_DB_PATH}")
        logger.info("You can now run 'python app.py' to start the chatbot")

    except Exception as e:
        logger.error(f"Failed to build vector database: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
