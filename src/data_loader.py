"""Data loading functionality for the RAG chatbot."""

import logging
from pathlib import Path
from typing import List

from langchain.document_loaders.csv_loader import CSVLoader
from langchain.schema import Document

logger = logging.getLogger(__name__)


class ReviewDataLoader:
    """Loads and processes hospital review data from CSV files."""

    def __init__(self, csv_path: Path, source_column: str = "review"):
        """
        Initialize the ReviewDataLoader.

        Args:
            csv_path: Path to the CSV file containing reviews.
            source_column: Name of the column containing review text.
        """
        self.csv_path = csv_path
        self.source_column = source_column

    def load_reviews(self) -> List[Document]:
        """
        Load reviews from the CSV file.

        Returns:
            List of Document objects containing the reviews.

        Raises:
            FileNotFoundError: If the CSV file doesn't exist.
        """
        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV file not found at {self.csv_path}")

        logger.info(f"Loading reviews from {self.csv_path}")
        loader = CSVLoader(file_path=str(self.csv_path), source_column=self.source_column)
        reviews = loader.load()
        logger.info(f"Loaded {len(reviews)} reviews successfully")
        return reviews

    def get_review_count(self, reviews: List[Document]) -> int:
        """Return the number of reviews."""
        return len(reviews)
