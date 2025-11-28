"""Utility functions for the RAG chatbot application."""

import logging
import os
from pathlib import Path

from dotenv import load_dotenv


def setup_logging(log_level: int = logging.INFO) -> None:
    """Configure logging for the application."""
    logging.basicConfig(level=log_level, format="%(asctime)s - %(levelname)s - %(message)s")


def ensure_directory(path: Path) -> None:
    """Ensure a directory exists."""
    path.mkdir(parents=True, exist_ok=True)


def get_api_key(env_var: str = "GOOGLE_API_KEY") -> str:
    """Retrieve the API key from an environment variable."""
    load_dotenv()
    api_key = os.getenv(env_var)
    if not api_key:
        raise ValueError(f"API key not found in environment variable {env_var}")
    return api_key
