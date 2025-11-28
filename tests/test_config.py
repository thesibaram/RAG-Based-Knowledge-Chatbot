"""Basic tests for configuration constants."""

from pathlib import Path

from src import config


def test_data_path_exists():
    assert isinstance(config.REVIEWS_CSV_PATH, Path)
    assert config.REVIEWS_CSV_PATH.name == "reviews.csv"


def test_chroma_directory():
    assert config.CHROMA_DB_PATH.name == "chroma_data"
    assert config.CHROMA_DB_PATH.parent.name == "artifacts"
