"""Configuration settings for the RAG chatbot application."""

from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "raw"
ARTIFACTS_DIR = BASE_DIR / "artifacts"

# File paths
REVIEWS_CSV_PATH = DATA_DIR / "reviews.csv"
CHROMA_DB_PATH = ARTIFACTS_DIR / "chroma_data"

# Model configurations
EMBEDDING_MODEL = "models/gemini-embedding-004"
CHAT_MODEL = "gemini-2.5-flash"
TEMPERATURE = 0
TOP_K_RETRIEVAL = 10

# Batch processing settings
BATCH_SIZE = 20
BATCH_WAIT_TIME = 30  # seconds

# API key environment variable name
API_KEY_ENV_VAR = "GOOGLE_API_KEY"

# Prompt template
SYSTEM_PROMPT_TEMPLATE = """Your job is to use patient reviews to answer questions about their experience at a hospital.
Use the following context to answer questions.
Be as detailed as possible, but don't make up any information that's not from the context.
If you don't know an answer, say you don't know.

{context}
"""

HUMAN_PROMPT_TEMPLATE = "{question}"
