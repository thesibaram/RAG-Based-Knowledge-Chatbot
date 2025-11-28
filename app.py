"""Main application entry point for the Hospital Review RAG Chatbot."""

import argparse
import logging
from typing import List, Tuple

import gradio as gr

from src.config import (
    API_KEY_ENV_VAR,
    BATCH_SIZE,
    BATCH_WAIT_TIME,
    CHAT_MODEL,
    CHROMA_DB_PATH,
    EMBEDDING_MODEL,
    REVIEWS_CSV_PATH,
    TOP_K_RETRIEVAL,
)
from src.data_loader import ReviewDataLoader
from src.embeddings import BatchEmbeddingProcessor
from src.rag_chain import RAGChainConfig, ReviewRAGChain
from src.utils import ensure_directory, get_api_key, setup_logging
from src.vectorstore import VectorStoreManager

logger = logging.getLogger(__name__)


def setup_vector_database(api_key: str, recreate: bool = False):
    """Set up or load the vector database."""
    vector_store_manager = VectorStoreManager(
        persist_directory=CHROMA_DB_PATH,
        embedding_model=EMBEDDING_MODEL,
        api_key=api_key,
    )

    if recreate or not CHROMA_DB_PATH.exists():
        logger.info("Creating new vector store...")
        data_loader = ReviewDataLoader(csv_path=REVIEWS_CSV_PATH)
        reviews = data_loader.load_reviews()

        batch_processor = BatchEmbeddingProcessor(
            batch_size=BATCH_SIZE,
            wait_time=BATCH_WAIT_TIME,
        )
        vector_db = batch_processor.process_documents_in_batches(
            documents=reviews,
            vector_store_manager=vector_store_manager,
        )
    else:
        logger.info("Loading existing vector store...")
        vector_db = vector_store_manager.load_vector_store()
        if vector_db is None:
            raise RuntimeError(
                "Vector store not found. Run `python build_vectorstore.py` or start the app with --recreate-db."
            )

    return vector_db


def build_chatbot(api_key: str, recreate_db: bool = False):
    """Build the complete RAG chatbot."""
    ensure_directory(CHROMA_DB_PATH.parent)

    vector_store = setup_vector_database(api_key, recreate=recreate_db)

    rag_config = RAGChainConfig(
        chat_model=CHAT_MODEL,
        api_key=api_key,
        top_k=TOP_K_RETRIEVAL,
    )
    rag_chain = ReviewRAGChain(vector_store=vector_store, config=rag_config)
    return rag_chain


def respond_to_user_question(question: str, history: List[Tuple[str, str]], rag_chain: ReviewRAGChain) -> str:
    """Process user questions and return responses."""
    try:
        return rag_chain.answer_question(question)
    except Exception as e:
        logger.error(f"Error processing question: {e}", exc_info=True)
        return f"Sorry, I encountered an error: {str(e)}"


def launch_gradio_interface(rag_chain: ReviewRAGChain, share: bool = False):
    """Launch the Gradio chat interface."""
    interface = gr.ChatInterface(
        fn=lambda question, history: respond_to_user_question(question, history, rag_chain),
        title="🏥 Hospital Review Assistant",
        description="Ask questions about patient experiences at hospitals based on real reviews.",
        examples=[
            "Has anyone complained about communication with the hospital staff?",
            "What did patients say about the discharge process?",
            "Were there any positive experiences mentioned?",
            "What are common complaints about the facilities?",
        ],
        theme=gr.themes.Soft(),
    )

    interface.launch(share=share, server_name="0.0.0.0", server_port=7860)


def main():
    """Main function to run the chatbot application."""
    parser = argparse.ArgumentParser(description="Hospital Review RAG Chatbot")
    parser.add_argument(
        "--recreate-db",
        action="store_true",
        help="Recreate the vector database from scratch",
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create a public share link for the Gradio interface",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Set the logging level",
    )

    args = parser.parse_args()

    setup_logging(getattr(logging, args.log_level))
    logger.info("Starting Hospital Review RAG Chatbot")

    try:
        api_key = get_api_key(API_KEY_ENV_VAR)
        rag_chain = build_chatbot(api_key, recreate_db=args.recreate_db)
        logger.info("Chatbot initialized successfully")
        launch_gradio_interface(rag_chain, share=args.share)
    except Exception as e:
        logger.error(f"Failed to start chatbot: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
