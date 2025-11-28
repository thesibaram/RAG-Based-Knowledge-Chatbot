"""Simple CLI demo for the Hospital Review RAG Chatbot."""

import logging

from src.config import (
    API_KEY_ENV_VAR,
    CHAT_MODEL,
    CHROMA_DB_PATH,
    EMBEDDING_MODEL,
    TOP_K_RETRIEVAL,
)
from src.rag_chain import RAGChainConfig, ReviewRAGChain
from src.utils import get_api_key, setup_logging
from src.vectorstore import VectorStoreManager

logger = logging.getLogger(__name__)


def main():
    """Run a simple CLI demo of the chatbot."""
    setup_logging()

    print("\n" + "=" * 80)
    print("🏥 Hospital Review RAG Chatbot - CLI Demo")
    print("=" * 80)
    print("\nInitializing chatbot...")

    try:
        api_key = get_api_key(API_KEY_ENV_VAR)

        vector_store_manager = VectorStoreManager(
            persist_directory=CHROMA_DB_PATH,
            embedding_model=EMBEDDING_MODEL,
            api_key=api_key,
        )
        vector_store = vector_store_manager.load_vector_store()

        if vector_store is None:
            print("\n❌ Error: Vector store not found!")
            print("Please run: python build_vectorstore.py")
            return

        config = RAGChainConfig(
            chat_model=CHAT_MODEL,
            api_key=api_key,
            top_k=TOP_K_RETRIEVAL,
        )
        rag_chain = ReviewRAGChain(vector_store=vector_store, config=config)

        print("✅ Chatbot initialized successfully!\n")
        print("Try these example questions:")
        print("  1. Has anyone complained about communication with the hospital staff?")
        print("  2. What did patients say about the discharge process?")
        print("  3. Were there any positive experiences mentioned?")
        print("\nType 'quit' to exit.\n")
        print("=" * 80)

        while True:
            question = input("\n💬 Your question: ").strip()

            if question.lower() in ["quit", "exit", "q"]:
                print("\nGoodbye! 👋\n")
                break

            if not question:
                continue

            print("\n🤖 Assistant: ", end="", flush=True)
            try:
                response = rag_chain.answer_question(question)
                print(response)
            except Exception as e:
                print(f"\n❌ Error: {e}")

    except Exception as e:
        logger.error(f"Failed to start demo: {e}", exc_info=True)
        print(f"\n❌ Error: {e}")


if __name__ == "__main__":
    main()
