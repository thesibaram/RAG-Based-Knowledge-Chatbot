"""Evaluation script for the RAG chatbot."""

import argparse
import logging

from src.config import (
    API_KEY_ENV_VAR,
    CHROMA_DB_PATH,
    EMBEDDING_MODEL,
    TOP_K_RETRIEVAL,
)
from src.evaluation import EvaluationSample, RetrieverEvaluator, summarize_evaluation
from src.utils import get_api_key, setup_logging
from src.vectorstore import VectorStoreManager

logger = logging.getLogger(__name__)


def main():
    """Evaluate the RAG chatbot retriever."""
    parser = argparse.ArgumentParser(description="Evaluate the RAG chatbot")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Set the logging level",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=TOP_K_RETRIEVAL,
        help="Number of documents to retrieve",
    )
    args = parser.parse_args()

    setup_logging(getattr(logging, args.log_level))

    try:
        api_key = get_api_key(API_KEY_ENV_VAR)
        logger.info("Loading vector store for evaluation")

        vector_store_manager = VectorStoreManager(
            persist_directory=CHROMA_DB_PATH,
            embedding_model=EMBEDDING_MODEL,
            api_key=api_key,
        )
        vector_store = vector_store_manager.load_vector_store()

        if vector_store is None:
            logger.error("Vector store not found. Please run build_vectorstore.py first.")
            return

        # Define evaluation samples
        samples = [
            EvaluationSample(
                question="Has anyone complained about communication with the hospital staff?",
                expected_keywords=["communication", "staff", "coordination", "nursing"],
            ),
            EvaluationSample(
                question="What did patients say about the discharge process?",
                expected_keywords=["discharge", "process", "seamless", "released"],
            ),
            EvaluationSample(
                question="Were there any positive experiences mentioned?",
                expected_keywords=["positive", "great", "excellent", "wonderful"],
            ),
            EvaluationSample(
                question="What are common complaints about the facilities?",
                expected_keywords=["facilities", "parking", "room", "equipment"],
            ),
        ]

        evaluator = RetrieverEvaluator(vector_store, top_k=args.top_k)
        results = evaluator.evaluate(samples)

        print("\n" + "=" * 80)
        print("EVALUATION RESULTS")
        print("=" * 80)
        print(results.to_string(index=False))
        print("\n" + "=" * 80)

        summary = summarize_evaluation(results)
        print("SUMMARY STATISTICS")
        print("=" * 80)
        for metric, value in summary.items():
            print(f"{metric:20s}: {value:.2f}")
        print("=" * 80 + "\n")

    except Exception as e:
        logger.error(f"Evaluation failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
