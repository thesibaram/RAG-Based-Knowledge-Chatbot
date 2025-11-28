"""Evaluation utilities for the RAG chatbot."""

import logging
from dataclasses import dataclass
from typing import Dict, List

import pandas as pd
from langchain_chroma import Chroma

logger = logging.getLogger(__name__)


@dataclass
class EvaluationSample:
    """Represents a single evaluation sample."""

    question: str
    expected_keywords: List[str]


class RetrieverEvaluator:
    """Evaluates the retriever performance using keyword matching."""

    def __init__(self, vector_store: Chroma, top_k: int = 5) -> None:
        self.vector_store = vector_store
        self.top_k = top_k

    def evaluate(self, samples: List[EvaluationSample]) -> pd.DataFrame:
        """Evaluate the retriever against provided samples."""
        results = []

        for sample in samples:
            retrieved_docs = self.vector_store.similarity_search(sample.question, self.top_k)
            combined_text = " ".join(doc.page_content.lower() for doc in retrieved_docs)
            keyword_hits = sum(keyword.lower() in combined_text for keyword in sample.expected_keywords)
            hit_rate = keyword_hits / len(sample.expected_keywords)

            results.append(
                {
                    "question": sample.question,
                    "keywords": ", ".join(sample.expected_keywords),
                    "hit_rate": round(hit_rate, 2),
                    "documents_retrieved": len(retrieved_docs),
                }
            )

        df_results = pd.DataFrame(results)
        logger.info("Evaluation completed. Average hit rate: %.2f", df_results["hit_rate"].mean())
        return df_results


def summarize_evaluation(df_results: pd.DataFrame) -> Dict[str, float]:
    """Generate summary statistics from evaluation results."""
    return {
        "average_hit_rate": df_results["hit_rate"].mean(),
        "median_hit_rate": df_results["hit_rate"].median(),
        "min_hit_rate": df_results["hit_rate"].min(),
        "max_hit_rate": df_results["hit_rate"].max(),
    }
