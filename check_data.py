"""Utility script to check data integrity and structure."""

import logging

import pandas as pd

from src.config import REVIEWS_CSV_PATH
from src.utils import setup_logging

logger = logging.getLogger(__name__)


def main():
    """Check data file integrity."""
    setup_logging()

    logger.info("Checking data integrity...")

    if not REVIEWS_CSV_PATH.exists():
        logger.error(f"Data file not found at {REVIEWS_CSV_PATH}")
        return

    try:
        df = pd.read_csv(REVIEWS_CSV_PATH)

        print("\n" + "=" * 80)
        print("DATA INTEGRITY CHECK")
        print("=" * 80)

        print(f"\n✅ File exists: {REVIEWS_CSV_PATH}")
        print(f"✅ Total rows: {len(df)}")
        print(f"✅ Total columns: {len(df.columns)}")

        print("\n📊 Column Information:")
        print("-" * 80)
        for col in df.columns:
            non_null = df[col].notna().sum()
            null_count = df[col].isna().sum()
            print(f"  {col:20s}: {non_null:5d} non-null values ({null_count} missing)")

        print("\n📝 Sample Reviews:")
        print("-" * 80)
        if "review" in df.columns:
            for i, review in enumerate(df["review"].head(3), 1):
                preview = review[:100] + "..." if len(review) > 100 else review
                print(f"  {i}. {preview}\n")

        print("=" * 80)
        logger.info("Data integrity check complete!")

    except Exception as e:
        logger.error(f"Error reading data: {e}", exc_info=True)


if __name__ == "__main__":
    main()
