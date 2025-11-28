"""Quick test script to validate the setup without running the full app."""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_imports():
    """Test that all required packages can be imported."""
    print("Testing imports...")
    try:
        import langchain
        import langchain_google_genai
        import langchain_chroma
        import gradio
        import pandas
        print("✅ All required packages imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False


def test_src_modules():
    """Test that src modules can be imported."""
    print("\nTesting src modules...")
    try:
        from src import config, utils, data_loader, vectorstore, embeddings, rag_chain, evaluation

        print("✅ All src modules imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False


def test_config():
    """Test configuration values."""
    print("\nTesting configuration...")
    try:
        from src.config import REVIEWS_CSV_PATH, CHROMA_DB_PATH, EMBEDDING_MODEL, CHAT_MODEL

        print(f"  Data path: {REVIEWS_CSV_PATH}")
        print(f"  DB path: {CHROMA_DB_PATH}")
        print(f"  Embedding model: {EMBEDDING_MODEL}")
        print(f"  Chat model: {CHAT_MODEL}")
        print("✅ Configuration loaded successfully")
        return True
    except Exception as e:
        print(f"❌ Configuration error: {e}")
        return False


def test_data_file():
    """Test that the data file exists."""
    print("\nTesting data file...")
    try:
        from src.config import REVIEWS_CSV_PATH

        if REVIEWS_CSV_PATH.exists():
            print(f"✅ Data file exists at {REVIEWS_CSV_PATH}")
            return True
        else:
            print(f"❌ Data file not found at {REVIEWS_CSV_PATH}")
            return False
    except Exception as e:
        print(f"❌ Error checking data file: {e}")
        return False


def main():
    """Run all tests."""
    print("\n" + "=" * 80)
    print("QUICK SETUP TEST")
    print("=" * 80 + "\n")

    results = []
    results.append(test_imports())
    results.append(test_src_modules())
    results.append(test_config())
    results.append(test_data_file())

    print("\n" + "=" * 80)
    if all(results):
        print("✅ ALL TESTS PASSED - Setup is complete!")
        print("\nNext steps:")
        print("  1. Set up your .env file with GOOGLE_API_KEY")
        print("  2. Run: python build_vectorstore.py")
        print("  3. Run: python app.py")
    else:
        print("❌ SOME TESTS FAILED - Please fix the issues above")
        sys.exit(1)
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
