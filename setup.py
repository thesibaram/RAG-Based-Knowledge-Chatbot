"""Setup file for the Hospital Review RAG Chatbot package."""

from pathlib import Path
from setuptools import setup, find_packages

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name="hospital-review-rag-chatbot",
    version="1.0.0",
    author="ML Intern Project Maintainers",
    description="A RAG chatbot for answering questions about hospital patient reviews",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/hospital-review-rag-chatbot",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Healthcare Industry",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.9",
    install_requires=[
        "langchain>=0.3.13",
        "langchain-core>=0.3.28",
        "langchain-community>=0.3.13",
        "langchain-chroma>=0.1.4",
        "langchain-google-genai>=2.0.8",
        "chromadb>=0.5.23",
        "google-generativeai>=0.8.3",
        "gradio>=5.7.1",
        "pandas>=2.2.3",
        "numpy>=2.2.1",
        "python-dotenv>=1.0.1",
    ],
    entry_points={
        "console_scripts": [
            "rag-chatbot=app:main",
            "rag-build-db=build_vectorstore:main",
            "rag-evaluate=evaluate:main",
            "rag-demo=demo:main",
        ],
    },
)
