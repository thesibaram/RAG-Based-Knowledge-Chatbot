# Quick Start Guide

Get the Hospital Review RAG Chatbot up and running in 5 minutes!

## Prerequisites

- Python 3.9+
- Google API Key ([Get one here](https://makersuite.google.com/app/apikey))

## Step 1: Clone and Install

```bash
git clone <your-repo-url>
cd hospital-review-rag-chatbot
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Step 2: Configure API Key

```bash
cp .env.example .env
```

Edit `.env` and add your Google API key:
```
GOOGLE_API_KEY=your_actual_key_here
```

## Step 3: Verify Setup

```bash
python scripts/quick_test.py
```

This should output:
```
✅ ALL TESTS PASSED - Setup is complete!
```

## Step 4: Build Vector Database

```bash
python build_vectorstore.py
```

⏱️ This takes ~25 minutes for 1000 reviews (batched to avoid rate limits).

## Step 5: Launch the Chatbot

```bash
python app.py
```

🎉 Open your browser to: `http://localhost:7860`

## Try These Questions

- "Has anyone complained about communication with the hospital staff?"
- "What did patients say about the discharge process?"
- "Were there any positive experiences mentioned?"

## Alternative Interfaces

### CLI Demo
```bash
python demo.py
```

### Evaluation
```bash
python evaluate.py
```

### Docker (Recommended for Production)
```bash
docker-compose up --build
```

## Troubleshooting

### "Vector store not found"
Run: `python build_vectorstore.py` first

### "API key not found"
Check your `.env` file has `GOOGLE_API_KEY=...`

### Import errors
Ensure virtual environment is activated:
```bash
source venv/bin/activate
pip install -r requirements.txt
```

## Next Steps

- Read the full [README.md](README.md) for detailed documentation
- Check [CONTRIBUTING.md](CONTRIBUTING.md) to contribute
- Review [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md) for an overview

---

**Need help?** Open an issue on GitHub!
