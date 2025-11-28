# Use Python 3.11 slim image for smaller size
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY app.py .
COPY build_vectorstore.py .
COPY evaluate.py .
COPY demo.py .

# Copy data directory (optional, can mount as volume instead)
COPY data/ ./data/

# Create artifacts directory for vector store
RUN mkdir -p artifacts

# Expose Gradio default port
EXPOSE 7860

# Set environment variable to avoid buffering
ENV PYTHONUNBUFFERED=1

# Default command
CMD ["python", "app.py"]
