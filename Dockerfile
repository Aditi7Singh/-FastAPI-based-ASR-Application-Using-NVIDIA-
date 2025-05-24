# Use Python slim base image for smaller size
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libsndfile1 \
    libsndfile1-dev \
    ffmpeg \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better Docker layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Create models directory
RUN mkdir -p models

# Copy application code
COPY main.py .
COPY convert_to_onnx.py .
COPY download_model.sh .

# Make the download script executable
RUN chmod +x download_model.sh

# Set environment variables
ENV MODEL_PATH="/app/models/stt_hi_conformer_ctc_medium.onnx"
ENV PYTHONPATH="/app"
ENV CUDA_VISIBLE_DEVICES=""

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash app
RUN chown -R app:app /app
USER app

# Download and convert model during build (optional - can be done at runtime)
# RUN python convert_to_onnx.py --model nvidia/stt_hi_conformer_ctc_medium --output-dir ./models --verify

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start the application
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
