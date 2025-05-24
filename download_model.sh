#!/bin/bash

# Download and convert NeMo ASR model to ONNX
set -e

echo "Starting model download and conversion process..."

# Create models directory if it doesn't exist
mkdir -p models

# Check if model already exists
MODEL_PATH="models/stt_hi_conformer_ctc_medium.onnx"
if [ -f "$MODEL_PATH" ]; then
    echo "Model already exists at $MODEL_PATH"
    exit 0
fi

# Install additional dependencies if needed
echo "Installing additional dependencies..."
pip install --no-cache-dir torch torchaudio --index-url https://download.pytorch.org/whl/cpu

# Convert model to ONNX
echo "Converting NeMo model to ONNX format..."
python convert_to_onnx.py \
    --model nvidia/stt_hi_conformer_ctc_medium \
    --output-dir ./models \
    --verify

echo "Model conversion completed successfully!"

# List files in models directory
echo "Files in models directory:"
ls -la models/

echo "Model is ready for use!"
