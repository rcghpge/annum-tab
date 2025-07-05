k#!/bin/sh

# Exit immediately if a command exits with a non-zero status
set -e

# Dataset URL
DATASET_URL="https://storage.googleapis.com/mathematics-dataset/mathematics_dataset-v1.0.tar.gz"

# Output file name
OUTPUT_FILE="math_dataset-v1.0.tar.gz"

echo "Downloading DeepMind Math Dataset from:"
echo "$DATASET_URL"

# Download with curl
curl -L -o "$OUTPUT_FILE" "$DATASET_URL"

echo "Download complete: $OUTPUT_FILE"

