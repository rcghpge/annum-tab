#!/bin/sh

# Exit immediately if a command exits with a non-zero status
set -e

# Input tar.gz file
INPUT_FILE="math_dataset-v1.0.tar.gz"

# Output directory
OUTPUT_DIR="math_dataset"

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

echo "Extracting $INPUT_FILE to $OUTPUT_DIR"

# Extract
tar -xzf "$INPUT_FILE" -C "$OUTPUT_DIR"

echo "Extraction complete! Files are in: $OUTPUT_DIR"
