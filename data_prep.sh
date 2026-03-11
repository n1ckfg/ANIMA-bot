#!/bin/bash

# Convert documents to markdown using docling

INPUT_DIR="data_prep"
OUTPUT_DIR="data"

if [ ! -d "$INPUT_DIR" ]; then
    echo "Error: Input directory '$INPUT_DIR' does not exist"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

source venv/bin/activate
echo "Converting documents from $INPUT_DIR to $OUTPUT_DIR..."
docling "$INPUT_DIR" --output "$OUTPUT_DIR"
deactivate

echo "Done."
