#!/bin/bash

# Ensure we're in the right directory
cd "$(dirname "$0")"

# Python command
PYTHON_CMD="python3.10"

# Check if an image path was provided
if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <image_path> [model_type]"
    echo "  image_path: Path to the food image"
    echo "  model_type: 'cnn' or 'vit' (default: cnn)"
    exit 1
fi

# Default model type to CNN if not specified
MODEL_TYPE=${2:-"cnn"}

# Run the classification script
if [ "$MODEL_TYPE" == "cnn" ]; then
    echo "Using CNN model..."
    $PYTHON_CMD classify_food.py --image "$1" --model cnn --cnn_model custom_cnn_final.keras
elif [ "$MODEL_TYPE" == "vit" ]; then
    echo "Using ViT model..."
    $PYTHON_CMD classify_food.py --image "$1" --model vit --vit_model vit_best_food101.pth
else
    echo "Invalid model type. Please specify 'cnn' or 'vit'."
    exit 1
fi 