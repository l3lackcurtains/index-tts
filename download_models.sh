#!/bin/bash

# Create checkpoints directory if it doesn't exist
mkdir -p checkpoints

# Define model files to download
declare -a models=(
    "bigvgan_discriminator.pth"
    "bigvgan_generator.pth"
    "bpe.model"
    "dvae.pth"
    "gpt.pth"
    "unigram_12000.vocab"
)

# Base URL
BASE_URL="https://huggingface.co/IndexTeam/Index-TTS/resolve/main"

# Download each model
for model in "${models[@]}"; do
    echo "Downloading ${model}..."
    wget -q --show-progress "${BASE_URL}/${model}" -P checkpoints
    
    # Check if download was successful
    if [ $? -eq 0 ]; then
        echo "✓ Successfully downloaded ${model}"
    else
        echo "✗ Failed to download ${model}"
        exit 1
    fi
done

echo "All models downloaded successfully!"