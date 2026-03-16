#!/bin/bash

ENV_NAME="nvidia"

if conda info --envs | grep -q "^${ENV_NAME}\s"; then
    echo "Conda environment '${ENV_NAME}' already exists. Removing it."
    conda remove --name ${ENV_NAME} --all -y
fi

conda create --name ${ENV_NAME} python=3.14 -y

echo "Activating conda environment '${ENV_NAME}'..."
conda run -n ${ENV_NAME} pip install pip wheel --upgrade

echo "Installing PyTorch ${TORCH_VERSION} support..."
conda run -n ${ENV_NAME} pip install torch torchvision torchaudio pytorch_triton
echo "Installing additional dependencies..."

conda run -n ${ENV_NAME} pip install sentencepiece transformers hf_xet accelerate diffusers protobuf xformers sacremoses peft scipy backoff peft datasets trl bitsandbytes ctranslate2 ollama

# clear huggingface model cache
# rm -rf ~/.cache/huggingface/hub/

echo "Conda environment '${ENV_NAME}' with ${TORCH_VERSION} is set up successfully."