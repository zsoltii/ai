#!/bin/bash

ENVIRONMENT_NAME="rocm-unsloth"

if conda info --envs | grep -q "^${ENVIRONMENT_NAME}\s"; then
    echo "Conda environment '${ENVIRONMENT_NAME}' already exists. Removing it."
    conda remove --name ${ENVIRONMENT_NAME} --all -y
fi

conda create --name ${ENVIRONMENT_NAME} python=3.13 -y

echo "Activating conda environment '${ENVIRONMENT_NAME}'..."
conda run -n ${ENVIRONMENT_NAME} pip install pip wheel --upgrade

echo "Installing PyTorch support..."
conda run -n ${ENVIRONMENT_NAME} pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.4

echo "Installing unsloth dependencies..."
conda run -n ${ENVIRONMENT_NAME} pip install "unsloth[amd] @ git+https://github.com/unslothai/unsloth.git"

# clear huggingface model cache
#rm -rf ~/.cache/huggingface/hub/

echo "Conda environment '${ENVIRONMENT_NAME}' with ${TORCH_VERSION} is set up successfully."