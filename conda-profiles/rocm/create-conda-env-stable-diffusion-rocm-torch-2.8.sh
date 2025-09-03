#!/bin/bash

ENV_NAME="stable-diffusion-rocm"
PYTHON_VERSION="3.12"
ROCM_VERSION="6.3"

if conda info --envs | grep -q "^$ENV_NAME\\s"; then
    echo "Conda environment '$ENV_NAME' already exists. Removing it."
    conda remove --name $ENV_NAME --all -y
fi

conda create --name $ENV_NAME python=$PYTHON_VERSION -y

echo "Activating conda environment '$ENV_NAME'..."
conda run -n $ENV_NAME pip install pip wheel --upgrade

echo "Installing PyTorch 2.8.0 with ROCm ${ROCM_VERSION} support..."
conda run -n $ENV_NAME pip install torch==2.8.0+rocm${ROCM_VERSION} torchvision==0.23.0+rocm${ROCM_VERSION} torchaudio==2.8.0+rocm${ROCM_VERSION} pytorch_triton==3.4.0 --extra-index-url https://download.pytorch.org/whl/
echo "Installing additional dependencies..."
#conda run -n $ENV_NAME pip install sentencepiece transformers hf_xet accelerate diffusers protobuf sacremoses
conda run -n $ENV_NAME pip install transformers diffusers accelerate

# clear huggingface model cache
#rm -rf ~/.cache/huggingface/hub/

echo "Conda environment '$ENV_NAME' with PyTorch 2.8.0 is set up successfully."
conda activate $ENV_NAME
