#!/bin/bash

if conda info --envs | grep -q "^rocm\s"; then
    echo "Conda environment 'rocm' already exists. Removing it."
    conda remove --name rocm --all -y
fi

conda create --name rocm python=3.13 -y

echo "Activating conda environment 'rocm'..."
conda run -n rocm pip install pip wheel --upgrade

echo "Installing PyTorch 2.8.0 with ROCm 6.4 support..."
conda run -n rocm pip install torch==2.8.0+rocm6.4 torchvision==0.23.0+rocm6.4 torchaudio==2.8.0+rocm6.4 pytorch_triton==3.4.0 --extra-index-url https://download.pytorch.org/whl/
echo "Installing additional dependencies..."
conda run -n rocm pip install sentencepiece transformers hf_xet accelerate diffusers protobuf xformers sacremoses

# clear huggingface model cache
#rm -rf ~/.cache/huggingface/hub/

echo "Conda environment 'rocm' with PyTorch 2.8.0 is set up successfully."