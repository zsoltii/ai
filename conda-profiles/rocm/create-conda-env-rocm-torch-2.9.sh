#!/bin/bash

if conda info --envs | grep -q "^rocm\s"; then
    echo "Conda environment 'rocm' already exists. Removing it."
    conda remove --name rocm --all -y
fi

conda create --name rocm python=3.13 -y

TORCH_VERSION="2.9.0+rocm6.4"
TORCHVISION_VERSION="0.24.0+rocm6.4"
PYTORCH_TRITION_VERSION="3.4.0"

echo "Activating conda environment 'rocm'..."
conda run -n rocm pip install pip wheel --upgrade

echo "Installing PyTorch ${TORCH_VERSION} support..."
conda run -n rocm pip install torch==${TORCH_VERSION} torchvision==${TORCHVISION_VERSION} torchaudio==${TORCH_VERSION} pytorch_triton==${PYTORCH_TRITION_VERSION} --extra-index-url https://download.pytorch.org/whl/
echo "Installing additional dependencies..."

conda run -n rocm pip install sentencepiece transformers hf_xet accelerate diffusers protobuf xformers sacremoses peft scipy backoff

echo "Installing bitsandbytes rocm version"
# sudo apt-get install -y build-essential cmake
rm -rf bitsandbytes
git clone -b multi-backend-refactor https://github.com/bitsandbytes-foundation/bitsandbytes.git && cd bitsandbytes/
cmake -DCOMPUTE_BACKEND=hip -S .
make
conda run -n rocm pip install -e .
cd ..
rm -rf bitsandbytes


# clear huggingface model cache
#rm -rf ~/.cache/huggingface/hub/

echo "Conda environment 'rocm' with ${TORCH_VERSION} is set up successfully."