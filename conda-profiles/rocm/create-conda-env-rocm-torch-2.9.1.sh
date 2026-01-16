#!/bin/bash

if conda info --envs | grep -q "^rocm\s"; then
    echo "Conda environment 'rocm' already exists. Removing it."
    conda remove --name rocm --all -y
fi

conda create --name rocm python=3.13 -y

TORCH_VERSION="2.9.1+rocm6.4"
TORCHVISION_VERSION="0.24.1+rocm6.4"
PYTORCH_TRITION_VERSION="3.4.0"

echo "Activating conda environment 'rocm'..."
conda run -n rocm pip install pip wheel --upgrade

echo "Installing PyTorch ${TORCH_VERSION} support..."
conda run -n rocm pip install torch==${TORCH_VERSION} torchvision==${TORCHVISION_VERSION} torchaudio==${TORCH_VERSION} pytorch_triton==${PYTORCH_TRITION_VERSION} --extra-index-url https://download.pytorch.org/whl/

echo "Installing additional dependencies..."
conda run -n rocm pip install sentencepiece transformers hf_xet accelerate diffusers protobuf sacremoses peft scipy backoff peft datasets trl ctranslate2

echo "Installing xformers rocm version"
export PYTORCH_ROCM_ARCH="gfx803;gfx900;gfx906;gfx908;gfx90a;gfx940;gfx941;gfx942;gfx1010;gfx1011;gfx1012;gfx1030;gfx1031;gfx1100;gfx1101;gfx1102;gfx1103"
conda run -n rocm pip install --no-cache-dir "git+https://github.com/facebookresearch/xformers.git@v0.0.33"

echo "Installing additional dependencies..."
conda run -n rocm pip install ctranslate2 onnxruntime-rocm optimum-onnx

echo "Installing bitsandbytes rocm version"
# sudo amdgpu-install # you need rocm 6.4.x version
# sudo apt-get install -y build-essential cmake rocm-dev rocm-libs
rm -rf bitsandbytes
git clone https://github.com/bitsandbytes-foundation/bitsandbytes.git && cd bitsandbytes/
conda run -n rocm cmake -DCOMPUTE_BACKEND=hip -S .
conda run -n rocm make
conda run -n rocm pip install .
cd ..
rm -rf bitsandbytes

# clear huggingface model cache
#rm -rf ~/.cache/huggingface/hub/

echo "Conda environment 'rocm' with ${TORCH_VERSION} is set up successfully."