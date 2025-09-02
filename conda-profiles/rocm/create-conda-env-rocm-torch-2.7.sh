#!/bin/bash

conda create --name rocm python=3.10 -y
conda run -n rocm pip install pip wheel --upgrade

#wget https://repo.radeon.com/rocm/manylinux/rocm-rel-6.4/torch-2.6.0%2Brocm6.4.0.git2fb0ac2b-cp310-cp310-linux_x86_64.whl
#wget https://repo.radeon.com/rocm/manylinux/rocm-rel-6.4/torchvision-0.21.0%2Brocm6.4.0.git4040d51f-cp310-cp310-linux_x86_64.whl
#wget https://repo.radeon.com/rocm/manylinux/rocm-rel-6.4/torchaudio-2.6.0%2Brocm6.4.0.gitd8831425-cp310-cp310-linux_x86_64.whl
#wget https://repo.radeon.com/rocm/manylinux/rocm-rel-6.4/pytorch_triton_rocm-3.2.0%2Brocm6.4.0.git6da9e660-cp310-cp310-linux_x86_64.whl
conda run -n rocm pip uninstall torch torchvision pytorch-triton-rocm pytorch_triton
#pip install --force-reinstall torch-2.6.0+rocm6.4.0.git2fb0ac2b-cp310-cp310-linux_x86_64.whl torchvision-0.21.0+rocm6.4.0.git4040d51f-cp310-cp310-linux_x86_64.whl torchaudio-2.6.0+rocm6.4.0.gitd8831425-cp310-cp310-linux_x86_64.whl pytorch_triton_rocm-3.2.0+rocm6.4.0.git6da9e660-cp310-cp310-linux_x86_64.whl
conda run -n rocm pip install torch==2.7.1+rocm6.2.4 torchvision==0.22.1+rocm6.2.4 torchaudio==2.7.1+rocm6.2.4 pytorch_triton==3.3.0 --extra-index-url https://download.pytorch.org/whl/
conda run -n rocm pip install sentencepiece transformers hf_xet accelerate diffusers protobuf xformers sacremoses
#
#rm -f torch-2.6.0+rocm6.4.0.git2fb0ac2b-cp310-cp310-linux_x86_64.whl
#rm -f torchvision-0.21.0+rocm6.4.0.git4040d51f-cp310-cp310-linux_x86_64.whl
#rm -f torchaudio-2.6.0+rocm6.4.0.gitd8831425-cp310-cp310-linux_x86_64.whl
#rm -f pytorch_triton_rocm-3.2.0+rocm6.4.0.git6da9e660-cp310-cp310-linux_x86_64.whl

# clear huggingface model cache
#rm -rf ~/.cache/huggingface/hub/