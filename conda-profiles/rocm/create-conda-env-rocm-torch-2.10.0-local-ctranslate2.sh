#!/bin/bash

ENV_NAME="rocm"
INSTALL_DIR=$(pwd)
cd "$INSTALL_DIR"

if conda info --envs | grep -q "^${ENV_NAME}\s"; then
    echo "Conda environment '${ENV_NAME}' already exists. Removing it."
    conda remove --name ${ENV_NAME} --all -y
fi

conda create --name ${ENV_NAME} python=3.14 -y

conda install -n ${ENV_NAME} cmake make gxx_linux-64 -y
conda install -n ${ENV_NAME} -c conda-forge onednn -y

TORCH_VERSION="2.10.0+rocm7.1"
TORCHVISION_VERSION="0.25.0+rocm7.1"
PYTORCH_TRITION_VERSION="3.5.1"

echo "Activating conda environment '${ENV_NAME}'..."
conda run -n ${ENV_NAME} pip install pip wheel --upgrade

echo "Installing PyTorch ${TORCH_VERSION} support..."
conda run -n ${ENV_NAME} pip install torch==${TORCH_VERSION} torchvision==${TORCHVISION_VERSION} torchaudio==${TORCH_VERSION} pytorch_triton_rocm==${PYTORCH_TRITION_VERSION} --extra-index-url https://download.pytorch.org/whl/
echo "Installing additional dependencies..."

conda run -n ${ENV_NAME} pip install sentencepiece transformers hf_xet accelerate diffusers protobuf xformers sacremoses peft scipy backoff peft datasets trl

echo "Installing bitsandbytes rocm version"
# sudo amdgpu-install # you need rocm 7.1+ version
# sudo apt-get install -y build-essential cmake rocm-dev rocm-libs
rm -rf bitsandbytes
git clone https://github.com/bitsandbytes-foundation/bitsandbytes.git && cd bitsandbytes/
cmake -DCOMPUTE_BACKEND=hip -S .
make
conda run -n ${ENV_NAME} pip install . --force-reinstall
cd ..
rm -rf bitsandbytes

echo "Installing ctranslate2 rocm version"
# sudo apt install cmake g++ libgoogle-glog-dev libboost-all-dev libdnnl-dev
# sudo apt install rocm-libs rocblas miopen-hip

rm -rf CTranslate2
git clone --recursive https://github.com/OpenNMT/CTranslate2.git
cd CTranslate2
mkdir build && cd build

cmake .. \
    -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
    -DCMAKE_PREFIX_PATH=$CONDA_PREFIX \
    -DDNNL_ROOT=$CONDA_PREFIX \
    -DWITH_MKL=OFF \
    -DWITH_DNNL=ON \
    -DWITH_OPENMP=ON \
    -DOPENMP_RUNTIME=COMP \
    -DWITH_CUDA=OFF \
    -DBUILD_CLI=OFF \
    -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX \
    -DBUILD_PYTHON_MODULE=ON

make -j$(nproc)
make install

cd ../python
conda run -n ${ENV_NAME} pip install . --force-reinstall

cd "$INSTALL_DIR"
rm -rf CTranslate2

# clear huggingface model cache
#rm -rf ~/.cache/huggingface/hub/

echo "Conda environment '${ENV_NAME}' with ${TORCH_VERSION} is set up successfully."
