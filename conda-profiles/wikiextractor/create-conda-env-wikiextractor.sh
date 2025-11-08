#!/bin/bash

ENVIRONMENT_NAME="wikiextractor"

if conda info --envs | grep -q "^${ENVIRONMENT_NAME}\s"; then
    echo "Conda environment '${ENVIRONMENT_NAME}' already exists. Removing it."
    conda remove --name ${ENVIRONMENT_NAME} --all -y
fi

conda create --name ${ENVIRONMENT_NAME} python=3.10 -y
conda run -n ${ENVIRONMENT_NAME} pip install pip wheel --upgrade

conda run -n ${ENVIRONMENT_NAME} pip install wikiextractor
