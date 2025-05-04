#!/bin/bash

conda create --name m2m100 python=3.10
conda activate m2m100
pip install pip wheel --upgrade
pip install sentencepiece transformers torch hf_xet