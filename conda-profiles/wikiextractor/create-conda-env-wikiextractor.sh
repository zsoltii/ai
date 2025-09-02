#!/bin/bash

conda create --name wikiextractor python=3.10 -y
conda run -n wikiextractor pip install pip wheel --upgrade

conda run -n wikiextractor pip install wikiextractor
