#!/usr/bin/env bash

rm -rf venv
uv venv --python 3.10 venv
source venv/bin/activate

uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
uv pip install setuptools wheel
uv pip install --no-build-isolation -r requirements.txt
