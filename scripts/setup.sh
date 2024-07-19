#!/bin/bash

set -euo pipefail

# extract models_smplx_v1_1.zip into data/body_models
unzip -o models_smplx_v1_1.zip -d data/body_models
mv data/body_models/models/smplx data/body_models/
rm -rf data/body_models/models


# Define the virtual environment directory
VENV_DIR="wandr_env"

# Create a Python virtual environment named wandr_test using Python 3.10
python3.10 -m venv $VENV_DIR

# Activate the virtual environment
. $VENV_DIR/bin/activate

# Install PyTorch
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install packages from requirements.txt
pip install -r requirements.txt


echo "Setup complete."
