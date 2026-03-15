#!/bin/bash
set -e

echo "Updating apt packages..."
apt-get update -y

echo "Installing system dependencies..."
apt-get install -y python3-pip python3-venv libgl1-mesa-glx libglib2.0-0

echo "Setting up Virtual Environment..."
if [ ! -d "/mnt/c/123/train_env" ]; then
    python3 -m venv /mnt/c/123/train_env
fi

source /mnt/c/123/train_env/bin/activate

echo "Installing Python Dependencies..."
pip install tensorflow[and-cuda] opencv-python pandas scikit-learn matplotlib
echo "Configuring LD_LIBRARY_PATH for CUDA..."
export LD_LIBRARY_PATH=$(ls -d /mnt/c/123/train_env/lib/python3.*/site-packages/nvidia/*/lib | paste -sd ':' -):$LD_LIBRARY_PATH

echo "Checking GPU Availability in TensorFlow..."
python3 -c "import tensorflow as tf; print('Num GPUs Available: ', len(tf.config.list_physical_devices('GPU')))"

echo "Starting Training..."
python3 /mnt/c/123/dcgan_train.py

echo "Done!"
