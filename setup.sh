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
echo "Configuring environment variables for GPU..."
export LD_LIBRARY_PATH=$(ls -d /mnt/c/123/train_env/lib/python3.*/site-packages/nvidia/*/lib | paste -sd ':' -):$LD_LIBRARY_PATH
export TF_CPP_MIN_LOG_LEVEL=1

echo "Checking GPU Availability in TensorFlow..."
python3 -c "import tensorflow as tf; print('Found GPUs: ', tf.config.list_physical_devices('GPU'))"

echo "--------------------------------------------------------"
echo "NOTE: Since you have a new RTX 5070, TensorFlow might"
echo "take a few minutes to 'JIT compile' kernels on the first run."
echo "If it seems stuck, please wait 2-5 minutes."
echo "--------------------------------------------------------"

MODE=$1
if [ -z "$MODE" ]; then
    MODE="normal"
fi

echo "Starting Training in $MODE mode..."
python3 /mnt/c/123/dcgan_train.py --mode $MODE

echo "Done!"
