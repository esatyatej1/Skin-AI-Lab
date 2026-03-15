export TF_CPP_MIN_LOG_LEVEL=0
export LD_LIBRARY_PATH=$(ls -d /mnt/c/123/train_env/lib/python3.*/site-packages/nvidia/*/lib | paste -sd ':' -):$LD_LIBRARY_PATH
/mnt/c/123/train_env/bin/python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

