#!/bin/bash
export LD_LIBRARY_PATH=$(ls -d /mnt/c/123/train_env/lib/python3.*/site-packages/nvidia/*/lib | paste -sd ':' -):$LD_LIBRARY_PATH
/mnt/c/123/train_env/bin/python /mnt/c/123/test_gpu.py
