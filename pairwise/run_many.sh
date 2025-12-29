#!/bin/bash

SCRIPT_DIR="/root/hot_detector/pairwise/configs_to_test"

for config in "$SCRIPT_DIR"/*.yaml; do
    echo ">>> Running: $config"
    python "/root/hot_detector/pairwise/train.py" "$config"
done
