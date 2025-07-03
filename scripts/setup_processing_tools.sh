#!/bin/bash

# Clone required third-party tools for data processing
set -e

if [ ! -d AlphaPose ]; then
    echo "Cloning AlphaPose ..."
    git clone --depth 1 https://github.com/MVIG-SJTU/AlphaPose.git
fi

if [ ! -d DensePose ]; then
    echo "Cloning DensePose ..."
    git clone --depth 1 https://github.com/facebookresearch/DensePose.git
fi

echo "✅ Tools downloaded"

