#!/bin/bash
# Example usage: bash data/scripts/get_crowdpose.sh

# Make dataset directories
mkdir -p data/datasets/crowdpose

gdown -O data/datasets/crowdpose/images.zip https://drive.google.com/uc?id=1VprytECcLtU4tKP32SYi_7oDRbw7yUTL
gdown -O data/datasets/crowdpose/crowdpose_trainval.json https://drive.google.com/uc?id=13xScmTWqO6Y6m_CjiQ-23ptgX9sC-J9I
gdown -O data/datasets/crowdpose/crowdpose_test.json https://drive.google.com/uc?id=1FUzRj-dPbL1OyBwcIX2BgFPEaY5Yrz7S
unzip -q -d data/datasets/crowdpose data/datasets/crowdpose/images.zip
rm data/datasets/crowdpose/images.zip
