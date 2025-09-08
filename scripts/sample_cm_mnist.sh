#!/bin/bash

# Image sampling script for MNIST EDM model
# This script matches the configuration from train_edm_mnist.sh

python image_sample.py \
  --training_mode consistency_distillation \
  --batch_size 64 \
  --sampler onestep \
  --model_path /workspace/logs/openai-2025-09-02-23-24-16-529174/model028000.pt \
  --attention_resolutions 16,8 \
  --class_cond False \
  --dropout 0.0 \
  --image_size 32 \
  --num_channels 128 \
  --num_head_channels 32 \
  --num_res_blocks 2 \
  --num_samples 10000 \
  --resblock_updown True \
  --use_fp16 True \
  --use_scale_shift_norm True \
  --weight_schedule uniform \
  --seed 42 \
  --generator determ