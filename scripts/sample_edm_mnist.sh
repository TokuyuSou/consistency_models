#!/bin/bash

# Image sampling script for MNIST EDM model
# This script matches the configuration from train_edm_mnist.sh

python image_sample.py \
  --training_mode edm \
  --batch_size 64 \
  --sigma_max 80 \
  --sigma_min 0.002 \
  --s_churn 0 \
  --steps 40 \
  --sampler heun \
  --model_path /workspace/logs/openai-2025-09-02-12-50-44-245996/model049000.pt \
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
  --weight_schedule karras \
  --seed 42 \
  --generator determ