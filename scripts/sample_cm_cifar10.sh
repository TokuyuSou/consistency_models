#!/bin/bash

# Image sampling script for MNIST EDM model
# This script matches the configuration from train_edm_mnist.sh

python image_sample.py \
  --batch_size 128 \
  --training_mode consistency_training \
  --sampler onestep \
  --model_path /workspace/logs/openai-2025-10-02-22-55-51-324873/model209000.pt \
  --attention_resolutions 16,8 \
  --class_cond True \
  --use_scale_shift_norm True \
  --dropout 0.0 \
  --image_size 32 \
  --num_channels 128 \
  --num_head_channels 32 \
  --num_res_blocks 2 \
  --num_samples 10000 \
  --resblock_updown True \
  --use_fp16 True \
  --weight_schedule karras
