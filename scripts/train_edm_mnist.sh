# 単GPUなら mpiexec は不要（そのまま python ...）
python edm_train.py \
  --class_cond False \
  --attention_resolutions 16,8 \
  --use_scale_shift_norm True \
  --dropout 0.0 \
  --ema_rate 0.9999 \
  --global_batch_size 256 \
  --image_size 32 \
  --lr 1e-4 \
  --num_channels 128 \
  --num_head_channels 32 \
  --num_res_blocks 2 \
  --resblock_updown True \
  --schedule_sampler lognormal \
  --use_fp16 True \
  --weight_decay 0.0 \
  --weight_schedule karras \
  --data_dir /workspace/consistency_models/datasets/celeba32
