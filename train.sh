#!/bin/bash
cd /data/remote/yy_git_code/cub_baseline
output_ckpt="/data/remote/output_ckpt_with_logs/cub/ckpt/fix_resize_512_crop_448_lr_01_150_epoch"
if [ ! -d "$output_ckpt" ]; then
  mkdir "$output_ckpt"
fi
horovodrun -np 8 -H localhost:8 python -W ignore train.py \
--batch-size=64 --num_thread=32 --fp16=0 --epochs=150 --warmup-epochs=0 --base-lr=0.0125 \
--num_classes=200 --model_name="resnet50" --optimizer="sgd" --finetune=0 \
--pretrainmodel="imagenet" --imagenet_pretrain=1  --model_freeze=0 \
--train-dir="/data/remote/yy_git_code/cub_baseline/dataset/cub_train.txt" \
--val-dir="/data/remote/yy_git_code/cub_baseline/dataset/cub_test.txt" \
--log-dir="/data/remote/output_ckpt_with_logs/cub/logs/fix_resize_512_crop_448_lr_01_150_epoch" \
--checkpoint-format="$output_ckpt"