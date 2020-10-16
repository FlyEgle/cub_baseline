#!/bin/bash
cd /data/remote/yy_git_code/cub_baseline
output_ckpt="/data/remote/output_ckpt_with_logs/cub/ckpt/resnet50_448_lr_01_90_epoch"
if [ ! -d "$output_ckpt" ]; then
  mkdir "$output_ckpt"
fi
horovodrun -np 8 -H localhost:8 python -W ignore train.py \
--batch-size=16 --num_thread=32 --fp16=1 --epochs=90 --warmup-epochs=0 --base-lr=0.0125 \
--num_classes=200 --model_name="resnet50" --optimizer="sgd" --finetune=0 \
--pretrainmodel="imagenet" --imagenet_pretrain=1 --model_freeze=0 \
--log-dir="/data/remote/output_ckpt_with_logs/cub/logs/resnet50_448_lr_01_90_epoch" \
--checkpoint-format="$output_ckpt" \
--train-dir="/data/remote/yy_git_code/cub_baseline/dataset/cub_train.txt" \
--val-dir="/data/remote/yy_git_code/cub_baseline/dataset/cub_test.txt" --cutmix=0 --cosine_lr=0