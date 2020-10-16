#!/bin/bash
cd /data/remote/yy_git_code/cub_baseline
output_ckpt="/data/remote/output_ckpt_with_logs/accv/ckpt/resnet50_448_48_lr_01_40_epoch_cutmix_cosinelr"
if [ ! -d "$output_ckpt" ]; then
  mkdir "$output_ckpt"
fi
horovodrun -np 8 -H localhost:8 python -W ignore train.py \
--batch-size=48 --num_thread=32 --fp16=1 --epochs=40 --warmup-epochs=0 --base-lr=0.0125 \
--num_classes=5000 --model_name="resnet50" --optimizer="sgd" --finetune=0 \
--pretrainmodel="imagenet" --imagenet_pretrain=1 --model_freeze=0 \
--log-dir="/data/remote/output_ckpt_with_logs/accv/logs/resnet50_448_48_lr_01_40_epoch_cutmix_cosinelr" \
--checkpoint-format="$output_ckpt" \
--train-dir="/data/remote/yy_git_code/cub_baseline/dataset/train_accv_shuf_pingtai.txt" \
--val-dir="" --cutmix=1 --cosine_lr=1
# cub
# --train-dir="/data/remote/yy_git_code/cub_baseline/dataset/cub_train.txt" \
# --val-dir="/data/remote/yy_git_code/cub_baseline/dataset/cub_test.txt" \

# train-dir="/data/remote/yy_git_code/cub_baseline/dataset/train_accv_shuf_pingtai.txt"gpugpu