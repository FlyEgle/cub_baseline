#!/bin/bash
# cd /data/remote/yy_git_code/cub_baseline
# output_ckpt="/data/remote/output_ckpt_with_logs/accv/logs/101_no_repeat_clean"
# if [ ! -d "$output_ckpt" ]; then
#   mkdir "$output_ckpt"
# fi
# horovodrun -np 8 -H localhost:8 python -W ignore train.py \
# --batch-size=64 --num_thread=48 --fp16=1 --epochs=40 --warmup-epochs=1 --base-lr=0.0125 \
# --num_classes=5000 --model_name="regnet" --optimizer="sgd" --finetune=0 \
# --pretrainmodel="imagenet" --imagenet_pretrain=1 --model_freeze=0 \
# --log-dir="/data/remote/output_ckpt_with_logs/accv/logs/regnet-clean-224" \
# --checkpoint-format="$output_ckpt" \
# --train-dir="/data/remote/yy_git_code/cub_baseline/dataset/train_accv_clean_v1.log" \
# --val-dir="" --cutmix=1 --cosine_lr=0 --size_type="resnet50" --use_focalloss=0 --labelSmooth=1 \
# --resume_from_epoch=0
# horovodrun -np 4 -H localhost:4 python -W ignore train.py \
# --batch-size=32 --num_thread=32 --fp16=1 --epochs=40 --warmup-epochs=1 --base-lr=0.0125 \
# --num_classes=5000 --model_name="efficientnet-b5" --optimizer="sgd" --finetune=1 \
# --pretrainmodel="imagenet" --imagenet_pretrain=1 --model_freeze=0 \
# --log-dir="/data/remote/output_ckpt_with_logs/accv/logs/efnetb5-clean" \
# --checkpoint-format="$output_ckpt" \
# --train-dir="/data/remote/yy_git_code/cub_baseline/dataset/train_accv_clean_v1.log" \
# --val-dir="" --cutmix=1 --cosine_lr=1 --size_type="efnet-b5" --use_focalloss=0 --labelSmooth=1 \
# --resume_from_epoch=0

# resnest101
# cd /data/remote/yy_git_code/cub_baseline
# output_ckpt="/data/remote/output_ckpt_with_logs/accv/logs/101_no_repeat_clean"
# if [ ! -d "$output_ckpt" ]; then
#   mkdir "$output_ckpt"
# fi
# horovodrun -np 8 -H localhost:8 python -W ignore train.py \
# --batch-size=32 --num_thread=32 --fp16=1 --epochs=35 --warmup-epochs=0 --base-lr=0.0125 \
# --num_classes=5000 --model_name="resnest101" --optimizer="sgd" --finetune=0 \
# --pretrainmodel="imagenet" --imagenet_pretrain=1 --model_freeze=0 \
# --log-dir="/data/remote/output_ckpt_with_logs/accv/logs/101_no_repeat_clean" \
# --checkpoint-format="$output_ckpt" \
# --train-dir="/data/remote/yy_git_code/cub_baseline/dataset/train_accv_clean_v3_aliyun.log" \
# --val-dir="" --cutmix=1 --cosine_lr=0 --size_type="resnet50_448" --use_focalloss=1 --labelSmooth=1 \
# --resume_from_epoch=0


# /data/remote/yy_git_code/cub_baseline/dataset/train_accv_clean_v1_aliyun.log
# resnest200
cd /data/remote/yy_git_code/cub_baseline
output_ckpt="/data/remote/output_ckpt_with_logs/accv/ckpt/200_focalloss_freeze_repeat_data_512"
if [ ! -d "$output_ckpt" ]; then
  mkdir "$output_ckpt"
fi
horovodrun -np 8 -H localhost:8 python -W ignore train.py \
--batch-size=96 --num_thread=32 --fp16=1 --epochs=10 --warmup-epochs=0 --base-lr=0.0125 \
--num_classes=5000 --model_name="resnest200" --optimizer="sgd" --finetune=0 \
--pretrainmodel="/data/remote/output_ckpt_with_logs/accv/ckpt/freeze_fc/resnest200-448-freeze-fc-57.828.pth.tar" --imagenet_pretrain=1 --model_freeze=1 \
--log-dir="/data/remote/output_ckpt_with_logs/accv/logs/200_focalloss_freeze_repeat_data_512" \
--checkpoint-format="$output_ckpt" \
--train-dir="/data/remote/yy_git_code/cub_baseline/dataset/train_accv_clean_v3_with_repeat_plabel.log" \
--val-dir="" --cutmix=1 --cosine_lr=1 --size_type="resnet50_512" --use_focalloss=1 --labelSmooth=0 \
--resume_from_epoch=0 --wd=1e-4 --use_cbfocalloss=0 --use_ldamloss=0

# horovodrun -np 8 -H localhost:8 python -W ignore train.py \
# --batch-size=32 --num_thread=48 --fp16=1 --epochs=40 --warmup-epochs=1 --base-lr=0.0125 \
# --num_classes=5000 --model_name="efficientnet-b5" --optimizer="sgd" --finetune=0 \
# --pretrainmodel="imagenet" --imagenet_pretrain=1 --model_freeze=0 \
# --log-dir="/data/remote/output_ckpt_with_logs/accv/logs/efnetb5-clean" \
# --checkpoint-format="$output_ckpt" \
# --train-dir="/data/remote/yy_git_code/cub_baseline/dataset/train_accv_clean_v1.log" \
# --val-dir="" --cutmix=1 --cosine_lr=0 --size_type="efnet-b5" --use_focalloss=0 --labelSmooth=1 \
# --resume_from_epoch=0

# resnext
# horovodrun -np 8 -H localhost:8 python -W ignore train.py \
# --batch-size=24 --num_thread=48 --fp16=1 --epochs=40 --warmup-epochs=0 --base-lr=0.0125 \
# --num_classes=5000 --model_name="resnext" --optimizer="sgd" --finetune=0 \
# --pretrainmodel="imagenet" --imagenet_pretrain=1 --model_freeze=0 \
# --log-dir="/data/remote/output_ckpt_with_logs/accv/logs/resnext-clean" \
# --checkpoint-format="$output_ckpt" \
# --train-dir="/data/remote/yy_git_code/cub_baseline/dataset/train_accv_clean_v1_aliyun.log" \
# --val-dir="" --cutmix=1 --cosine_lr=0 --size_type="resnet50" --use_focalloss=0 --labelSmooth=0 \
# --resume_from_epoch=0

# inceptionv3
# horovodrun -np 8 -H localhost:8 python -W ignore train.py \
# --batch-size=128 --num_thread=48 --fp16=1 --epochs=90 --warmup-epochs=0 --base-lr=0.00125 \
# --num_classes=5000 --model_name="inceptionv3" --optimizer="sgd" --finetune=0 \
# --pretrainmodel="imagenet" --imagenet_pretrain=1 --model_freeze=0 \
# --log-dir="/data/remote/output_ckpt_with_logs/accv/logs/inceptionv3-pretrain-clean-299" \
# --checkpoint-format="$output_ckpt" \
# --train-dir="/data/remote/yy_git_code/cub_baseline/dataset/train_accv_clean_v1.log" \
# --val-dir="" --cutmix=1 --cosine_lr=0 --size_type="inceptionv3" --use_focalloss=0 --labelSmooth=0 \
# --resume_from_epoch=0