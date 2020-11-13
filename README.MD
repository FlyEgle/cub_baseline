# CUB_200_2011 FGCV baseline

# pretrain model
resnest50 -> 75.58%


数据增强方法:
```
transforms.Resize((self.BASE_RESIZE_SIZE, self.BASE_RESIZE_SIZE), Image.BILINEAR),
                transforms.RandomCrop(self.INPUT_SIZE),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(degrees=15),
                transforms.ColorJitter(brightness=self.BRIGHTNESS, contrast=self.CONTRAST, hue=self.HUE, saturation=self.SATURATION),
                transforms.ToTensor(),
                self.imagenet_normalization_paramters
```

学习率:
> 0.1初始化，不进行warmup 90个epoch，0.3, 0.6, 0.9 * epoch 按ratio=0.1进行线性衰减

### performance

|model|accuracy|
|:---:|:---:|
|resne50(baseline)256->224|76.47%|
|resnet50(baseline)512->448|84.55%|
|efnet-b0 256->224| 77.93%|
|efnet-b1 272->240| 79.63%|
|efnet-b2 292->260| 81.22%|
|efnet-b3 352->300| 83.12%|
|efnet-b4 416->380| 84.45%|
|efnet-b5 512->456| 85.64%|
|efnet-b6 600->528| 87.05%|
|efnet-b7 640->600| 88.11%|
<!-- |resnest50 256->224|| -->



### training
```bash train.sh```
其中，horovod可以指定几个node，对应的就是几块GPU

### test
```CUDA_VISIBLE_DEVICES=0 python test.py```


### pretrain download link

[cub_resnet50_08456](https://drive.google.com/file/d/1cl1GgHNaFzF9KNfGAkgaGveLinO-Yr9x/view?usp=sharing)



#### accvbaseline
|model|test acc|
|:---:|:---:|
|efnetb2 260| 41.975|
|resnet50 224|36.11|
|efnetb2+resnet50_224 ensemble1|43.105|
|efnetb3 300|	42.395|
|resnet50 448|	41.63|
|efnetb3+resnet50_448 ensemble2|	47.05|
|ensemble1+ensemble2|47.395|

- 20201016 update

|model|test acc|trick|
|:---:|:---:|:---:|
|efnetb5 456|51.04|cosine_lr+cutmix+40epoch+no_warmup+32bs+0.1lr|
|r50 448|47.64|cosine_lr+cutmix+40epoch+no_warmup+48bs+0.1lr|
|ensemble r50_efnetb5|52.408|same|


- 20201018 update
  
|model|test acc|trick|
|:---:|:---:|:---:|
|regnet12gf |53.578|cosine_lr+cutmix+40epoch+no_warmup+48bs+0.1lr|
|regnet12gf + efnetb5 |55.758|same|


- 20201026 update


|model|test acc|trick|
|:---:|:---:|:---:|
|resnest200 |56.145|cosine_lr+cutmix+60epoch+no_warmup+24bs+0.1lr+ls|
|resnest200 + regnet12gf |58.35%|same|
|resnest200 + regnet12gf + efnetb5 |58.558%|same|


- 20201111 update 

|model|test acc|trick|
|:---:|:---:|:---:|
|resnest101 + resnest200 + resnest 269 |60.135 |data_clean_v3.log + freeze_fc + focalloss + 40 epoch + cutmix|
|resnest200 data_v1 + resnest200 data_v2 + resnest200 data_v3 | 60.808| freeze_fc + focalloss + 40 epoch + cutmix |
|ensemble|61.548|mean|


# 数据记录

- /data/remote/yy_git_code/cub_baseline/dataset/train_accv_clean_v2.log 使用56.145模型进行数据清洗
- /data/remote/yy_git_code/cub_baseline/dataset/train_accv_clean_v3.log 对重复标签的数据进行清洗
- /data/remote/yy_git_code/cub_baseline/dataset/train_accv_clean_v3_remove_small.log remove小于48尺寸的图片
- /data/remote/yy_git_code/cub_baseline/dataset/train_accv_clean_v3_with_repeat_plabel.log 使用resnest系列的3个模型对有二义性的数据生成伪标签，填充到cleanv3里面进行训练。