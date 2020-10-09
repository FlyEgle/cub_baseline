# CUB_200_2011 FGVC baseline

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


|model|accuracy|
|:---:|:---:|
|resnet50(baseline)|84.12%|

### 训练
```bash train.sh```
其中，horovod可以指定几个node，对应的就是几块GPU

### 测试
```CUDA_VISIBLE_DEVICES=0 python test.py```