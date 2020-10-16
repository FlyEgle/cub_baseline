#!/usr/bin/python3
# coding:utf-8
"""Imagenet dataset
"""
import warnings
warnings.filterwarnings("ignore")

import cv2
import os
import torch
import random
import time
import requests
import numpy as np
import urllib.request as ur
from urllib.error import URLError, HTTPError
import matplotlib.pyplot as plt

from tqdm import tqdm
from io import BytesIO
from PIL import Image
from torch.utils.data import Dataset, DataLoader, Sampler
from torchvision.transforms import transforms

from utils.autoaugment import ImageNetPolicy
from .config import ModelSize

np.random.seed(42)
random.seed(42)


# train dataset
class ImageNetTrainingDataset(Dataset):
    def __init__(self, image_file, autoaugment=False):
        super(ImageNetTrainingDataset, self).__init__()
        self.image_file = image_file
        # self.data = None
        with open(self.image_file, "r") as file:
            self.data = file.readlines()
        # shuffle the dataset
        for i in range(10):
            random.shuffle(self.data)
        self.imagenet_normalization_paramters = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        # 先resize到512 再crop到448
        # 用config来指定模型的size
        self.model_size = ModelSize("resnet50_448")
        model_size = self.model_size.imagesize_choice()
        self.BASE_RESIZE_SIZE = model_size["resize"]
        self.INPUT_SIZE = model_size["input"]
        self.BRIGHTNESS = 0.4
        self.HUE = 0.1
        self.CONTRAST = 0.4
        self.SATURATION = 0.4

        # autoaugment
        self.Autoaugment = autoaugment

        self.index_sampler = [i for i in range(len(self.data))]
        

        # 当前的数据增强【随机crop, 随机水平翻转，颜色变换，随机灰度，】
        if self.Autoaugment:
            self.image_transforms = transforms.Compose(
                [
                    transforms.Resize((self.BASE_RESIZE_SIZE, self.BASE_RESIZE_SIZE), Image.BILINEAR),
                    transforms.RandomCrop(self.INPUT_SIZE),
                    transforms.RandomHorizontalFlip(),
                    ImageNetPolicy(),
                    transforms.ToTensor(),
                    self.imagenet_normalization_paramters
                ]
            )
        else: 
            self.image_transforms = transforms.Compose(
                [
                    # transforms.RandomResizedCrop(self.INPUT_SIZE, scale=(0.2, 1.)),
                    transforms.Resize((self.BASE_RESIZE_SIZE, self.BASE_RESIZE_SIZE), Image.BILINEAR),
                    transforms.RandomCrop(self.INPUT_SIZE),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomRotation(degrees=15),
                    transforms.ColorJitter(brightness=self.BRIGHTNESS, contrast=self.CONTRAST, hue=self.HUE, saturation=self.SATURATION),
                    transforms.ToTensor(),
                    self.imagenet_normalization_paramters
                ]
            )

    def _get_image_label(self, text_lines):
        image_path = text_lines.split('\n')[0].split(',')[0]
        image_label = int(float(text_lines.split('\n')[0].split(',')[1]))
        return image_path, image_label

    def _resize_image(self, image, resize_size):
        """Resize form the [256 ... 448] random choice a size to crop
        """
        w, h = image.size
        scale = resize_size / float(min(h, w))
        resize_h, resize_w = int(h * scale), int(w * scale)
        image = image.resize((resize_w, resize_h), Image.BILINEAR)
        return image

    def __getitem__(self, idx):

        for i in range(10):
            image_url, image_lbl = self._get_image_label(self.data[idx])
            # print(image_url)
            assert type(image_lbl) == int, "the label type must be the int"
            try:
                with ur.urlopen(image_url, timeout=10.0) as file:
                    imagebyte = file.read()
                    image = Image.open(BytesIO(imagebyte))

                if image.mode != "RGB":
                    image = image.convert("RGB")
                
                image_tensor = self.image_transforms(image)

                label = torch.tensor(np.array(image_lbl)).long()
                return image_tensor, label

            except HTTPError as e:
            # do something
                print(image_url)
                print('Error code: ', e.code)
                idx = random.choice(self.index_sampler)
                print("random idx: ", idx)
                continue

            except URLError as e:
                print(image_url)
                print('Reason: ', e.reason)
                idx = random.choice(self.index_sampler)
                print("random idx: ", idx)
                continue

            except Exception as e:
                print(image_url, e)
                idx = random.choice(self.index_sampler)
                print("random idx: ", idx)
                continue

    def __len__(self):
        return len(self.data)


# validation dataset
class ImageNetValidationDataset(Dataset):
    def __init__(self, image_file):
        super(ImageNetValidationDataset, self).__init__()
        self.image_file = image_file
        self.data = None
        with open(self.image_file, "r") as file:
            self.data = file.readlines()
        self.imagenet_normalization_paramters = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        self.model_size = ModelSize("resnet50_448")
        model_size = self.model_size.imagesize_choice()
        self.RESIZE_SIZE = model_size["resize"]
        self.INPUT_SIZE = model_size["input"]
        self.index_sampler = [i for i in range(len(self.data))]

    def _get_image_label(self, text_lines):
        image_path = text_lines.split('\n')[0].split(',')[0]
        image_label = int(float(text_lines.split('\n')[0].split(',')[1]))
        return image_path, image_label

    def _resize_image(self, image, resize_size):
        w, h = image.size
        scale = resize_size / float(min(h, w))
        resize_h, resize_w = int(h * scale), int(w * scale)
        image = image.resize((resize_w, resize_h), Image.BILINEAR)
        return image

    def __getitem__(self, idx):
        # image_read_type = "pil"
        for i in range(10):
            try:
                image_url, image_lbl = self._get_image_label(self.data[idx])
                assert type(image_lbl) == int, "the label type must be the int"
                with ur.urlopen(image_url) as file:
                    # if image_read_type == "pil":
                    imagebyte = file.read()
                    image = Image.open(BytesIO(imagebyte))

                if image.mode != "RGB":
                    image = image.convert("RGB")
                # 保持长宽比进行resize
                # image = self._resize_image(image, self.RESIZE_SIZE)
                image_transforms = transforms.Compose(
                    [
                        transforms.Resize((self.RESIZE_SIZE, self.RESIZE_SIZE)),
                        transforms.CenterCrop((self.INPUT_SIZE, self.INPUT_SIZE)),
                        transforms.ToTensor(),
                        self.imagenet_normalization_paramters
                    ]
                )
                image_tensor = image_transforms(image)
                # translate to long tensor for multilabel crossentropy entropy
                
                label = torch.tensor(np.array(image_lbl)).long()
                return image_tensor, label
                
            except HTTPError as e:
            # do something
                print(image_url)
                print('Error code: ', e.code)
                idx = random.choice(self.index_sampler)
                print("random idx: ", idx)
                continue

            except URLError as e:
                print(image_url)
                print('Reason: ', e.reason)
                idx = random.choice(self.index_sampler)
                print("random idx: ", idx)
                continue

            except Exception as e:
                print(image_url, e)
                idx = random.choice(self.index_sampler)
                print("random idx: ", idx)
                continue


    def __len__(self):
        return len(self.data)


if __name__ == "__main__":
    import math
    import time
    from torch.utils.data import DataLoader
    train_file = "/data/remote/yy_git_code/cub_baseline/dataset/train_accv_shuf.txt"
    train_datset = ImageNetTrainingDataset(train_file)
    # for image, lbl in train_datset:
    #     print(image.shape, lbl.shape)
    train_dataloader = DataLoader(
        train_datset,
        batch_size=64,
        num_workers=32,
        shuffle=False,
        drop_last=False
    ) 
    print("data_loader")
    

    total_iter = int(len(train_datset) / 64)

    start_time = time.time()
    for idx, (image, target) in enumerate(train_dataloader):
        print("idx: {}/{}".format(idx, total_iter))
    print("waste time is ", time.time() - start_time)






