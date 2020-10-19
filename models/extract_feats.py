"""
-*- coding:utf-8 -*-
resnet50 baseline
"""
import os
import torch
import torch.nn as nn
from tqdm import tqdm
from efficientnet_pytorch import EfficientNet
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
import numpy as np
import urllib.request as urt
import torch.backends.cudnn as cudnn
from PIL import Image
from torchvision.transforms import transforms
from torch.utils.data import Dataset, DataLoader

global feat_result_efb5
feat_result_efb5 = []

class BuildModel(object):

    def __init__(self, net_name, num_classes, is_pretrained, map_location="cpu"):
        super(BuildModel, self).__init__()
        self.net_name = net_name
        self.num_classes = num_classes
        self.remove_aa_jit = False
        self.model_params = {
            "num_classes": 5000,
            "remove_aa_jit": self.remove_aa_jit
        }
        self.regnet_config = "12GF"

        if self.net_name == "resnet50":
            self.net = resnet50(pretrained=False)
            if is_pretrained:
                model_state_dict = torch.load(
                    "/data/remote/code/classification_trick_with_model/models/weights/resnet50.pth", 
                    map_location=map_location)
                self.net.load_state_dict(model_state_dict)
                print("Load the imagenet weights model!!!")

        elif self.net_name == "resnet50_4096":
            self.net = resnet50(pretrained=False)
            if is_pretrained:
                model_state_dict = torch.load(
                    "/data/remote/code/classification_trick_with_model/models/weights/resnet50.pth", 
                    map_location=map_location)
                self.net.load_state_dict(model_state_dict)
                print("Load the imagenet weights model!!!")

        elif self.net_name == "resnet101":
            self.net = resnet101(pretrained=False)
            if is_pretrained:
                model_state_dict = torch.load(
                    "/data/remote/code/classification_trick_with_model/models/weights/resnet101.pth",
                    map_location=map_location
                )
                self.net.load_state_dict(model_state_dict)
                print("Load the imagenet weights model!!!")


        elif self.net_name == "resnest50":
            self.net = resnest50_fast_4s2x40d(pretrained=False)
            if is_pretrained:
                model_state_dict = torch.load(
                    "/data/remote/code/classification_trick_with_model/models/weights/resnest50_fast_4s2x40d-41d14ed0.pth", 
                    map_location=map_location)
                self.net.load_state_dict(model_state_dict)
                print("Load the imagenet weights model!!!")

        # efficientnet 
        elif "efficientnet" in self.net_name:
            # self.net = EfficientNet.from_pretrained(self.net_name)
            self.net = EfficientNet.from_name(self.net_name)
            print("Load the imagenet pretrain model!!!efficientnet")

        # regnet
        elif "regnet" in self.net_name:
            self.net = regnety(self.regnet_config, pretrained=False)
            if is_pretrained:
                model_state_dict = torch.load(
                    "/data/remote/code/classification_trick_with_model/models/weights/regnet/RegNetY-{}_dds_8gpu.pyth".format(self.regnet_config),
                    map_location=map_location)
                self.net.load_state_dict(model_state_dict["model_state"])
                print("Load the imagenet weights model!!!")
        
    def __call__(self):

        if "resnet" in self.net_name or "resnest" in self.net_name:
            self.net.avgpool = nn.AdaptiveAvgPool2d(1)
            self.net.fc = nn.Linear(self.net.fc.in_features, self.num_classes)
            return self.net
        # if "resnet50_4096" == self.net_name:
            

        elif "efficientnet" in self.net_name:
            self.net._fc = nn.Linear(self.net._fc.in_features, self.num_classes)
            return self.net
        elif "regnet" in self.net_name:
            self.net.head.fc = nn.Linear(self.net.head.fc.in_features, self.num_classes)
            return self.net

def get_features_hook(self, input, output):
    # number of input:
    # print('len(input): ',len(input))
    # # number of output:
    # print('len(output): ',len(output))
    # print('###################################')
    # print(input[0].shape)
    # print('###################################')
    # print(output.shape)
    feat_result_efb5.append(output.data.cpu().numpy())

class TestDataSet(Dataset):

    def __init__(self):
        super(TestDataSet, self).__init__()
        self.test_file = "images_train.txt"
        # self.test_file = "/data/remote/code/classification_trick_with_model/data/val_imagenet_128w.txt"
        self.test_list = [(int(float(x.strip().split(' ')[0])), x.strip().split(' ')[1]) for x in open(self.test_file).readlines()]
        self.Resize_size = 456
        # self.input_size = 300
        self.imagenet_normalization_paramters = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

    def __getitem__(self, idx):
        lbl, url = self.test_list[idx][0], self.test_list[idx][1]
        # image = Image.open(BytesIO(urt.urlopen(url).read()))
        url = url.replace('\\',os.sep)
        image = Image.open(os.path.join("/home/ACCV_Datesets", url))
        if image.mode != "RGB":
            image = image.convert("RGB")
        image = transforms.Resize((self.Resize_size, self.Resize_size), Image.BILINEAR)(image)
        # image = transforms.CenterCrop(self.input_size)(image)
        image = transforms.ToTensor()(image)
        image = self.imagenet_normalization_paramters(image)
        return image, url

    def __len__(self):
        return len(self.test_list)


if __name__ == '__main__':
    build_model = BuildModel("efficientnet-b5", 5000, False)
    model = build_model()
    # print(model)
    model_state_dict = torch.load("efnetb5_456_05104.pth.tar", map_location = "cpu")
    model.load_state_dict(model_state_dict["model"])

    for name, m in model.named_modules():
        if isinstance(m, torch.nn.AdaptiveAvgPool2d):
            print(m)
            handle_feat = m.register_forward_hook(get_features_hook)  # conv1

    torch.backends.cudnn.enabled = False
    test_dataset = TestDataSet()
    testLoader = DataLoader(test_dataset, batch_size=72, num_workers=8)

    # modelp = torch.nn.DataParallel(model, device_ids=[0, 1])

    device = torch.device(
        "cuda:1" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model = model.eval()
    f = open('features_efnetb5.txt','a')
    feat_results = []

    with tqdm(total=len(testLoader), desc="processing predict logits") as t:
        with torch.no_grad():
            for idx, data in tqdm(enumerate(testLoader)):
                feat_result = []
                image_tensor, image_path = data[0], data[1]
                image_tensor = image_tensor.to(device)
                model(image_tensor)
                for i in range(len(image_path)):
                    result = {"image_path": image_path[i], "feats": feat_result[0][i].flatten().tolist()}
                    feats_string = ''
                    for ele in feat_result[0][i].flatten():
                        feats_string += ' '+str(ele)
                    f.write(image_path[i] + feats_string+'\n')
                    feat_results.append(result)

