"""
-*- coding:utf-8 -*-
resnet50 baseline
"""
import os
import torch
import torch.nn as nn

# resnet
from .resnet import resnet50, resnet101, resnet152

# resnest
from .resnest.torch import resnest50_fast_4s2x40d, resnest200, resnest101, resnest269

# efficientnet
from efficientnet_pytorch import EfficientNet

# regnet
from pycls.models import regnety, resnext

# resnext

# inceptionv3
from .inceptionv3 import inception_v3


class BuildModel(object):

    def __init__(self, net_name, num_classes, is_pretrained, map_location="cpu"):
        super(BuildModel, self).__init__()
        self.net_name = net_name
        self.num_classes = num_classes
        self.remove_aa_jit = False
        self.model_params = {
            "num_classes": 1000,
            "remove_aa_jit": self.remove_aa_jit
        }
        self.regnet_config = "1.6GF"
        self.resnext_config = "152"
        self.use_maxpool = False

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

        elif self.net_name == "resnest200":
            print("use the resnest200 model!!!")
            self.net = resnest200(pretrained=False)
            if is_pretrained:
                model_state_dict = torch.load(
                    "/data/remote/code/classification_trick_with_model/models/weights/resnest200-75117900.pth",
                    map_location=map_location
                )
                self.net.load_state_dict(model_state_dict)
                print("Load the resnest200 imagenet weights model!!!")

        elif self.net_name == "resnest101":
            print("use the resnest101 model!!!")
            self.net = resnest101(pretrained=False)
            if is_pretrained:
                model_state_dict = torch.load(
                    "/data/remote/code/classification_trick_with_model/models/weights/resnest101-22405ba7.pth",
                    map_location=map_location
                )
                self.net.load_state_dict(model_state_dict)
                print("Load the resnest101 imagenet weights model!!!")

        elif self.net_name == "resnest269":
            print("use the resnest269 model!!!")
            self.net = resnest269(pretrained=False)
            if is_pretrained:
                model_state_dict = torch.load(
                    "/data/remote/code/classification_trick_with_model/models/weights/resnest269-0cc87c48.pth",
                    map_location=map_location
                )
                self.net.load_state_dict(model_state_dict)
                print("Load the resnest101 imagenet weights model!!!")

        # efficientnet 
        elif "efficientnet" in self.net_name:
            self.net = EfficientNet.from_pretrained(self.net_name)
            print("Load the imagenet pretrain model!!!efficientnet")

        # regnet
        elif "regnet" in self.net_name:
            self.net = regnety(self.regnet_config, pretrained=False)
            if is_pretrained:
                model_state_dict = torch.load(
                    "/data/remote/code/classification_trick_with_model/models/weights/regnet/RegNetY-{}_dds_8gpu.pyth".format(self.regnet_config),
                    map_location=map_location)
                self.net.load_state_dict(model_state_dict["model_state"])
                print("Load the imagenet regnet_{} weights model!!!".format(self.regnet_config))

        elif "resnext" in self.net_name:
            self.net = torch.hub.load('facebookresearch/WSL-Images', 'resnext101_32x32d_wsl')
            # print(self.net)
            print("Load the resnext101_32x32d_wsl imagenet weights!")

        elif "inceptionv3" in self.net_name:
            self.net = inception_v3(pretrained=False)
            self.net.fc = nn.Linear(2048, 8142)
            if is_pretrained:
                model_state_dict = torch.load(
                    "/data/remote/code/classification_trick_with_model/models/weights/iNat_2018_InceptionV3.pth.tar", 
                    map_location=map_location
                )
                self.net.load_state_dict(model_state_dict["state_dict"])
                print("Load the imagenet weights model!!!")
            
        
    def __call__(self):

        if "resnet" in self.net_name or "resnest" in self.net_name:
            self.net.avgpool = nn.AdaptiveAvgPool2d(1)
            self.net.fc = nn.Linear(self.net.fc.in_features, self.num_classes)
            return self.net

        elif "efficientnet" in self.net_name:
            self.net._fc = nn.Linear(self.net._fc.in_features, self.num_classes)
            return self.net
        elif "regnet" in self.net_name:
            self.net.head.fc = nn.Linear(self.net.head.fc.in_features, self.num_classes)
            return self.net

        elif "resnext" in self.net_name:
            self.net.fc = nn.Linear(self.net.fc.in_features, self.num_classes)
            return self.net

        elif "inceptionv3" in self.net_name:
            self.net.fc = nn.Linear(self.net.fc.in_features, self.num_classes)
            self.net.aux_logits = False
            return self.net
        


if __name__ == '__main__':
    build_model = BuildModel("regnet", 5000, True)
    model = build_model()
    print(model)
    # model = EfficientNet.from_pretrained('efficientnet-b3')
    # print(model)
    # build_model = BuildModel("regnet", 11, True)
    # net = build_model()
    # print(net)
