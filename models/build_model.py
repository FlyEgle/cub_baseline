"""
-*- coding:utf-8 -*-
resnet50 baseline
"""
import os
import torch
import torch.nn as nn

# resnet
from .resnet import resnet50, resnet101

# resnest
from .resnest.torch import resnest50_fast_4s2x40d

# efficientnet
from efficientnet_pytorch import EfficientNet

# regnet
from pycls.models import regnety

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
        



if __name__ == '__main__':
    # build_model = BuildModel("resnest50", 11, True)
    # model = build_model()
    # print(model)
    # model = EfficientNet.from_pretrained('efficientnet-b3')
    # print(model)
    build_model = BuildModel("regnet", 11, True)
    net = build_model()
    print(net)