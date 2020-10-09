# -*-coding:utf-8-*-
import os
import torch
import torch.nn as nn

from .resnet import resnet50

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
        if self.net_name == "resnet50":
            self.net = resnet50(pretrained=is_pretrained)
            if is_pretrained:
                model_state_dict = torch.load(
                    "/data/remote/code/classification_trick_with_model/models/weights/resnet50.pth", 
                    map_location=map_location)
                self.net.load_state_dict(model_state_dict)
                print("Load the imagenet weights model!!!")

    def __call__(self):

        if self.net_name == "resnet50":
            self.net.avgpool = nn.AdaptiveAvgPool2d(1)
            self.net.fc = nn.Linear(self.net.fc.in_features, self.num_classes)
            return self.net

if __name__ == '__main__':
    build_model = BuildModel("resnest50_mutil_task", 11, True)
    # model = build_model().cuda()
    # summary(model, input_size=(3, 224, 224))
    print(build_model)