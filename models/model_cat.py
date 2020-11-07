"""Build the model concat Use the backbone is r50"""

import torch 
import torch.nn as nn 
import torch.nn.functional as F

from pycls.models import resnet


class ModelConcat(nn.Module):
    def __init__(self, num_classes, style="concate"):
        super(ModelConcat, self).__init__()
        self.style = style
        self.backbone = resnet("50", pretrained=False)
        self.checkpoints = torch.load("/data/remote/code/classification_trick_with_model/models/weights/R-50-1x64d_dds_8gpu.pyth", map_location="cpu")
        self.backbone.load_state_dict(self.checkpoints["model_state"])
        self.net = nn.Sequential(
            self.backbone.stem,
            self.backbone.s1,
            self.backbone.s2,
            self.backbone.s3,
            self.backbone.s4
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        if self.style == "concate":
            self.fc = nn.Linear(self.backbone.head.fc.in_features*2, num_classes)
        else:
            self.fc = nn.Linear(self.backbone.head.fc.in_features. num_classes)

    def forward(self, data1, data2):
        feature1 = self.net(data1)
        feature2 = self.net(data2)
        if self.style == "concate":
            assert feature1.shape == feature2.shape
            feature = torch.cat([feature1, feature2], dim=1)
        else:
            assert feature1.shape == feature2.shape 
            feature = feature1 + feature2
        x = self.avgpool(feature)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x 
            

if __name__ == "__main__":
    net = ModelConcat(num_classes=5000)
    a = torch.rand(12, 3, 224, 224)
    b = torch.rand(12, 3, 224, 224)
    output = net(a, b)
    print(output)