"""
-*-coding:utf-8-*-
train the image with the fc*2
"""
import torch 
import torch.nn as nn

from .resnet import resnet50


class ModelFCX2(nn.Module):
    def __init__(self, num_classes):
        super(ModelFCX2, self).__init__()
        self.num_classes = num_classes
        self.backbone = resnet50(pretrained=False)
        if True:
            model_state_dict = torch.load(
                        "/data/remote/code/classification_trick_with_model/models/weights/resnet50.pth", map_location="cpu")
            self.backbone.load_state_dict(model_state_dict)
            print("load the imagenet checkpoints!!!")
        in_feature = self.backbone.fc.in_features
        out_feature = in_feature * 2
        # self.fc_conv = torch.nn.Conv2d(in_feature, out_feature, 1, 1)
        self.fc_linear = nn.Linear(in_feature, out_feature)
        self.feature = torch.nn.Sequential(
            self.backbone.conv1,
            self.backbone.bn1,
            self.backbone.relu,
            self.backbone.maxpool,
            self.backbone.layer1,
            self.backbone.layer2,
            self.backbone.layer3,
            self.backbone.layer4
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classification = nn.Linear(out_feature, self.num_classes)
        # self.bn = nn.BatchNorm2d(out_feature)
        # self.relu2 = nn.ReLU()

    def forward(self, x):
        
        x = self.feature(x)
        x = self.avgpool(x) # bs,2048, 1, 1
        x = torch.flatten(x, 1)
        x = self.fc_linear(x)
        x = self.classification(x)
        return x 


if __name__ == "__main__":
    image = torch.rand(1, 3, 448, 448)
    model =  ModelFCX2(5000)
    output = model(image)
    print(output.shape)       
    print(model)

    


