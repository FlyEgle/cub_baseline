import os 
import torch 
import torch.nn as nn 

from build_model import BuildModel


resnet50_model = BuildModel("resnet50", num_classes=5000, is_pretrained=False)()
efnetb5_model = BuildModel("efficientnet-b5", num_classes=5000, is_pretrained=False)()

# r50 feature model
class ResFeatureModel(nn.Module):
    def __init__(self):
        super(ResFeatureModel, self).__init__()
        # load model
        state_dict = torch.load("/data/remote/output_ckpt_with_logs/accv/ckpt/r50_448_baseline_04764.pth.tar", map_location="cpu")
        self.backbone = resnet50_model
        self.backbone.load_state_dict(state_dict["model"])
        
        self.feature = nn.Sequential(
            self.backbone.conv1,
            self.backbone.bn1,
            self.backbone.relu,
            self.backbone.maxpool,
            self.backbone.layer1,
            self.backbone.layer2,
            self.backbone.layer3,
            self.backbone.layer4,
            self.backbone.avgpool
        )

    def forward(self, x):
        feature = self.feature(x)
        return feature


# efficient feature model
class efnetb5_modelFeatureModel(nn.Module):
    def __init__(self):
        super(efnetb5_modelFeatureModel, self).__init__()
        # load model
        state_dict = torch.load("/data/remote/output_ckpt_with_logs/accv/ckpt/efnetb5_456_05104.pth.tar", map_location="cpu")
        self.backbone = efnetb5_model
        self.backbone.load_state_dict(state_dict["model"])

    def forward(self, x):
        x = self.backbone._conv_stem(x)
        x = self.backbone._bn0(x)
        for m in self.backbone._blocks:
            x = m(x)
        x = self.backbone._conv_head(x)
        x = self.backbone._bn1(x)
        feature = self.backbone._avg_pooling(x)
        return feature


if __name__ == "__main__":
    x = torch.rand(1, 3, 456, 456)
    net = efnetb5_modelFeatureModel()
    # print(net)
    output = net(x)
    print(output.shape)