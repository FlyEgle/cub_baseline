"""
-*- coding:utf-8 -*-
Bilinear CNN reference from https://github.com/HaoMood/bilinear-cnn.git
paper: http://vis-www.cs.umass.edu/bcnn/docs/bcnn_iccv15.pdf
"""
import torch 
import torch.nn as nn 

# use resnet50
from .resnet import resnet50


class BCNN_fc(nn.Module):
    def __init__(self, num_classes):
        super(BCNN_fc, self).__init__()
        self.num_classes = num_classes
        self.weights = "/data/remote/output_ckpt_with_logs/cub/ckpt/best_acc_08456.pth.tar"
        self.backbone = self._load_weights(self.weights)
        
        self.features = nn.Sequential(
            self.backbone.conv1,
            self.backbone.bn1,
            self.backbone.relu,
            self.backbone.maxpool,
            self.backbone.layer1,
            self.backbone.layer2,
            self.backbone.layer3,
            self.backbone.layer4
        )
        self.classification_linear = nn.Linear(2048*2048, self.num_classes)
        # freeze the feature layer parameters
        for param in self.features.parameters():
            param.requires_grad = False

        # init the classification layer
        nn.init.kaiming_normal_(self.classification_linear.weight.data)
        if self.classification_linear.bias is not None:
            nn.init.constant_(self.classification_linear.bias.data, val=0)

    def _load_weights(self, weights):
        self.net = resnet50(pretrained=False)
        self.net.avgpool = nn.AdaptiveAvgPool2d(1)
        self.net.fc = nn.Linear(self.net.fc.in_features, self.num_classes)
        state_dict = torch.load(self.weights, map_location="cpu")
        self.net.load_state_dict(state_dict["model"])
        return self.net 


    def forward(self, x):
        N = x.size()[0]
        assert x.size() == (N, 3, 448, 448)
        x = self.features(x)
        assert x.size() == (N, 2048, 14, 14)
        # print(x.shape)
        x = x.view(N, 2048, 14*14)
        x = torch.bmm(x, torch.transpose(x, 1, 2)) / (14*14)
        assert x.size() == (N, 2048, 2048)
        x = x.view(N, 2048*2048)
        x = torch.sqrt(x + 1e-7)
        x = torch.nn.functional.normalize(x)
        x = self.classification_linear(x)
        assert x.size() == (N, self.num_classes)
        return x 
        

if __name__ == "__main__":
    x = torch.rand(1, 3, 448, 448).cuda()
    bcnn_fc = BCNN_fc(200).cuda()
    output = bcnn_fc(x)
    print(output.shape)