import torch
import torch.nn as nn
import torchvision.models as models


class GCPModel(nn.Module):

    def __init__(self):
        super().__init__()

        # pretrained ResNet18 backbone
        backbone = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        self.features = backbone.features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.coord_head = nn.Linear(1280, 2)
        self.shape_head = nn.Linear(1280, 3)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)

        coords = torch.sigmoid(self.coord_head(x))
        shape = self.shape_head(x)

        return coords, shape