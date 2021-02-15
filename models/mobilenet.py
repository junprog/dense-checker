import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn import functional as F

class MobileNet(nn.Module):
    def __init__(self, features):
        super(MobileNet, self).__init__()
        self.features = features
        self.reg_layer = nn.Sequential(
            nn.Conv2d(1280, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 1, 1)
        )

    def forward(self, x):
        x = self.features(x)
        x = F.interpolate(x, scale_factor=4, mode='bilinear', align_corners=False)
        x = self.reg_layer(x)
        return torch.abs(x)

def mobilenet_v2():
    model = MobileNet(models.mobilenet_v2(pretrained=False).features)
    return model