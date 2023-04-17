import torch
import torchvision
import torch.nn as nn


class ResNet50(torch.nn.Module):
    """Modified ResNet50 for feature extraction"""
    def __init__(self):
        super(ResNet50, self).__init__()
        model = torchvision.models.resnet50(pretrained=True)
        self.features = nn.Sequential(*list(model.children())[:-1])

    def forward(self, x):
        # features@: pool5
        x = self.features(x)

        return x.view(x.size(0), -1)