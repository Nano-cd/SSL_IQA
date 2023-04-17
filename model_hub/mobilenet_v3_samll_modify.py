from torchvision.models import mobilenet_v3_small
import torch.nn as nn
import torch


class MVS(nn.Module):
    def __init__(self, num_classes):
        super(MVS, self).__init__()
        base_model = mobilenet_v3_small(pretrained=True)
        self.base = nn.Sequential(*list(base_model.children())[:-2])
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(576, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, num_classes),
            torch.nn.Softmax(dim=1)
        )

    def forward(self, x):
        features = self.base(x)
        out = self.avg_pool(features)
        out = torch.flatten(out, 1)
        activation = out
        out = self.fc(out)
        return out, activation
