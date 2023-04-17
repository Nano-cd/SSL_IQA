from torchvision.models import resnet18
import torch.nn as nn
import torch




class Resnet_18(nn.Module):
    def __init__(self, num_classes):
        super(Resnet_18, self).__init__()
        base_model = resnet18(pretrained=True)
        self.base = nn.Sequential(*list(base_model.children())[:-2])
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(512, 256),
            torch.nn.BatchNorm1d(256),
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
