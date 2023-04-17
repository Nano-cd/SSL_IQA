from torchvision.models import mobilenet_v2
import torch.nn as nn
import torch




class Mobilenet_v2(nn.Module):
    def __init__(self, num_classes):
        super(Mobilenet_v2, self).__init__()
        base_model = mobilenet_v2(pretrained=True)
        self.base = nn.Sequential(*list(base_model.children())[:-1])
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(1280, 512),
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
