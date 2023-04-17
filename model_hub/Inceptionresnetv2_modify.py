from model_hub import inceptionresnetv2
import torch.nn as nn
import torch


class Inceptionresnet_v2(nn.Module):
    def __init__(self, num_classes):
        super(Inceptionresnet_v2, self).__init__()
        base_model = inceptionresnetv2.inceptionresnetv2(num_classes=5, pretrained='imagenet')
        self.base = nn.Sequential(*list(base_model.children())[:-2])
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.drop_out = nn.Dropout()
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(1536, 512),
            torch.nn.Linear(512, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, num_classes),
            torch.nn.Softmax(dim=1)
        )

    def forward(self, x):
        features = self.base(x)
        out = self.avg_pool(features)
        out = self.drop_out(out)
        out = torch.flatten(out, 1)
        activation = out
        out = self.fc(out)
        return out, activation
