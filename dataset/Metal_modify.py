import torch
import torch.nn as nn
import math

import torchvision.models


class BaselineModel1(nn.Module):
    def __init__(self, num_classes, keep_probability, inputsize):

        super(BaselineModel1, self).__init__()
        self.fc1 = nn.Linear(inputsize, 1024)
        self.bn1 = nn.BatchNorm1d(1024)
        self.drop_prob = (1 - keep_probability)
        self.relu1 = nn.PReLU()
        self.drop1 = nn.Dropout(self.drop_prob)
        self.fc2 = nn.Linear(1024, 512)
        self.bn2 = nn.BatchNorm1d(512)
        self.relu2 = nn.PReLU()
        self.drop2 = nn.Dropout(p=self.drop_prob)
        self.fc3 = nn.Linear(512, 5)
        self.sig = nn.Softmax(dim=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Weight initialization reference: https://arxiv.org/abs/1502.01852
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            # elif isinstance(m, nn.Linear):
            #     m.weight.data.normal_(0, 0.02)
            #     m.bias.data.zero_()

    def forward(self, x):
        """
        Feed-forward pass.
        :param x: Input tensor
        : return: Output tensor
        """
        out = self.fc1(x)

        out = self.bn1(out)
        out = self.relu1(out)
        out = self.drop1(out)
        out = self.fc2(out)

        out = self.bn2(out)
        out = self.relu2(out)
        out = self.drop2(out)
        out = self.fc3(out)
        out = self.sig(out)
        # out_a = torch.cat((out_a, out_p), 1)

        # out_a = self.sig(out)
        return out


class Net(nn.Module):
    def __init__(self, resnet=torchvision.models.resnet18(), net = BaselineModel1(5,0.5,1000)):
        super(Net, self).__init__()
        self.resnet_layer = resnet
        self.net = net

    def forward(self, x):
        x = self.resnet_layer(x)
        x = self.net(x)

        return x





