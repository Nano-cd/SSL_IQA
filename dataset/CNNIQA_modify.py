r"""CNNIQA Model.
Created by: https://github.com/lidq92/CNNIQA
Modified by: Chaofeng Chen (https://github.com/chaofengc)
Modification:
    - We use 3 channel RGB input.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNIQAnet(nn.Module):
    def __init__(self, ker_size=7, n_kers=50, n1_nodes=800, n2_nodes=800):
        super(CNNIQAnet, self).__init__()
        self.conv1  = nn.Conv2d(1, n_kers, ker_size)
        self.fc1    = nn.Linear(2 * n_kers, n1_nodes)
        self.fc2    = nn.Linear(n1_nodes, n2_nodes)
        self.fc3    = nn.Linear(n2_nodes, 5)
        self.dropout = nn.Dropout()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x  = x.view(-1, x.size(-3), x.size(-2), x.size(-1))  #

        h  = self.conv1(x)

        # h1 = F.adaptive_max_pool2d(h, 1)
        # h2 = -F.adaptive_max_pool2d(-h, 1)
        h1 = F.max_pool2d(h, (h.size(-2), h.size(-1)))
        h2 = -F.max_pool2d(-h, (h.size(-2), h.size(-1)))
        h  = torch.cat((h1, h2), 1)  # max-min pooling
        h  = h.squeeze(3).squeeze(2)

        h  = F.relu(self.fc1(h))
        h  = self.dropout(h)
        h  = F.relu(self.fc2(h))

        q  = self.fc3(h)
        q = self.softmax(q)
        return q