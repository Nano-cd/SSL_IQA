import argparse
import math
import os

from tqdm import tqdm
import numpy as np
from torch.utils.tensorboard import SummaryWriter as sum_writer
import torch.autograd
from torch.autograd import Variable
import torch.backends.cudnn

import config_hub
import creator
from scipy.signal import convolve2d
import torch.nn as nn
from utils import compute_metric

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
        self.fc3 = nn.Linear(512, num_classes)
        self.sig = nn.Sigmoid()

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
    def __init__(self, resnet, net):
        super(Net, self).__init__()
        self.resnet_layer = resnet
        self.net = net

    def forward(self, x):
        x = self.resnet_layer(x)
        x = self.net(x)

        return x


class Trainer(object):
    def __init__(self, config):
        torch.manual_seed(config.seed)
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.test_dataloader = creator.datasets_creator(self.config)
        self.model = creator.model_creator(config)
        self.model.load_state_dict(torch.load(config.ckpt_path + '/NetBL_2K-00100.pt')['state_dict'])
        self.model.to(self.device)
        if torch.cuda.device_count() > 1:
            self.model = torch.nn.DataParallel(self.model, device_ids=[0, 1])
        self.model_name = type(self.model).__name__ + self.config.train_description

        self.ckpt_path = config.ckpt_path
        self.start_epoch = 0
        self.max_epoch = self.config.max_epoch
        self.global_step = 1
        self.loss_fn, self.optimizer = creator.optimal_creator(self.config, self.model)
        self.loss_fn.to(self.device)


    def _test(self):
        y_ = []
        y_pred = []
        self.model.eval()
        with torch.no_grad():
            for _, (x, y) in enumerate(self.test_dataloader):
                x = x.cuda()
                y = y.cuda()
                y = torch.reshape(Variable(y), [-1, 5])
                outputs = self.model(x)
                sum_index = torch.tensor([1, 2, 3, 4, 5], requires_grad=False, device=self.device,
                                         dtype=torch.float).unsqueeze(1)
                outputs = torch.mm(outputs, sum_index).squeeze(dim=1)
                labels = torch.mm(y, sum_index).squeeze(dim=1)
                y_.append(labels.cpu().numpy())
                y_pred.append(outputs.cpu().numpy())

            RMSE, PLCC, SROCC, KROCC = compute_metric(np.array(y_).flatten(), np.array(y_pred).flatten())
            return PLCC, SROCC


def main(cfg):
    t = Trainer(cfg)
    plcc_, srocc_ = t._test()
    print(plcc_, srocc_)


if __name__ == "__main__":
    config = config_hub.Metal_config()
    for i in range(0, 10):
        config = config_hub.Metal_config()
        split = i + 1
        config.split = split
        config.ckpt_path = os.path.join(config.ckpt_path, str(config.split))
        if not os.path.exists(config.ckpt_path):
            os.makedirs(config.ckpt_path)
        print(config.network, config.train_description)
        main(config)
