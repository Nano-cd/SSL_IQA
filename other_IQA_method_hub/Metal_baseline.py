import argparse
import math
import os
import torch.nn as nn
from tqdm import tqdm
import numpy as np
# from torch.utils.tensorboard import SummaryWriter as sum_writer
import torch.autograd
from torch.autograd import Variable

import config_hub
import creator

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def exp_lr_scheduler(optimizer, epoch, lr_decay_epoch=10):
    """Decay learning rate by a factor of DECAY_WEIGHT every lr_decay_epoch epochs."""

    decay_rate = 0.8 ** (epoch // lr_decay_epoch)
    if epoch % lr_decay_epoch == 0:
        print('decay_rate is set to {}'.format(decay_rate))

    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * decay_rate

    return optimizer


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
    def __init__(self, config, model):
        torch.manual_seed(config.seed)
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.train_dataloader = creator.datasets_creator(self.config)
        self.model = model
        self.model = self.model.to(self.device)

        if torch.cuda.device_count() > 1:
            self.model = torch.nn.DataParallel(self.model, device_ids=[0, 1])
        self.model_name = type(self.model).__name__ + self.config.train_description

        self.ckpt_path = config.ckpt_path
        self.start_epoch = 0
        self.max_epoch = self.config.max_epoch
        self.global_step = 1
        runs_path = os.path.join(self.config.tensorboard_path, self.model_name + str(self.config.split))
        # self.logger = sum_writer(runs_path)
        self.loss_fn, self.optimizer = creator.optimal_creator(self.config, self.model)
        self.loss_fn.cuda()

    def fit(self):
        for epoch in tqdm(range(self.start_epoch, self.max_epoch)):
            self._train_one_epoch(epoch)
            self.optimizer = exp_lr_scheduler(optimizer=self.optimizer, epoch=epoch)
            if epoch + 1 == self.max_epoch:
                model_name = '{}-{:0>5d}.pt'.format(self.model_name, epoch + 1)
                model_name = os.path.join(self.ckpt_path, model_name)
                self._save_checkpoint({
                    'epoch': epoch,
                    'state_dict': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                }, model_name)

    def _train_one_epoch(self, epoch):
        # start training
        self.model.train()
        for _, (x, y) in enumerate(self.train_dataloader):
            x = Variable(x)
            x = x.cuda()
            y = y.cuda()
            self.optimizer.zero_grad()
            predict_student = self.model(x)
            self.loss = self.loss_fn(predict_student, y.float().detach())
            self.loss.backward()
            self.optimizer.step()

            # self.logger.add_scalar(tag='sum_loss',
            #                        scalar_value=self.loss.item(),
            #                        global_step=self.global_step)
            # self.global_step += 1
            # if (epoch + 1) == self.config.max_epoch:

    # save checkpoint
    @staticmethod
    def _save_checkpoint(state, filename='checkpoint.pth.tar'):
        torch.save(state, filename)


def main(cfg):
    model = torch.load('../model_hub/TID2013_IQA_Meta_resnet18.pt')
    model.resnet_layer.avgpool = nn.AvgPool2d(kernel_size=7, stride=1, padding=0)
    model.net.fc3 = nn.Linear(512, 5)
    model.net.sig = nn.Softmax(dim=1)
    t = Trainer(cfg, model)
    t.fit()


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
