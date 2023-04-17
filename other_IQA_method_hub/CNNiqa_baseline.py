import argparse
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

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _get_nonoverlap_patches(x):
    """non overlap patches"""
    patch_num = 0

    b, c, h, w = x.shape
    th = tw = 64
    cropped_x = []

    for i in range(0, h - th, th):
        for j in range(0, w - tw, tw):
            cropped_x.append(x[:, :, i:i + 32, j:j + 32])
            patch_num += 1

    cropped_x = torch.stack(cropped_x, dim=1).reshape(-1, c, 32, 32)
    return cropped_x


def LocalNormalization(patch, P=3, Q=3, C=1):
    kernel = np.ones((P, Q)) / (P * Q)
    patch_mean = convolve2d(patch, kernel, boundary='symm', mode='same')
    patch_sm = convolve2d(np.square(patch), kernel, boundary='symm', mode='same')
    patch_std = np.sqrt(np.maximum(patch_sm - np.square(patch_mean), 0)) + C
    patch_ln = torch.from_numpy((patch - patch_mean) / patch_std).float().unsqueeze(0)
    return patch_ln


class Trainer(object):
    def __init__(self, config):
        torch.manual_seed(config.seed)
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.train_dataloader = creator.datasets_creator(self.config)
        self.model = creator.model_creator(self.config)
        self.model = self.model.to(device)
        if torch.cuda.device_count() > 1:
            self.model = torch.nn.DataParallel(self.model, device_ids=[0, 1])
        self.model_name = type(self.model).__name__ + self.config.train_description

        self.ckpt_path = config.ckpt_path
        self.start_epoch = 0
        self.max_epoch = self.config.max_epoch
        self.global_step = 1
        runs_path = os.path.join(self.config.tensorboard_path, self.model_name + str(self.config.split))
        self.logger = sum_writer(runs_path)
        self.loss_fn, self.optimizer, self.scheduler = creator.optimal_creator(self.config, self.model)
        self.loss_fn.to(self.device)

    def fit(self):
        for epoch in tqdm(range(self.start_epoch, self.max_epoch)):
            self._train_one_epoch(epoch)
            # self.scheduler.step()
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
            y = torch.reshape(Variable(y), [-1, 5])
            x = x.to(self.device)
            y = y.to(self.device)
            self.optimizer.zero_grad()
            predict_student = self.model(x)
            self.loss = self.loss_fn(predict_student, y.float().detach())
            self.loss.backward()
            self.optimizer.step()

            self.logger.add_scalar(tag='sum_loss',
                                   scalar_value=self.loss.item(),
                                   global_step=self.global_step)
            self.global_step += 1
            # if (epoch + 1) == self.config.max_epoch:

    # save checkpoint
    @staticmethod
    def _save_checkpoint(state, filename='checkpoint.pth.tar'):
        torch.save(state, filename)


def main(cfg):
    t = Trainer(cfg)
    t.fit()


if __name__ == "__main__":
    config = config_hub.CNNIQA_config()
    for i in range(0, 10):
        config = config_hub.CNNIQA_config()
        split = i + 1
        config.split = split
        config.ckpt_path = os.path.join(config.ckpt_path, str(config.split))
        if not os.path.exists(config.ckpt_path):
            os.makedirs(config.ckpt_path)
        print(config.network, config.train_description)
        main(config)
