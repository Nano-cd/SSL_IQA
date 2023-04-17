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

from utils import compute_metric

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



class Trainer(object):
    def __init__(self, config):
        torch.manual_seed(config.seed)
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.test_dataloader = creator.datasets_creator(self.config)
        self.model = creator.model_creator(self.config)
        self.model = self.model.to(device)
        if torch.cuda.device_count() > 1:
            self.model = torch.nn.DataParallel(self.model, device_ids=[0, 1])
        self.model_name = type(self.model).__name__ + self.config.train_description

        self.ckpt_path = config.ckpt_path
        self.start_epoch = 0
        self.max_epoch = self.config.max_epoch
        self.global_step = 1
        self.loss_fn, self.optimizer, self.scheduler = creator.optimal_creator(self.config, self.model)
        self.loss_fn.to(self.device)

    def _load_checkpoint(self, ckpt1):
        if os.path.isfile(ckpt1):
            print("[*] loading checkpoint '{}'".format(ckpt1))
            checkpoint = torch.load(ckpt1)
            self.start_epoch = checkpoint['epoch'] + 1
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            print("[*] loaded checkpoint '{}' (epoch {})"
                  .format(ckpt1, checkpoint['epoch']))
        else:
            print("[!] no checkpoint found at '{}'".format(ckpt1))

    def _test(self):
        y_ = []
        y_pred = []
        buffer_out = []
        buffer_label = []
        indx = 0
        self.model.eval()
        ckpt = os.path.join(config.ckpt_path, config.ckpt)
        self._load_checkpoint(ckpt)
        with torch.no_grad():
            for _, (patches, (label, features)) in enumerate(self.test_dataloader):
                patches = patches.to(device)
                label = label.to(device)
                label = torch.reshape(label, [-1, 5])
                outputs_q = self.model(patches)[0]
                sum_index = torch.tensor([1, 2, 3, 4, 5], requires_grad=False, device=self.device,
                                         dtype=torch.float).unsqueeze(1)
                outputs = torch.mm(outputs_q, sum_index).squeeze(dim=1)
                labels = torch.mm(label, sum_index).squeeze(dim=1)
                buffer_out.append(outputs.cpu().numpy())
                buffer_label.append(labels.cpu().numpy())
                indx = indx+1
                if indx == 35:
                    y_.append(np.mean(buffer_out))
                    y_pred.append(np.mean(buffer_label))
                    buffer_out = []
                    buffer_label = []
                    indx = 0

            RMSE, PLCC, SROCC, KROCC = compute_metric(y_, y_pred)
            return PLCC, SROCC

def main(cfg):
    t = Trainer(cfg)
    plcc_, srocc_ = t._test()
    print(plcc_, srocc_)


if __name__ == "__main__":
    config = config_hub.NASSADNN_config()
    for i in range(0, 10):
        config = config_hub.NASSADNN_config()
        split = i + 1
        config.split = split
        config.ckpt_path = os.path.join(config.ckpt_path, str(config.split))
        if not os.path.exists(config.ckpt_path):
            os.makedirs(config.ckpt_path)
        print(config.network, config.train_description)
        main(config)
