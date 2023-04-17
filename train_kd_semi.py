import argparse
import os
from itertools import cycle

import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

import ramps
import utils
from utils import compute_metric
from utils import RkdDistance, RKdAngle
from utils import relative_rank_loss
from torch.utils.tensorboard import SummaryWriter as sum_writer
import model_hub
import dataset.create_dataset
from model_hub import Google_net_modify, Resnet101_modify, Resnet18_modify, Alexnet_modify, Resnet50_modify
from model_hub import Inceptionresnetv2_modify, swin_vit_modify, Tres_modify

import torch.autograd
from torch.autograd import Variable
import torch.backends.cudnn


def parse_config():
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_mode", type=str, default='ML', help='Mutual learning')
    parser.add_argument("--data_mode", type=str, default='semi-train', help='')
    parser.add_argument('--data', type=str, default='kon10k2000')
    parser.add_argument("--train", type=bool, default=True)
    parser.add_argument('--ratio', type=int, default=2)

    parser.add_argument("--network", type=str, default='tres', help='googlenet, alexnet')
    parser.add_argument("--train_description1", type=str, default='rkd_tres1:2_2KS',
                        help='train_description')
    parser.add_argument("--train_description2", type=str, default='rkd_tres1:2_2KT',
                        help='train_description')
    parser.add_argument('--weights', default='./checkpoint/vit_pretrained/swin_base_patch4_window7_224.pth', type=str,
                        metavar='PATH', help='path to vit checkpoints')

    parser.add_argument("--seed", type=int, default=19980206)
    parser.add_argument('--split', type=int, default='1')
    parser.add_argument('--ckpt_path', default='./checkpoint', type=str,
                        metavar='PATH', help='path to checkpoints')
    parser.add_argument('--ckpt1', default='',
                        type=str, help='name of the first checkpoint to load')
    parser.add_argument('--ckpt2', default='',
                        type=str, help='name of the second checkpoint to load')
    parser.add_argument('--tensorboard_path', default='./logs', type=str,
                        metavar='PATH', help='path to checkpoints')

    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--number_workers", type=int, default=4)
    parser.add_argument("--pin_memory", type=bool, default=True)
    parser.add_argument("--max_epoch", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.0002)
    parser.add_argument("--decay_interval", type=int, default=2)
    parser.add_argument("--decay_ratio", type=float, default=0.5)

    return parser.parse_args()


os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Trainer(object):
    def __init__(self, config):
        torch.manual_seed(config.seed)
        self.global_step = 0
        self.config = config
        self.train_mode = config.train_mode
        self.data_mode = config.data_mode
        # initialize the data_loader
        if self.config.data == 'kon10k1000':
            self.train_dataloader, self.train_un_dataloader = dataset.create_dataset.kon10k_1000(self.config)
        elif self.config.data == 'kon10k2000':
            self.train_dataloader, self.train_un_dataloader = dataset.create_dataset.kon10k_2000(self.config)
        elif self.config.data == 'kadid10k1000':
            self.train_dataloader, self.train_un_dataloader = dataset.create_dataset.kadid10k_1000(self.config)
        elif self.config.data == 'kadid10k2000':
            self.train_dataloader, self.train_un_dataloader = dataset.create_dataset.kadid10k_2000(self.config)

        # initialize the model
        if config.network == 'googlenet':
            print('googlenet and resnet101 model selected')
            self.model1 = Google_net_modify.Google_net(num_classes=5)
            self.model2 = Resnet101_modify.Resnet_101(num_classes=5)
        elif config.network == 'alexnet':
            print('alxenet and resnet101 model selected')
            self.model1 = Alexnet_modify.Alex_net(num_classes=5)
            self.model2 = Resnet101_modify.Resnet_101(num_classes=5)
        elif config.network == 'inceptionresv2':
            print('INCEPTIONRESV2 model selected')
            self.model1 = Alexnet_modify.Alex_net(num_classes=5)
            self.model2 = Inceptionresnetv2_modify.Inceptionresnet_v2(num_classes=5)
        elif config.network == 'tres':
            print('tres model selected')
            self.model1 = Alexnet_modify.Alex_net(num_classes=5)
            self.model2 = Tres_modify.Net()
        else:
            raise NotImplementedError("Not supported network, need to be added!")

        self.model1 = self.model1.to(device)
        self.model_name1 = type(self.model1).__name__ + self.config.train_description1
        if torch.cuda.device_count() > 1:
            self.model1 = torch.nn.DataParallel(self.model1, device_ids=[0, 1])

        self.model2 = self.model2.to(device)
        self.model_name2 = type(self.model2).__name__ + self.config.train_description2
        if torch.cuda.device_count() > 1:
            self.model2 = torch.nn.DataParallel(self.model2, device_ids=[0, 1])

        # initialize the loss function and optimizer
        self.start_epoch = 0
        self.max_epoch = config.max_epoch
        self.loss_fn = torch.nn.MSELoss()
        self.rkd_dis_loss = RkdDistance()
        self.rkd_ag_loss = RKdAngle()
        self.ckpt_path = config.ckpt_path
        self.loss_fn.to(device)
        self.initial_lr = config.lr
        self.optimizer1 = torch.optim.Adam(self.model1.parameters(), lr=config.lr)
        self.optimizer2 = torch.optim.Adam(self.model2.parameters(), lr=config.lr)
        self.scheduler1 = torch.optim.lr_scheduler.StepLR(self.optimizer1,
                                                          last_epoch=self.start_epoch - 1,
                                                          step_size=config.decay_interval,
                                                          gamma=config.decay_ratio)
        self.scheduler2 = torch.optim.lr_scheduler.StepLR(self.optimizer2,
                                                          last_epoch=self.start_epoch - 1,
                                                          step_size=config.decay_interval,
                                                          gamma=config.decay_ratio)
        runs_path = os.path.join(self.config.tensorboard_path, self.model_name1 + str(self.config.split))
        self.logger = sum_writer(runs_path)
        if not config.train:
            ckpt1 = os.path.join(config.ckpt_path, config.ckpt1)
            ckpt2 = os.path.join(config.ckpt_path, config.ckpt2)
            self._load_checkpoint(ckpt1=ckpt1, ckpt2=ckpt2)

    def fit(self):
        if self.train_mode == 'ML':
            for epoch in tqdm(range(self.start_epoch, self.max_epoch)):
                self._train_one_epoch(epoch)
                self.scheduler1.step()
                self.scheduler2.step()

    def _train_one_epoch(self, epoch):
        # print('Adam learning rate: {:.8f}'.format(self.optimizer.param_groups[0]['lr']))
        self.model1.train()
        self.model2.train()
        for _, data in enumerate(zip(self.train_dataloader, self.train_un_dataloader)):
            x = Variable(data[0][0]).cuda(non_blocking=True)
            y = torch.reshape(Variable(data[0][1]), [self.config.batch_size, -1]).cuda(non_blocking=True)
            x_un = Variable(data[1][0]).cuda(non_blocking=True)

            predict_student, a1 = self.model1(x)
            predict_teacher, a2 = self.model2(x)
            predict_student_un, activation1 = self.model1(x_un)
            predict_teacher_un, activation2 = self.model2(x_un)

            self.loss1 = self.loss_fn(predict_student, y.float().detach())
            self.loss_IKD = self.loss_fn(predict_student_un, predict_teacher_un.detach()) \
                            + self.loss_fn(predict_student, predict_teacher.detach())
            self.loss_AGKD = self.rkd_ag_loss(activation1, activation2) + self.rkd_ag_loss(a1, a2)
            self.loss2 = self.loss_fn(predict_teacher, y.float().detach())

            self.model1_loss = self.loss1 + ramps.sigmoid_rampup(epoch, 10)*(self.loss_IKD+100*self.loss_AGKD)
            self.model2_loss = self.loss2
            self.sum_loss = self.model1_loss + self.model2_loss

            self.optimizer1.zero_grad()
            self.optimizer2.zero_grad()
            self.sum_loss.backward()
            self.optimizer1.step()
            self.optimizer2.step()

            self.logger.add_scalar(tag='Branch_1!/supervison',
                                   scalar_value=self.loss1.item(),
                                   global_step=self.global_step)
            # self.logger.add_scalar(tag='Branch_1!/consis',
            #                        scalar_value=self.loss_consistency1.item(),
            #                        global_step=self.global_step)
            self.logger.add_scalar(tag='Branch_1!/model_loss',
                                   scalar_value=self.model1_loss.item(),
                                   global_step=self.global_step)
            self.logger.add_scalar(tag='Branch_2!/supervison',
                                   scalar_value=self.loss2.item(),
                                   global_step=self.global_step)
            self.logger.add_scalar(tag='Branch_2!/model_loss',
                                   scalar_value=self.model2_loss.item(),
                                   global_step=self.global_step)
            self.logger.add_scalar(tag='sum_loss/sum_loss',
                                   scalar_value=self.sum_loss.item(),
                                   global_step=self.global_step)

            self.global_step += 1

            if (epoch + 1) == self.max_epoch:
                model_name1 = '{}-{:0>5d}.pt'.format(self.model_name1, epoch)
                model_name1 = os.path.join(self.ckpt_path, model_name1)
                model_name2 = '{}-{:0>5d}.pt'.format(self.model_name2, epoch)
                model_name2 = os.path.join(self.ckpt_path, model_name2)
                self._save_checkpoint({
                    'epoch': epoch,
                    'state_dict': self.model1.state_dict(),
                    'optimizer': self.optimizer1.state_dict(),
                }, model_name1)
                # self._save_checkpoint({
                #     'epoch': epoch,
                #     'state_dict': self.model2.state_dict(),
                #     'optimizer': self.optimizer2.state_dict(),
                # }, model_name2)

    def evl(self):
        y_ = []
        y_pred = []
        self.model1.eval()
        self.model2.eval()
        if self.config.data_mode == 'test':
            with torch.no_grad():
                for index, (images, labels) in enumerate(self.train_dataloader):
                    images = images.cuda()
                    outputs = self.model1(images)
                    y_pred.extend(outputs.squeeze(dim=1).cpu())
                    y_.extend(labels)

                RMSE, PLCC, SROCC, KROCC = compute_metric(np.array(y_), np.array(y_pred))
        return PLCC, SROCC

    def _load_checkpoint(self, ckpt1, ckpt2):
        if os.path.isfile(ckpt1):
            print("[*] loading checkpoint '{}'".format(ckpt1))
            checkpoint = torch.load(ckpt1)
            self.start_epoch = checkpoint['epoch'] + 1
            self.model1.load_state_dict(checkpoint['state_dict'])
            self.optimizer1.load_state_dict(checkpoint['optimizer'])
            if self.initial_lr is not None:
                for param_group in self.optimizer1.param_groups:
                    param_group['initial_lr'] = self.initial_lr
            print("[*] loaded checkpoint '{}' (epoch {})"
                  .format(ckpt1, checkpoint['epoch']))
        else:
            print("[!] no checkpoint found at '{}'".format(ckpt1))

        if os.path.isfile(ckpt2):
            print("[*] loading checkpoint '{}'".format(ckpt2))
            checkpoint = torch.load(ckpt2)
            self.start_epoch = checkpoint['epoch'] + 1
            self.model2.load_state_dict(checkpoint['state_dict'])
            self.optimizer2.load_state_dict(checkpoint['optimizer'])
            if self.initial_lr is not None:
                for param_group in self.optimizer1.param_groups:
                    param_group['initial_lr'] = self.initial_lr
            print("[*] loaded checkpoint '{}' (epoch {})"
                  .format(ckpt2, checkpoint['epoch']))
        else:
            print("[!] no checkpoint found at '{}'".format(ckpt2))

    # save checkpoint
    @staticmethod
    def _save_checkpoint(state, filename='checkpoint.pth.tar'):
        torch.save(state, filename)


def main(cfg):
    t = Trainer(cfg)
    if cfg.train:
        t.fit()
    else:
        plcc_, srocc_ = t.evl()
        print(plcc_, srocc_)


if __name__ == "__main__":
    config = parse_config()
    for i in range(0, 10):
        config = parse_config()
        split = i + 1
        config.split = split
        config.ckpt_path = os.path.join(config.ckpt_path, str(config.split))
        if not os.path.exists(config.ckpt_path):
            os.makedirs(config.ckpt_path)
        print(config.train_description1)
        main(config)
