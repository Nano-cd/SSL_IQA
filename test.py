import argparse
import os

from tqdm import tqdm
import numpy as np
from torch.utils.tensorboard import SummaryWriter as sum_writer
import dataset.create_dataset
import torch.autograd
from torch.autograd import Variable
import torch.backends.cudnn
from model_hub import Google_net_modify, Resnet101_modify, Alexnet_modify, Shufflenet_v205_modify, VGG_11_modify, \
    mobilenet_v3_samll_modify, Resnet50_modify, Mnas_net_modify, Inceptionresnetv2_modify, DBCNN_modify, CNNIQA, \
    WaDIQaM_modify, Paq2piq_modify, densent_modify
from utils import compute_metric


def parse_config():
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_mode", type=str, default='normal')

    parser.add_argument("--network", type=str, default='alexnet'
                        , help='resnet34, vgg16, '
                               'densenet121, inceptionresv2, alexnet, Inceptionresnetv2')

    parser.add_argument("--data_mode", type=str, default='test', help='train test')
    parser.add_argument("--train", type=bool, default=False)
    parser.add_argument('--data', type=str, default='kon10k2000')
    parser.add_argument("--train_description", type=str, default='BL2K',
                        help='train_description')
    parser.add_argument("--seed", type=int, default=19980206)
    parser.add_argument('--split', type=int, default='1')

    parser.add_argument('--ckpt_path', default='./checkpoint', type=str,
                        metavar='PATH', help='path to checkpoints')
    parser.add_argument('--ckpt', default='Alex_netRKD100_resnet_1_2KS-00009.pt',
                        type=str, help='name of the checkpoint to load')
    parser.add_argument('--tensorboard_path', default='./runs', type=str,
                        metavar='PATH', help='path to checkpoints')

    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--number_workers", type=int, default=4)
    parser.add_argument("--pin_memory", type=bool, default=True)
    parser.add_argument("--max_epoch", type=int, default=40)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--decay_interval", type=int, default=10)
    parser.add_argument("--decay_ratio", type=float, default=0.5)
    parser.add_argument("--init", type=str, default='kaiming_norm')

    return parser.parse_args()


# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Trainer(object):
    def __init__(self, config):
        torch.manual_seed(config.seed)
        self.config = config
        self.train_mode = config.train_mode
        self.data_mode = config.data_mode
        # initialize the data_loader
        if self.config.data == 'kon10k1000':
            self.train_dataloader = dataset.create_dataset.kon10k_1000(self.config)
        elif self.config.data == 'kon10k2000':
            self.train_dataloader = dataset.create_dataset.kon10k_2000(self.config)
        elif self.config.data == 'kon10k3000':
            self.train_dataloader = dataset.create_dataset.kon10k_3000(self.config)
        elif self.config.data == 'kon10k8000':
            self.train_dataloader = dataset.create_dataset.kon10k_8000(self.config)
        elif self.config.data == 'kadid10k1000':
            self.train_dataloader = dataset.create_dataset.kadid10k_1000(self.config)
        elif self.config.data == 'kadid10k2000':
            self.train_dataloader = dataset.create_dataset.kadid10k_2000(self.config)
        elif self.config.data == 'kadid10k3000':
            self.train_dataloader = dataset.create_dataset.kadid10k_3000(self.config)
        elif self.config.data == 'kadid10k8000':
            self.train_dataloader = dataset.create_dataset.kadid10k_3000(self.config)
        elif self.config.data == 'LIVE_C':
            self.train_dataloader = dataset.create_dataset.live_c(self.config)
        elif self.config.data == 'NNID':
            self.train_dataloader = dataset.create_dataset.NNID(self.config)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # initialize the model
        if config.network == 'resnet101':
            print('resnet101 model selected')
            self.model = Resnet101_modify.Resnet_101(num_classes=5)
        elif config.network == 'alexnet':
            print('alexnet model selected')
            self.model = Alexnet_modify.Alex_net(num_classes=5)
        elif config.network == 'googlenet':
            print('googlenet model selected')
            self.model = Google_net_modify.Google_net(num_classes=5)
        elif config.network == 'shufflenet':
            print('shufflenet model selected')
            self.model = Shufflenet_v205_modify.Shuffle_net(num_classes=5)
        elif config.network == 'vgg11':
            print('vgg11 model selected')
            self.model = VGG_11_modify.VGG11(num_classes=5)
        elif config.network == 'mvs':
            print('mvs model selected')
            self.model = mobilenet_v3_samll_modify.MVS(num_classes=5)
        elif config.network == 'resnet50':
            print('resnet50 model selected')
            self.model = Resnet50_modify.Resnet_50(num_classes=5)
        elif config.network == 'mnas':
            print('mnas model selected')
            self.model = Mnas_net_modify.Mnas_net(num_classes=5)
        elif config.network == 'inceptionresv2':
            print('INCEPTIONRESV2 model selected')
            self.model = Inceptionresnetv2_modify.Inceptionresnet_v2(num_classes=5)
        elif config.network == 'DBCNN':
            print('DBCNN model selected')
            self.model = DBCNN_modify.DBCNN(scnn_root='./checkpoint/scnn.pkl',num_classes=5)
        elif config.network == 'CNNIQA':
            print('CNNIQA model selected')
            self.model = CNNIQA.CNNIQA()
        elif config.network == 'P2P':
            print('P2P model selected')
            self.model = Paq2piq_modify.PAQ2PIQ(num_classes=5,pretrained=False)
        elif config.network == 'WADIQ':
            print('WADIQ model selected')
            self.model = WaDIQaM_modify.WaDIQaM(num_classes=5)
        elif config.network == 'densenet':
            print('densenet model selected')
            self.model = densent_modify.Densenet_121(num_classes=5)
        else:
            raise NotImplementedError("Not supported network, need to be added!")

        self.model.to(self.device)
        if torch.cuda.device_count()>1:
            self.model = torch.nn.DataParallel(self.model, device_ids=[0,1])
        self.model_name = type(self.model).__name__ + self.config.train_description

        # initialize the loss function and optimizer
        self.start_epoch = 0
        self.max_epoch = config.max_epoch
        self.loss_fn = torch.nn.MSELoss()
        self.ckpt_path = config.ckpt_path
        self.loss_fn.to(self.device)
        self.initial_lr = config.lr
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.lr, betas=(0.9, 0.999))
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,
                                                         last_epoch=self.start_epoch - 1,
                                                         step_size=config.decay_interval,
                                                         gamma=config.decay_ratio)
        self.global_step = 1
        runs_path = os.path.join(self.config.tensorboard_path, self.model_name + str(self.config.split))
        self.logger = sum_writer(runs_path)

        if not config.train:
            ckpt = os.path.join(config.ckpt_path, config.ckpt)
            self._load_checkpoint(ckpt=ckpt)

    def fit(self):
        if self.train_mode == 'normal':
            for epoch in tqdm(range(self.start_epoch, self.max_epoch)):
                self._train_one_epoch(epoch)
                self.scheduler.step()

    def _train_one_epoch(self, epoch):
        # start training
        # print('Adam learning rate: {:.8f}'.format(self.optimizer.param_groups[0]['lr']))
        self.model.train()
        for _, (x, y) in enumerate(self.train_dataloader):
            x = Variable(x)
            y = Variable(y)
            x = x.to(self.device)
            y = y.to(self.device).view(-1, 1)
            self.optimizer.zero_grad()
            predict_student, _ = self.model(x)
            self.loss = self.loss_fn(predict_student, y.float().detach())
            self.loss.backward()
            self.optimizer.step()

            self.logger.add_scalar(tag='sum_loss',
                                   scalar_value=self.loss.item(),
                                   global_step=self.global_step)
            self.global_step += 1
            # if (epoch + 1) == self.config.max_epoch:
            if (epoch + 1) > 30 and (epoch + 1) % 10 == 0:
                model_name = '{}-{:0>5d}.pt'.format(self.model_name, epoch + 1)
                model_name = os.path.join(self.ckpt_path, model_name)
                self._save_checkpoint({
                    'epoch': epoch,
                    'state_dict': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                }, model_name)

    def evl(self):
        y_ = []
        y_pred = []
        self.model.eval()
        if self.config.data_mode == 'test':
            with torch.no_grad():
                for index, (images, labels) in enumerate(self.train_dataloader):
                    images = images.cuda()
                    labels = labels.cuda()
                    outputs, _ = self.model(images)
                    sum_index = torch.tensor([1, 2, 3, 4, 5], requires_grad=False,device=self.device,dtype=torch.float).unsqueeze(1)
                    outputs = torch.mm(outputs, sum_index).squeeze(dim=1)
                    labels = torch.mm(labels, sum_index).squeeze(dim=1)
                    y_.extend(labels.cpu().numpy())
                    y_pred.extend(outputs.cpu().numpy())
                y_ = np.mean(np.reshape(np.array(y_), (-1, 10)), axis=1)
                y_pred = np.mean(np.reshape(np.array(y_pred), (-1, 10)), axis=1)
                RMSE, PLCC, SROCC, KROCC = compute_metric(y_, y_pred)
        return PLCC, SROCC, RMSE, KROCC

    def _load_checkpoint(self, ckpt):
        if os.path.isfile(ckpt):
            print("[*] loading checkpoint '{}'".format(ckpt))
            checkpoint = torch.load(ckpt)
            self.start_epoch = checkpoint['epoch'] + 1
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            if self.initial_lr is not None:
                for param_group in self.optimizer.param_groups:
                    param_group['initial_lr'] = self.initial_lr
            print("[*] loaded checkpoint '{}' (epoch {})"
                  .format(ckpt, checkpoint['epoch']))
        else:
            print("[!] no checkpoint found at '{}'".format(ckpt))

    # save checkpoint
    @staticmethod
    def _save_checkpoint(state, filename='checkpoint.pth.tar'):
        torch.save(state, filename)


def main(cfg):
    t = Trainer(cfg)
    if cfg.train:
        t.fit()
    else:
        plcc_, srocc_,rmes_, krocc_ = t.evl()
        print(plcc_, srocc_,rmes_, krocc_)


if __name__ == "__main__":
    config = parse_config()
    # seed_torch(config)
    for i in range(0, 10):
        config = parse_config()
        split = i + 1
        config.split = split
        config.ckpt_path = os.path.join(config.ckpt_path, str(config.split))
        if not os.path.exists(config.ckpt_path):
            os.makedirs(config.ckpt_path)
        print(config.network, config.train_description)
        main(config)
