import argparse

# import ml_collections

r"""create_datasets.
创建各种不同的config文件，你可以在这里调用或者更改
"""


def CNNIQA_config():
    parser = argparse.ArgumentParser()

    parser.add_argument("--state", type=str, default='NNID', help='')
    parser.add_argument("--network", type=str, default='cnniqa', help='在creator中有说明可调用选项')
    parser.add_argument("--train_description", type=str, default='BL_2K', help='train_description')
    parser.add_argument("--seed", type=int, default=19980206)
    parser.add_argument('--split', type=int, default='1')
    parser.add_argument('--ckpt_path', default='./checkpoint', type=str,
                        metavar='PATH', help='path to checkpoints')
    parser.add_argument('--tensorboard_path', default='./logs', type=str,
                        metavar='PATH', help='path to checkpoints')
    parser.add_argument('--ckpt', default='CNNIQAnetBL_2K-00500.pt', type=str,
                        metavar='PATH', help='checkpoint name to load')

    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--number_workers", type=int, default=8)
    parser.add_argument("--pin_memory", type=bool, default=True)
    parser.add_argument("--max_epoch", type=int, default=500)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--decay_interval", type=int, default=2)
    parser.add_argument("--decay_ratio", type=float, default=0.5)
    return parser.parse_args()


def Mbcnn_config():
    parser = argparse.ArgumentParser()

    parser.add_argument("--state", type=str, default='test', help='')
    parser.add_argument("--network", type=str, default='mbcnn', help='在creator中有说明可调用选项')
    parser.add_argument("--train_description", type=str, default='BL_2K', help='train_description')
    parser.add_argument("--seed", type=int, default=19980206)
    parser.add_argument('--split', type=int, default='1')
    parser.add_argument('--ckpt_path', default='./checkpoint', type=str,
                        metavar='PATH', help='path to checkpoints')
    parser.add_argument('--ckpt', default='./checkpoint', type=str,
                        metavar='PATH', help='path to checkpoints')
    parser.add_argument('--tensorboard_path', default='./logs', type=str,
                        metavar='PATH', help='path to checkpoints')

    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--number_workers", type=int, default=8)
    parser.add_argument("--pin_memory", type=bool, default=True)
    parser.add_argument("--max_epoch", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--decay_interval", type=int, default=50)
    parser.add_argument("--decay_ratio", type=float, default=0.9)
    return parser.parse_args()


def Dbcnn_config():
    parser = argparse.ArgumentParser()

    parser.add_argument("--state", type=str, default='train', help='')
    parser.add_argument("--network", type=str, default='dbcnn', help='在creator中有说明可调用选项')
    parser.add_argument("--train_description", type=str, default='BL_2K', help='train_description')
    parser.add_argument("--seed", type=int, default=19980206)
    parser.add_argument('--split', type=int, default='1')
    parser.add_argument('--ckpt_path', default='./checkpoint', type=str,
                        metavar='PATH', help='path to checkpoints')
    parser.add_argument('--tensorboard_path', default='./logs', type=str,
                        metavar='PATH', help='path to checkpoints')

    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--number_workers", type=int, default=8)
    parser.add_argument("--pin_memory", type=bool, default=True)
    parser.add_argument("--max_epoch", type=int, default=50)
    parser.add_argument("--lr", type=float, default=0.00001)
    parser.add_argument("--decay_interval", type=int, default=50)
    parser.add_argument("--decay_ratio", type=float, default=0.9)
    return parser.parse_args()


def wad_config():
    parser = argparse.ArgumentParser()

    parser.add_argument("--state", type=str, default='test', help='')
    parser.add_argument("--network", type=str, default='wad', help='在creator中有说明可调用选项')
    parser.add_argument("--train_description", type=str, default='BL_2K', help='train_description')
    parser.add_argument("--seed", type=int, default=19980206)
    parser.add_argument('--split', type=int, default='1')
    parser.add_argument('--ckpt_path', default='./checkpoint', type=str,
                        metavar='PATH', help='path to checkpoints')
    parser.add_argument('--ckpt', default='NRnetBL_2K-03000.pt', type=str,
                        metavar='PATH', help='checkpoint name to load')
    parser.add_argument('--tensorboard_path', default='./logs', type=str,
                        metavar='PATH', help='path to checkpoints')

    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--number_workers", type=int, default=8)
    parser.add_argument("--pin_memory", type=bool, default=True)
    parser.add_argument("--max_epoch", type=int, default=3000)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--decay_interval", type=int, default=100)
    parser.add_argument("--decay_ratio", type=float, default=0.8)
    return parser.parse_args()


def P2p_config():
    parser = argparse.ArgumentParser()

    parser.add_argument("--state", type=str, default='train', help='')
    parser.add_argument("--network", type=str, default='p2p', help='在creator中有说明可调用选项')
    parser.add_argument("--train_description", type=str, default='BL_2K', help='train_description')
    parser.add_argument("--seed", type=int, default=19980206)
    parser.add_argument('--split', type=int, default='1')
    parser.add_argument('--ckpt_path', default='./checkpoint', type=str,
                        metavar='PATH', help='path to checkpoints')
    parser.add_argument('--ckpt', default='PAQ2PIQBL_2K-00010.pt', type=str,
                        metavar='PATH', help='checkpoint name to load')
    parser.add_argument('--tensorboard_path', default='./logs', type=str,
                        metavar='PATH', help='path to checkpoints')

    parser.add_argument("--batch_size", type=int, default=80)
    parser.add_argument("--number_workers", type=int, default=8)
    parser.add_argument("--pin_memory", type=bool, default=True)
    parser.add_argument("--max_epoch", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--decay_interval", type=int, default=100)
    parser.add_argument("--decay_ratio", type=float, default=0.8)
    return parser.parse_args()


def Metal_config():
    parser = argparse.ArgumentParser()

    parser.add_argument("--state", type=str, default='test', help='')
    parser.add_argument("--network", type=str, default='metal', help='在creator中有说明可调用选项')
    parser.add_argument("--train_description", type=str, default='BL_2K', help='train_description')
    parser.add_argument("--seed", type=int, default=19980206)
    parser.add_argument('--split', type=int, default='1')
    parser.add_argument('--ckpt_path', default='./checkpoint', type=str,
                        metavar='PATH', help='path to checkpoints')
    parser.add_argument('--ckpt', default='NetBL_2K-00100.pt', type=str,
                        metavar='PATH', help='checkpoint name to load')
    parser.add_argument('--tensorboard_path', default='./logs', type=str,
                        metavar='PATH', help='path to checkpoints')

    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--number_workers", type=int, default=8)
    parser.add_argument("--pin_memory", type=bool, default=True)
    parser.add_argument("--max_epoch", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--decay_interval", type=int, default=100)
    parser.add_argument("--decay_ratio", type=float, default=0.8)
    return parser.parse_args()


def SFA_config():
    parser = argparse.ArgumentParser()

    parser.add_argument("--state", type=str, default='train', help='')
    parser.add_argument("--network", type=str, default='sfa', help='在creator中有说明可调用选项')
    parser.add_argument("--train_description", type=str, default='BL_2K', help='train_description')
    parser.add_argument("--seed", type=int, default=19980206)
    parser.add_argument('--split', type=int, default='1')
    parser.add_argument('--ckpt_path', default='./checkpoint', type=str,
                        metavar='PATH', help='path to checkpoints')
    parser.add_argument('--ckpt', default='', type=str,
                        metavar='PATH', help='checkpoint name to load')
    parser.add_argument('--tensorboard_path', default='./logs', type=str,
                        metavar='PATH', help='path to checkpoints')

    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--number_workers", type=int, default=8)
    parser.add_argument("--pin_memory", type=bool, default=True)
    parser.add_argument("--max_epoch", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--decay_interval", type=int, default=100)
    parser.add_argument("--decay_ratio", type=float, default=0.8)
    return parser.parse_args()


def NASSADNN_config():
    parser = argparse.ArgumentParser()

    parser.add_argument("--state", type=str, default='train', help='')
    parser.add_argument("--network", type=str, default='nssadnn', help='在creator中有说明可调用选项')
    parser.add_argument("--train_description", type=str, default='BL_2K', help='train_description')
    parser.add_argument("--seed", type=int, default=19980206)
    parser.add_argument('--split', type=int, default='1')
    parser.add_argument('--ckpt_path', default='./checkpoint', type=str,
                        metavar='PATH', help='path to checkpoints')
    parser.add_argument('--ckpt', default='NSSADNNBL_2K-03000.pt', type=str,
                        metavar='PATH', help='checkpoint name to load')
    parser.add_argument('--tensorboard_path', default='./logs', type=str,
                        metavar='PATH', help='path to checkpoints')

    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--number_workers", type=int, default=8)
    parser.add_argument("--pin_memory", type=bool, default=True)
    parser.add_argument("--max_epoch", type=int, default=3000)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--decay_interval", type=int, default=100)
    parser.add_argument("--decay_ratio", type=float, default=0.8)
    return parser.parse_args()