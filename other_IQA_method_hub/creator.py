import torch.nn
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
import config_hub
from dataset_hub import metal_dataset, p2p_dataset, cnniqa_dataset, mbcnn_dataset, dbcnn_dataset, wad_dataset, \
    sfa_dataset, nssadnn_dataset, nnid_dataset, livec_dataset

from model_hub import Metal_modify, CNNIQA_modify, Paq2piq_modify, WaDIQaM_modify, DBCNN_modify, MB_CNN_modify, \
    SFA_modify, NSSADNN_modiify

r"""create_datasets.
创建数据集的函数接口，你只需要输入config，返回相对应的dataloder
"""


def datasets_creator(config):
    if config.state == 'train':
        if config.network == 'cnniqa':
            train_data = cnniqa_dataset.IQADataset('../data/kon10k/train2000/labeled/labeled_' + str(config.split),
                                                   '../data/kon10k/1024x768')
            train_dataloader = DataLoader(train_data,
                                          batch_size=config.batch_size,
                                          shuffle=True,
                                          drop_last=True,
                                          num_workers=config.number_workers,
                                          pin_memory=config.pin_memory)
            print("train data size{}".format(config.batch_size * len(train_dataloader)))
            return train_dataloader
        elif config.network == 'mbcnn':
            train_data = mbcnn_dataset.IQADataset('../data/kon10k/train2000/labeled/labeled_' + str(config.split),
                                                  '../data/kon10k/1024x768',
                                                  '../gradiant_data/1024x768')
            train_dataloader = DataLoader(train_data,
                                          batch_size=config.batch_size,
                                          shuffle=True,
                                          drop_last=True,
                                          num_workers=config.number_workers,
                                          pin_memory=config.pin_memory)
            print("train data size{}".format(config.batch_size * len(train_dataloader)))
            return train_dataloader
        elif config.network == 'dbcnn':
            train_data = dbcnn_dataset.IQADataset('../data/kon10k/train2000/labeled/labeled_' + str(config.split),
                                                  '../data/kon10k/1024x768')
            train_dataloader = DataLoader(train_data,
                                          batch_size=config.batch_size,
                                          shuffle=True,
                                          drop_last=True,
                                          num_workers=config.number_workers,
                                          pin_memory=config.pin_memory)
            print("train data size{}".format(config.batch_size * len(train_dataloader)))
            return train_dataloader

        elif config.network == 'wad':
            train_data = wad_dataset.IQADataset_less_memory(
                '../data/kon10k/train2000/labeled/labeled_' + str(config.split),
                '../data/kon10k/1024x768')
            train_dataloader = DataLoader(train_data,
                                          batch_size=config.batch_size,
                                          shuffle=True,
                                          drop_last=True,
                                          num_workers=config.number_workers,
                                          pin_memory=config.pin_memory)
            print("train data size{}".format(config.batch_size * len(train_dataloader)))
            return train_dataloader
        elif config.network == 'p2p':
            train_data = p2p_dataset.IQADataset('../data/kon10k/train2000/labeled/labeled_' + str(config.split),
                                                '../data/kon10k/1024x768')
            train_dataloader = DataLoader(train_data,
                                          batch_size=config.batch_size,
                                          shuffle=True,
                                          drop_last=True,
                                          num_workers=config.number_workers,
                                          pin_memory=config.pin_memory)
            print("train data size{}".format(config.batch_size * len(train_dataloader)))
            return train_dataloader

        elif config.network == 'nssadnn':
            train_data = nssadnn_dataset.IQADataset(
                '../data/kon10k/train2000/labeled/labeled_' + str(config.split),
                '../data/kon10k/1024x768')
            train_dataloader = DataLoader(train_data,
                                          batch_size=config.batch_size,
                                          shuffle=True,
                                          drop_last=True,
                                          num_workers=config.number_workers,
                                          pin_memory=config.pin_memory)
            print("train data size{}".format(config.batch_size * len(train_dataloader)))
            return train_dataloader

        elif config.network == 'sfa':
            train_data = sfa_dataset.IQADataset('../data/kon10k/train2000/labeled/labeled_' + str(config.split),
                                                '../data/kon10k/1024x768')
            test_data = sfa_dataset.IQADataset('../data/kon10k/test2000/test_' + str(config.split),
                                               '../data/kon10k/1024x768')
            print('sfa data slected'+ str(config.split))
            return train_data, test_data

        elif config.network == 'metal':

            def my_collate(batch):
                batch = list(filter(lambda x: x is not None, batch))
                return default_collate(batch)

            train_data = metal_dataset.IQADataset('../data/kon10k/train2000/labeled/labeled_' + str(config.split),
                                                  '../data/kon10k/1024x768')
            train_dataloader = DataLoader(train_data,
                                          batch_size=config.batch_size,
                                          shuffle=False,
                                          drop_last=True,
                                          num_workers=0,
                                          pin_memory=config.pin_memory,
                                          collate_fn=my_collate)
            print("train data size{}".format(config.batch_size * len(train_dataloader)))
            return train_dataloader

    elif config.state == 'test':
        if config.network == 'cnniqa':
            test_data = cnniqa_dataset.IQADataset('../data/kon10k/test2000/test_' + str(config.split),
                                                  '../data/kon10k/1024x768')
            test_dataloader = DataLoader(test_data,
                                         batch_size=1,
                                         shuffle=False,
                                         drop_last=False,
                                         num_workers=config.number_workers,
                                         pin_memory=config.pin_memory)
            print("cnniqa TEST data size{}".format(len(test_dataloader)))
            return test_dataloader

        elif config.network == 'wad':
            test_data = wad_dataset.IQADataset_less_memory('../data/kon10k/test2000/test_' + str(config.split),
                                                           '../data/kon10k/1024x768')
            test_dataloader = DataLoader(test_data,
                                         batch_size=1,
                                         shuffle=False,
                                         drop_last=False,
                                         num_workers=config.number_workers,
                                         pin_memory=config.pin_memory)
            print("wad TEST data size{}".format(len(test_dataloader)))
            return test_dataloader
        elif config.network == 'dbcnn':
            test_data = dbcnn_dataset.IQADataset('../data/kon10k/test2000/test_' + str(config.split),
                                                 '../data/kon10k/1024x768')
            test_dataloader = DataLoader(test_data,
                                         batch_size=1,
                                         shuffle=False,
                                         drop_last=False,
                                         num_workers=config.number_workers,
                                         pin_memory=config.pin_memory)
            print("dbcnn TEST data size{}".format(config.batch_size * len(test_dataloader)))
            return test_dataloader
        elif config.network == 'metal':
            test_data = metal_dataset.IQADataset('../data/kon10k/test2000/test_' + str(config.split),
                                                           '../data/kon10k/1024x768')
            test_dataloader = DataLoader(test_data,
                                         batch_size=1,
                                         shuffle=False,
                                         drop_last=False,
                                         num_workers=config.number_workers,
                                         pin_memory=config.pin_memory)
            print("metal TEST data size{}".format(len(test_dataloader)))
            return test_dataloader
        elif config.network == 'mbcnn':
            test_data = mbcnn_dataset.IQADataset('../data/kon10k/test2000/test_' + str(config.split),
                                                 '../data/kon10k/1024x768',
                                                 '../gradiant_data/1024x768')
            test_dataloader = DataLoader(test_data,
                                         batch_size=1,
                                         shuffle=False,
                                         drop_last=False,
                                         num_workers=config.number_workers,
                                         pin_memory=config.pin_memory)
            print("mbcnn TEST data size{}".format(len(test_dataloader)))
            return test_dataloader
        elif config.network == 'p2p':
            test_data = p2p_dataset.IQADataset('../data/kon10k/test2000/test_' + str(config.split),
                                               '../data/kon10k/1024x768')
            test_dataloader = DataLoader(test_data,
                                         batch_size=1,
                                         shuffle=False,
                                         drop_last=False,
                                         num_workers=config.number_workers,
                                         pin_memory=config.pin_memory)
            print("TEST data size{}".format(len(test_dataloader)))
            return test_dataloader

        elif config.network == 'nssadnn':
            test_data = nssadnn_dataset.IQADataset(
                '../data/kon10k/test2000/test_' + str(config.split),
                '../data/kon10k/1024x768')
            train_dataloader = DataLoader(test_data,
                                          batch_size=1,
                                          shuffle=False,
                                          drop_last=False,
                                          num_workers=config.number_workers,
                                          pin_memory=config.pin_memory)
            print("test data size{}".format(len(train_dataloader)))
            return train_dataloader

    elif config.state == 'NNID':
        test_data = nnid_dataset.IQADataset(
            '../data/kon10k/test2000/test_' + str(config.split),
            '../data/kon10k/1024x768')
        train_dataloader = DataLoader(test_data,
                                      batch_size=1,
                                      shuffle=False,
                                      drop_last=False,
                                      num_workers=config.number_workers,
                                      pin_memory=config.pin_memory)
        print("test data size{}".format(len(train_dataloader)))
        return train_dataloader

    elif config.state == 'LIVEC':
        test_data = livec_dataset.IQADataset(
            '../data/kon10k/test2000/test_' + str(config.split),
            '../data/kon10k/1024x768')
        train_dataloader = DataLoader(test_data,
                                      batch_size=1,
                                      shuffle=False,
                                      drop_last=False,
                                      num_workers=config.number_workers,
                                      pin_memory=config.pin_memory)
        print("test data size{}".format(len(train_dataloader)))
        return train_dataloader

def model_creator(config):
    if config.network == 'cnniqa':
        model = CNNIQA_modify.CNNIQAnet()
        print('cnnqia network chosed')
    elif config.network == 'wad':
        model = WaDIQaM_modify.NRnet()
        print('wadiqam network chosed')
    elif config.network == 'p2p':
        model = Paq2piq_modify.PAQ2PIQ(num_classes=5)
        print('p2p network chosed')
    elif config.network == 'dbcnn':
        model = DBCNN_modify.DBCNN(scnn_root='../model_hub/scnn.pkl', num_classes=5)
        print('dbcnn network chosed')
    elif config.network == 'mbcnn':
        model = MB_CNN_modify.Mbcnn(num_classes=5)
        print('mbcnn network chosed')
    elif config.network == 'metal':
        model = Metal_modify.Net()
        print('METAL_IQA network chosed')
    elif config.network == 'sfa':
        model = SFA_modify.ResNet50()
        print('SFA network chosed')
    elif config.network == 'nssadnn':
        model = NSSADNN_modiify.NSSADNN()
        print('nssadnn network chosed')
    else:
        raise NotImplementedError("Not supported network, need to be added!")

    return model


def optimal_creator(config, model):
    if config.network == 'cnniqa':
        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=0)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                    last_epoch=- 1,
                                                    step_size=config.decay_interval,
                                                    gamma=config.decay_ratio)

        return torch.nn.L1Loss(), optimizer, scheduler

    elif config.network == 'mbcnn':
        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=0)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                    last_epoch=- 1,
                                                    step_size=config.decay_interval,
                                                    gamma=config.decay_ratio)

        return torch.nn.L1Loss(), optimizer, scheduler

    elif config.network == 'dbcnn':
        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=0.00004)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                    last_epoch=- 1,
                                                    step_size=config.decay_interval,
                                                    gamma=config.decay_ratio)

        return torch.nn.MSELoss(), optimizer, scheduler

    elif config.network == 'wad':

        all_params = model.parameters()
        regression_params = []
        for pname, p in model.named_parameters():
            if pname.find('fc') >= 0:
                regression_params.append(p)
        regression_params_id = list(map(id, regression_params))
        features_params = list(filter(lambda p: id(p) not in regression_params_id, all_params))
        optimizer = torch.optim.Adam([{'params': regression_params},
                                      {'params': features_params, 'lr': config.lr * 0.8}],
                                     lr=config.lr, weight_decay=0)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                    last_epoch=- 1,
                                                    step_size=config.decay_interval,
                                                    gamma=config.decay_ratio)

        return torch.nn.L1Loss(), optimizer, scheduler

    elif config.network == 'p2p':
        all_params = model.parameters()
        regression_params = []
        for pname, p in model.named_parameters():
            if pname.find('head') >= 0:
                regression_params.append(p)
        regression_params_id = list(map(id, regression_params))
        features_params = list(filter(lambda p: id(p) not in regression_params_id, all_params))

        optimizer = torch.optim.Adam([{'params': regression_params, 'lr': 0.003},
                                      {'params': features_params, 'lr': 0.0003}], betas=(0.9, 0.99), weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                    last_epoch=- 1,
                                                    step_size=config.decay_interval,
                                                    gamma=config.decay_ratio)

        return torch.nn.MSELoss(), optimizer, scheduler

    elif config.network == 'metal':
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=0)

        return torch.nn.MSELoss(), optimizer

    elif config.network == 'nssadnn':

        optimizer = torch.optim.SGD(model.parameters(), lr=config.lr, weight_decay=0.0001, momentum=0.9)
        scheduler =torch.optim.lr_scheduler.StepLR(optimizer, 750, gamma=0.1, last_epoch=-1)


        return torch.nn.L1Loss(), optimizer, scheduler
