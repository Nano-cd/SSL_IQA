from torch.utils.data import DataLoader
import dataset.train_dataset
import dataset.test_dataset
import dataset.LIVE_dataset
import dataset.NNID_dataset



def kon10k_1000(config):
    print("kon10k 1000 bullied successfully   " + str(config.split))
    if config.data_mode == 'train':
        train_data = dataset.train_dataset.train_data('./data/kon10k/train1000/labeled/labeled_' + str(config.split),
                                                      './data/kon10k/1024x768')
        train_dataloader = DataLoader(train_data,
                                      batch_size=config.batch_size,
                                      shuffle=True,
                                      drop_last=True,
                                      num_workers=config.number_workers,
                                      pin_memory=config.pin_memory)
        print("train data size{}".format(config.batch_size * len(train_dataloader)))
        return train_dataloader

    if config.data_mode == 'semi-train':
        train_data = dataset.train_dataset.train_data('./data/kon10k/train1000/labeled/labeled_'
                                                      + str(config.split),
                                                      './data/kon10k/1024x768')
        train_dataloader = DataLoader(train_data,
                                      batch_size=config.batch_size,
                                      shuffle=True,
                                      drop_last=True,
                                      num_workers=config.number_workers,
                                      pin_memory=config.pin_memory)
        print("train data size{}".format(config.batch_size * len(train_dataloader)))

        train_data_unlabeled = dataset.train_dataset.train_data(
            './data/kon10k/train1000/unlabeled/unlabeled_' + str(config.split),
            './data/kon10k/1024x768')
        train_dataloader_unlabeled = DataLoader(train_data_unlabeled,
                                                batch_size=config.batch_size*3,
                                                shuffle=True,
                                                drop_last=True,
                                                num_workers=config.number_workers,
                                                pin_memory=config.pin_memory)
        print("train data unlabeled size{}".format(config.batch_size * len(train_dataloader) * 2))
        return train_dataloader, train_dataloader_unlabeled

    if config.data_mode == 'test':
        test_data = dataset.test_dataset.test_data('./data/kon10k/test1000/test_' + str(config.split),
                                                   './data/kon10k/1024x768')
        test_dataloader = DataLoader(test_data,
                                     batch_size=1,
                                     shuffle=False,
                                     drop_last=False,
                                     num_workers=config.number_workers,
                                     pin_memory=False)
        print("test data size{}".format(len(test_dataloader)))
        return test_dataloader


def kon10k_2000(config):
    print("kon10k 2000 bullied successfully   " + str(config.split))
    if config.data_mode == 'train':
        train_data = dataset.train_dataset.train_data('./data/kon10k/train2000/labeled/labeled_'
                                                      + str(config.split),
                                                      './data/kon10k/1024x768', 10)
        train_dataloader = DataLoader(train_data,
                                      batch_size=config.batch_size,
                                      shuffle=True,
                                      drop_last=True,
                                      num_workers=config.number_workers,
                                      pin_memory=config.pin_memory)
        print("train data size{}".format(config.batch_size * len(train_dataloader)))
        return train_dataloader

    if config.data_mode == 'semi-train':
        train_data = dataset.train_dataset.train_data('./data/kon10k/train2000/labeled/labeled_'
                                                      + str(config.split),
                                                      './data/kon10k/1024x768',10)
        train_dataloader = DataLoader(train_data,
                                      batch_size=config.batch_size,
                                      shuffle=True,
                                      drop_last=True,
                                      num_workers=config.number_workers,
                                      pin_memory=config.pin_memory)
        print("train data size{}".format(config.batch_size * len(train_dataloader)))

        train_data_unlabeled = dataset.train_dataset.train_data(
            './data/kon10k/train2000/unlabeled/unlabeled_' + str(config.split),
            './data/kon10k/1024x768',10)
        train_dataloader_unlabeled = DataLoader(train_data_unlabeled,
                                                batch_size=config.batch_size*config.ratio,
                                                shuffle=True,
                                                drop_last=True,
                                                num_workers=config.number_workers,
                                                pin_memory=config.pin_memory)
        print("train data unlabeled size{}".format(config.batch_size * len(train_dataloader) * config.ratio))
        return train_dataloader, train_dataloader_unlabeled

    if config.data_mode == 'test':
        test_data = dataset.test_dataset.test_data('./data/kon10k/test2000/test_' + str(config.split),
                                                   './data/kon10k/1024x768',10)
        test_dataloader = DataLoader(test_data,
                                     batch_size=1,
                                     shuffle=False,
                                     drop_last=False,
                                     num_workers=config.number_workers,
                                     pin_memory=False)
        print("test data size{}".format(len(test_dataloader)))
        return test_dataloader


def kon10k_3000(config):
    print("kon10k 3000 bullied successfully   " + str(config.split))
    if config.data_mode == 'train':
        train_data = dataset.train_dataset.train_data('./data/kon10k/train3000/labeled/labeled_' + str(config.split),
                                                      './data/kon10k/1024x768')
        train_dataloader = DataLoader(train_data,
                                      batch_size=config.batch_size,
                                      shuffle=True,
                                      drop_last=True,
                                      num_workers=config.number_workers,
                                      pin_memory=config.pin_memory)
        print("train data size{}".format(config.batch_size * len(train_dataloader)))
        return train_dataloader

    if config.data_mode == 'test':
        test_data = dataset.test_dataset.test_data('./data/kon10k/test3000/test_' + str(config.split),
                                                   './data/kon10k/1024x768')
        test_dataloader = DataLoader(test_data,
                                     batch_size=1,
                                     shuffle=False,
                                     drop_last=False,
                                     num_workers=config.number_workers,
                                     pin_memory=False)
        print("test data size{}".format(len(test_dataloader)))
        return test_dataloader


def kon10k_8000(config):
    print("kon10k 8000 bullied successfully  " + str(config.split))
    if config.data_mode == 'train':
        train_data = dataset.train_dataset.train_data('./data/kon10k/train2000/train_' + str(config.split),
                                                      './data/kon10k/1024x768',10)
        train_dataloader = DataLoader(train_data,
                                      batch_size=config.batch_size,
                                      shuffle=True,
                                      drop_last=True,
                                      num_workers=config.number_workers,
                                      pin_memory=config.pin_memory)
        print("train data size{}".format(config.batch_size * len(train_dataloader)))
        return train_dataloader

    if config.data_mode == 'test':
        test_data = dataset.test_dataset.test_data('./data/kon10k/test8000/test_' + str(config.split),
                                                   './data/kon10k/1024x768')
        test_dataloader = DataLoader(test_data,
                                     batch_size=1,
                                     shuffle=False,
                                     drop_last=False,
                                     num_workers=config.number_workers,
                                     pin_memory=False)
        print("test data size{}".format(len(test_dataloader)))
        return test_dataloader


def kadid10k_1000(config):
    print("kadid10k 1000 bullied successfully")
    if config.data_mode == 'train':
        train_data = dataset.train_dataset.train_data('./data/kadid10k/train1000/labeled/labeled_' + str(config.split) + '.txt',
                                                      './data/kadid10k/images')
        train_dataloader = DataLoader(train_data, batch_size=config.min_batch_size,
                                      shuffle=True, drop_last=True, num_workers=4, pin_memory=True)
        print("train data size{}".format(config.min_batch_size * len(train_dataloader)))
        return train_dataloader


    if config.data_mode == 'semi-train':
        train_data = dataset.train_dataset.train_data('./data/kadid10k/train1000/labeled/labeled_'
                                                      + str(config.split)+'.txt',
                                                      './data/kadid10k/images')
        train_dataloader = DataLoader(train_data,
                                      batch_size=config.batch_size,
                                      shuffle=True,
                                      drop_last=True,
                                      num_workers=config.number_workers,
                                      pin_memory=config.pin_memory)
        print("train data size{}".format(config.batch_size * len(train_dataloader)))

        train_data_unlabeled = dataset.train_dataset.train_data(
            './data/kadid10k/train1000/unlabeled/unlabeled_' + str(config.split)+'.txt',
            './data/kadid10k/images')
        train_dataloader_unlabeled = DataLoader(train_data_unlabeled,
                                                batch_size=config.batch_size*3,
                                                shuffle=True,
                                                drop_last=True,
                                                num_workers=config.number_workers,
                                                pin_memory=config.pin_memory)
        print("train data unlabeled size{}".format(config.batch_size * len(train_dataloader) * 3))
        return train_dataloader, train_dataloader_unlabeled

    if config.data_mode == 'test':
        test_data = dataset.test_dataset.test_data('./data/kadid10k/test1000/test_' + str(config.split) + '.txt',
                                                   './data/kadid10k/images')
        test_dataloader = DataLoader(test_data,
                                     batch_size=1,
                                     shuffle=False,
                                     drop_last=False,
                                     num_workers=config.number_workers,
                                     pin_memory=False)
        print("test data size{}".format(len(test_dataloader)))
        return test_dataloader


def kadid10k_2000(config):
    print("kadid10k 2000 bullied successfully")
    if config.data_mode == 'train':
        train_data = dataset.train_dataset.train_data('./data/kadid10k/train2000/labeled/labeled_' + str(config.split) + '.txt',
                                                      './data/kadid10k/images')
        train_dataloader = DataLoader(train_data, batch_size=config.min_batch_size,
                                      shuffle=True, drop_last=True, num_workers=4, pin_memory=True)
        print("train data size{}".format(config.min_batch_size * len(train_dataloader)))
        return train_dataloader


    if config.data_mode == 'semi-train':
        train_data = dataset.train_dataset.train_data('./data/kadid10k/train2000/labeled/labeled_'
                                                      + str(config.split)+'.txt',
                                                      './data/kadid10k/images')
        train_dataloader = DataLoader(train_data,
                                      batch_size=config.batch_size,
                                      shuffle=True,
                                      drop_last=True,
                                      num_workers=config.number_workers,
                                      pin_memory=config.pin_memory)
        print("train data size{}".format(config.batch_size * len(train_dataloader)))

        train_data_unlabeled = dataset.train_dataset.train_data(
            './data/kadid10k/train2000/unlabeled/unlabeled_' + str(config.split)+'.txt',
            './data/kadid10k/images')
        train_dataloader_unlabeled = DataLoader(train_data_unlabeled,
                                                batch_size=config.batch_size*3,
                                                shuffle=True,
                                                drop_last=True,
                                                num_workers=config.number_workers,
                                                pin_memory=config.pin_memory)
        print("train data unlabeled size{}".format(config.batch_size * len(train_dataloader) * 3))
        return train_dataloader, train_dataloader_unlabeled

    if config.data_mode == 'test':
        test_data = dataset.test_dataset.test_data('./data/kadid10k/test2000/test_' + str(config.split) + '.txt',
                                                   './data/kadid10k/images')
        test_dataloader = DataLoader(test_data,
                                     batch_size=10,
                                     shuffle=False,
                                     drop_last=True,
                                     num_workers=config.number_workers,
                                     pin_memory=False)
        print("test data size{}".format(len(test_dataloader)))
        return test_dataloader


def kadid10k_3000(config):
    print("kadid10k 3000 bullied successfully")
    if config.data_mode == 'train':
        train_data = dataset.train_dataset.train_data(
            './data/kadid10k/train3000/train_' + str(config.split) + '.txt',
            './data/kadid10k/images')
        train_dataloader = DataLoader(train_data, batch_size=config.min_batch_size,
                                      shuffle=True, drop_last=True, num_workers=4, pin_memory=True)
        print("train data size{}".format(config.min_batch_size * len(train_dataloader)))
        return train_dataloader

    if config.data_mode == 'test':
        test_data = dataset.test_dataset.test_data('./data/kadid10k/test3000/test_' + str(config.split) + '.txt',
                                                   './data/kadid10k/images')
        test_dataloader = DataLoader(test_data,
                                     batch_size=1,
                                     shuffle=False,
                                     drop_last=False,
                                     num_workers=config.number_workers,
                                     pin_memory=False)
        print("test data size{}".format(len(test_dataloader)))
        return test_dataloader


def kadid10k_8000(config):
    print("kadid10k 8000 bullied successfully")
    if config.data_mode == 'train':
        train_data = dataset.train_dataset.train_data(
            './data/kadid10k/train8000/train_' + str(config.split) + '.txt',
            './data/kadid10k/images')
        train_dataloader = DataLoader(train_data, batch_size=config.min_batch_size,
                                      shuffle=True, drop_last=True, num_workers=4, pin_memory=True)
        print("train data size{}".format(config.min_batch_size * len(train_dataloader)))
        return train_dataloader

    if config.data_mode == 'test':
        test_data = dataset.test_dataset.test_data('./data/kadid10k/test8000/test_' + str(config.split) + '.txt',
                                                   './data/kadid10k/images')
        test_dataloader = DataLoader(test_data,
                                     batch_size=1,
                                     shuffle=False,
                                     drop_last=False,
                                     num_workers=config.number_workers,
                                     pin_memory=False)
        print("test data size{}".format(len(test_dataloader)))
        return test_dataloader


def live_c(config):
    if config.data_mode == 'test':
        test_data = dataset.LIVE_dataset.test_data('./data/LIVE_C/labeled.txt',
                                                   './data/LIVE_C/Images')
        test_dataloader = DataLoader(test_data,
                                     batch_size=1,
                                     shuffle=False,
                                     drop_last=False,
                                     num_workers=config.number_workers,
                                     pin_memory=False)
        print("test data size{}".format(len(test_dataloader)))
        return test_dataloader


def NNID(config):
    if config.data_mode == 'test':
        test_data1 = dataset.NNID_dataset.test_data('./data/NNID/mos512_with_names.txt',
                                                   './data/NNID/sub512', 10)
        test_dataloader1 = DataLoader(test_data1,
                                     batch_size=1,
                                     shuffle=False,
                                     drop_last=False,
                                     num_workers=config.number_workers,
                                     pin_memory=False)
        print("test data size{}".format(len(test_dataloader1)))

        test_data2 = dataset.NNID_dataset.test_data('./data/NNID/mos1024_with_names.txt',
                                                   './data/NNID/sub1024', 10)
        test_dataloader2 = DataLoader(test_data2,
                                     batch_size=1,
                                     shuffle=False,
                                     drop_last=False,
                                     num_workers=config.number_workers,
                                     pin_memory=False)
        print("test data size{}".format(len(test_dataloader2)))

        test_data3 = dataset.NNID_dataset.test_data('./data/NNID/mos2048_with_names.txt',
                                                   './data/NNID/Sub2048', 10)
        test_dataloader3 = DataLoader(test_data3,
                                     batch_size=1,
                                     shuffle=False,
                                     drop_last=False,
                                     num_workers=config.number_workers,
                                     pin_memory=False)
        print("test data size{}".format(len(test_dataloader3)))
        return test_dataloader1,test_dataloader2,test_dataloader3
