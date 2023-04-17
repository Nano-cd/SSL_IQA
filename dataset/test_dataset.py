import torch
from torch.utils.data import Dataset
import os
from PIL import Image
import torchvision

data_transform_eval = torchvision.transforms.Compose([
    torchvision.transforms.Resize([512, 384]),
    torchvision.transforms.RandomCrop(size=224),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])


class test_data(Dataset):
    def __init__(self, label_file, image_root,patch_num):
        self.imagepaths = []
        self.labels = []
        self.unlabeled_index = []
        self.labeled_index = []
        self.transforms = data_transform_eval

        with open(label_file, 'r') as f:
            for line in f.readlines():  # 读取label文件
                for aug in range(patch_num):
                    self.imagepaths.append(os.path.join(image_root, line.split()[5]))
                    self.labels.append([float(line.split()[0]),float(line.split()[1]),float(line.split()[2])
                                    ,float(line.split()[3]),float(line.split()[4])])

    def __getitem__(self, item):
        x = Image.open(self.imagepaths[item])
        y = torch.tensor(self.labels[item], dtype=torch.float)
        return self.transforms(x), y

    def __len__(self):
        return len(self.imagepaths)
