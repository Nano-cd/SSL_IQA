r"""Paq2piq metric, proposed by
Ying, Zhenqiang, Haoran Niu, Praful Gupta, Dhruv Mahajan, Deepti Ghadiyaram, and Alan Bovik.
"From patches to pictures (PaQ-2-PiQ): Mapping the perceptual space of picture quality."
In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 3575-3585. 2020.
Ref url: https://github.com/baidut/paq2piq/blob/master/paq2piq/model.py
Modified by: Chaofeng Chen (https://github.com/chaofengc)
"""

import torch
import torch.nn as nn
import torchvision as tv
from torchvision.ops import RoIPool

default_model_urls = {
    'url':
        'https://github.com/chaofengc/IQA-PyTorch/releases/download/v0.1-weights/'
        'P2P_RoIPoolModel-fit.10.bs.120-ca69882e.pth',
}


class AdaptiveConcatPool2d(nn.Module):

    def __init__(self, sz=None):
        super().__init__()
        sz = sz or (1, 1)
        self.ap = nn.AdaptiveAvgPool2d(sz)
        self.mp = nn.AdaptiveMaxPool2d(sz)

    def forward(self, x):
        return torch.cat([self.mp(x), self.ap(x)], 1)


class PAQ2PIQ(nn.Module):

    def __init__(self, num_classes):
        super(PAQ2PIQ, self).__init__()

        model = tv.models.resnet18(pretrained=True)
        cut = -2
        spatial_scale = 1 / 32

        self.blk_size = 20, 20

        self.model_type = self.__class__.__name__
        self.body = nn.Sequential(*list(model.children())[:cut])
        self.head = nn.Sequential(AdaptiveConcatPool2d(), nn.Flatten(),
                                  nn.BatchNorm1d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                  nn.Dropout(p=0.25, inplace=False),
                                  nn.Linear(in_features=1024, out_features=512, bias=True), nn.ReLU(inplace=True),
                                  nn.BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                  nn.Dropout(p=0.5, inplace=False),
                                  nn.Linear(in_features=512, out_features=5, bias=True),
                                  nn.Softmax(dim=1))

        self.roi_pool = RoIPool((2, 2), spatial_scale)

    def forward(self, x):
        im_data = x
        batch_size = im_data.shape[0]

        feats = self.body(im_data)
        global_rois = torch.tensor([0, 0, x.shape[-1], x.shape[-2]]).reshape(1, 4).to(x)
        feats = self.roi_pool(feats, [global_rois] * batch_size)

        preds = self.head(feats)
        return preds.view(batch_size, -1)
