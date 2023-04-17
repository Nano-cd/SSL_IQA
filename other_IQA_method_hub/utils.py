import ml_collections
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import random

import torchvision
from matplotlib import pyplot as plt
from scipy import stats
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error
from torchsummary import summary


def make_png(att, scale):
    """
    func:
        unsampled the features into 3 channel pic by calculate the mean of the channel
    """
    samlper = nn.UpsamplingBilinear2d(scale_factor=scale)
    att_current = samlper(att)
    att_current = F.relu(att_current, inplace=True)
    att_current = torch.mean(att_current, dim=1)
    att_current = torch.stack([att_current, att_current, att_current], dim=1)
    return att_current


def convert_obj_score(ori_obj_score, MOS):
    """
    func:
        fitting the objetive score to the MOS scale.
        nonlinear regression fit
    """

    def logistic_fun(x, a, b, c, d):
        return (a - b) / (1 + np.exp(-(x - c) / abs(d))) + b

    # nolinear fit the MOSp
    param_init = [np.max(MOS), np.min(MOS), np.mean(ori_obj_score), 1]
    popt, pcov = curve_fit(logistic_fun, ori_obj_score, MOS,
                           p0=param_init, ftol=1e-8, maxfev=8000)
    # a, b, c, d = popt[0], popt[1], popt[2], popt[3]

    obj_fit_score = logistic_fun(ori_obj_score, popt[0], popt[1], popt[2], popt[3])

    return obj_fit_score


def compute_metric(y, y_pred):
    """
    func:
        calculate the sorcc etc
    """
    index_to_del = []
    for i in range(len(y_pred)):
        if y_pred[i] <= 0:
            print("your prediction seems like not quit good, we reconmand you remove it   ", y_pred[i])
            index_to_del.append(i)
    # for i in index_to_del:
    #     y_pred = np.delete(y_pred, i)
    #     y = np.delete(y, i)
    MSE = mean_squared_error
    RMSE = MSE(y_pred, y) ** 0.5
    PLCC = stats.pearsonr(convert_obj_score(y_pred, y), y)[0]
    SROCC = stats.spearmanr(y_pred, y)[0]
    KROCC = stats.kendalltau(y_pred, y)[0]

    return RMSE, PLCC, SROCC, KROCC


def model_parameter(model):
    """
    func:
        use torchsummary to summary the model parameters

    """
    summary(model, input_size=(3, 512, 384), batch_size=-1)
    return


def features_dis(map1, map2, images, bs):
    inv_normalize = torchvision.transforms.Normalize(
        mean=(-2.118, -2.036, -1.804),
        std=(4.367, 4.464, 4.444))
    att_map1 = make_png(map1, 32).permute(0, 2, 3, 1)
    att_map2 = make_png(map2, 32).permute(0, 2, 3, 1)
    images_flip = torch.flip(inv_normalize(images), [3]).permute(0, 2, 3, 1)
    images = inv_normalize(images).permute(0, 2, 3, 1)
    for j in range(bs):
        plt.subplot(2, 2, 1)
        plt.imshow(images[j].cpu().numpy())
        plt.imshow(att_map1[j].cpu().numpy()[:, :, 0], cmap=plt.cm.jet, alpha=0.4)
        # plt.colorbar()
        plt.subplot(2, 2, 2)
        plt.imshow(images_flip[j].cpu().numpy())
        plt.imshow(att_map2[j].cpu().numpy()[:, :, 0], cmap=plt.cm.jet, alpha=0.4)
        plt.subplot(2, 2, 3)
        plt.imshow(images[j].cpu().numpy())
        plt.subplot(2, 2, 4)
        plt.imshow(images_flip[j].cpu().numpy())
        plt.show()
    return


def plot_scatter(y_pred, y_):
    t = np.arctan2(np.array(y_pred), np.array(y_))
    plt.scatter(np.array(y_pred), np.array(y_), alpha=0.5, c=t, marker='.')
    plt.xlim(0, 5)
    plt.ylim(0, 5)
    plt.show()

    return


def dif_loss(activation, pre, ema_activation, ema_pre):
    """Takes each mini-batches predictions and activations, calculate their differences

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    pre_sort = torch.argsort(pre, dim=0)
    # ema_pre_sort = torch.argsort(ema_pre, dim=0)
    # ema_index_min = ema_pre_sort[0]
    # ema_index_max = ema_pre_sort[-1]
    index_min = pre_sort[0]
    index_max = pre_sort[-1]
    activation_min = activation[index_min]
    activation_max = activation[index_max]
    ema_activation_max = ema_activation[index_max]
    ema_activation_min = ema_activation[index_min]
    # part 1
    # dif_features = activation_max - activation_min
    # dif_ema_features = ema_activation_max - ema_activation_min
    # dif_features = dif_features.mm(dif_features.t())/256
    # dif_ema_features = dif_ema_features.mm(dif_ema_features.t())/1280
    # loss_consistency1 = F.mse_loss(dif_features, dif_ema_features)

    # part 2
    pre_dif = pre[index_max] - pre[index_min]
    ema_pre_dif = ema_pre[index_max] - ema_pre[index_min]
    loss_consistency2 = F.mse_loss(pre_dif, ema_pre_dif)

    loss_all = loss_consistency2
    return loss_all


def pdist(e, squared=False, eps=1e-12):
    e_square = e.pow(2).sum(dim=1)
    prod = e @ e.t()
    res = (e_square.unsqueeze(1) + e_square.unsqueeze(0) - 2 * prod).clamp(min=eps)

    if not squared:
        res = res.sqrt()

    res = res.clone()
    res[range(len(e)), range(len(e))] = 0
    return res


