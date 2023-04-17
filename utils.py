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
                           p0=param_init, ftol=1e-8, maxfev=6000)
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
    # print(y_pred.size)
    # print(y.size)
    MSE = mean_squared_error
    RMSE = MSE(y_pred, y) ** 0.5
    PLCC = stats.pearsonr(convert_obj_score(y_pred, y), y)[0]
    SROCC = stats.spearmanr(y_pred, y)[0]
    KROCC = stats.kendalltau(y_pred, y)[0]

    return RMSE, PLCC, SROCC, KROCC


def split_file():
    """
    func:
        split the train dataset and test dataset randomly and
        save it as txt.
            '1','2','3','4','5',
    """
    for i in ('1','2','3','4','5','6','7','8','9','10'):
        lines = []
        split_train = open('./data/kon10k/train2000/labeled/labeled_' + str(i), 'a')
        split_test = open('./data/kon10k/train2000/unlabeled/unlabeled_' + str(i), 'a')
        with open('data/kon10k/train2000/train_'+str(i), 'r') as infile:
            for line in infile:
                lines.append(line)
        # split_train = open('./data/kon10k/train2000/train_' + str(i), 'a')
        # split_test = open('./data/kon10k/test2000/test_' + str(i), 'a')
        # with open('data/kon10k_dis.txt', 'r') as infile:
        #     for line in infile:
        #         lines.append(line)
        # random.shuffle(lines)

        # length_train = int(len(lines) * 0.125)
        # length_test = int(len(lines) * 0.2)
        length_train = int(2000)
        length_test = int(2000)
        split_train.write(''.join(lines[:length_train]))
        split_test.write(''.join(lines[length_test:]))
        split_train.close()
        split_test.close()
    return

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


def relation_mse_loss(activations, ema_activations):
    """Takes softmax on both sides and returns MSE loss

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """

    # assert activations.size() == ema_activations.size()

    activations = torch.reshape(activations, (activations.shape[0], -1))
    ema_activations = torch.reshape(ema_activations, (ema_activations.shape[0], -1))

    similarity = activations.mm(activations.t())
    ema_similarity = ema_activations.mm(ema_activations.t())
    norm = torch.reshape(torch.norm(similarity, 2, 1), (-1, 1))
    ema_norm = torch.reshape(torch.norm(ema_similarity, 2, 1), (-1, 1))
    norm_similarity = similarity / norm
    norm_ema_similarity = ema_similarity / ema_norm

    # dia = torch.diagonal(norm_similarity)
    # ema_dia = torch.diagonal(norm_ema_similarity)

    # plt.matshow(similarity.detach().cpu())
    # plt.matshow(ema_similarity.detach().cpu())
    # plt.matshow(norm_similarity.detach().cpu())
    # plt.matshow(norm_ema_similarity.detach().cpu())
    r_mse = (norm_similarity - norm_ema_similarity) ** 2
    # r_mse = torch.cosine_similarity(norm_similarity, norm_ema_similarity, dim=0).unsqueeze(0)
    # plt.matshow(r_mse.detach().cpu())
    return torch.sum(r_mse) / activations.size(0)


def relative_rank_loss(unlabel_predict, ema_unlabel_predict):
    indexlabel = torch.argsort(ema_unlabel_predict, dim=0)  # small--> large
    anchor1 = torch.unsqueeze(unlabel_predict[indexlabel[0], ...].contiguous(), dim=0)  # d_min
    positive1 = torch.unsqueeze(unlabel_predict[indexlabel[1], ...].contiguous(), dim=0)  # d'_min+
    negative1_1 = torch.unsqueeze(unlabel_predict[indexlabel[-1], ...].contiguous(), dim=0)  # d_max+

    anchor2 = torch.unsqueeze(unlabel_predict[indexlabel[-1], ...].contiguous(), dim=0)  # d_max
    positive2 = torch.unsqueeze(unlabel_predict[indexlabel[-2], ...].contiguous(), dim=0)  # d'_max+
    negative2_1 = torch.unsqueeze(unlabel_predict[indexlabel[0], ...].contiguous(), dim=0)  # d_min+

    triplet_loss1 = nn.TripletMarginLoss(
        margin=(ema_unlabel_predict[indexlabel[-1]].item() - ema_unlabel_predict[indexlabel[1]].item()), p=1)
    # d_min,d'_min,d_max
    triplet_loss2 = nn.TripletMarginLoss(
        margin=(ema_unlabel_predict[indexlabel[-2]].item() - ema_unlabel_predict[indexlabel[0]].item()), p=1)

    tripletlosses = triplet_loss1(anchor1, positive1, negative1_1) + triplet_loss2(anchor2, positive2, negative2_1)

    return tripletlosses


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


class RkdDistance(nn.Module):
    def forward(self, student, teacher):
        with torch.no_grad():
            t_d = pdist(teacher, squared=False)
            mean_td = t_d[t_d > 0].mean()
            t_d = t_d / mean_td

        d = pdist(student, squared=False)
        mean_d = d[d > 0].mean()
        d = d / mean_d

        loss = F.smooth_l1_loss(d, t_d, reduction='mean')
        return loss


class RKdAngle(nn.Module):
    def forward(self, student, teacher):
        # N x C
        # N x N x C

        with torch.no_grad():
            td = (teacher.unsqueeze(0) - teacher.unsqueeze(1))
            norm_td = F.normalize(td, p=2, dim=2)
            t_angle = torch.bmm(norm_td, norm_td.transpose(1, 2)).view(-1)

        sd = (student.unsqueeze(0) - student.unsqueeze(1))
        norm_sd = F.normalize(sd, p=2, dim=2)
        s_angle = torch.bmm(norm_sd, norm_sd.transpose(1, 2)).view(-1)

        loss = F.smooth_l1_loss(s_angle, t_angle, reduction='mean')
        return loss

def get_triq_config():
    """Returns the ViT-B/16 configuration."""
    config = ml_collections.ConfigDict()
    config.backbone = 'resnet50'
    config.hidden_size = 32
    config.n_quality_levels = 5
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 64
    config.transformer.num_heads = 8
    config.transformer.num_layers = 2
    config.transformer.attention_dropout_rate = 0.0
    config.transformer.dropout_rate = 0.1
    config.classifier = 'token'
    config.representation_size = None
    return config


