"""
Main function to build TRIQ.
"""
from torch.nn.modules import batchnorm
from torchvision.models import vgg, vgg16, resnet50
import torch.nn as nn
import torch
from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm
import math
from os.path import join as pjoin
import copy
import ml_collections
import numpy as np
import logging
from scipy import ndimage


# from ResNetV2 import ResNetV2

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


logger = logging.getLogger(__name__)

ATTENTION_Q = "MultiHeadDotProductAttention_1/query"
ATTENTION_K = "MultiHeadDotProductAttention_1/key"
ATTENTION_V = "MultiHeadDotProductAttention_1/value"
ATTENTION_OUT = "MultiHeadDotProductAttention_1/out"
FC_0 = "MlpBlock_3/Dense_0"
FC_1 = "MlpBlock_3/Dense_1"
ATTENTION_NORM = "LayerNorm_0"
MLP_NORM = "LayerNorm_2"


class BackBone(nn.Module):
    def __init__(self, backbone='vgg16'):
        super(BackBone, self).__init__()

        if backbone == 'resnet50':
            self.head_temp = resnet50(pretrained=True)
            # self.head_temp = ([3, 4, 6, 3], 1)
            # self.head_temp.load_from(np.load('pretrained_weights/BiT-M-R50x1-ILSVRC2012.npz'))
        elif backbone == 'vgg16':
            self.head_temp = vgg16(pretrained=True)
        else:
            NotImplementedError

        self.head = nn.Sequential(*(list(self.head_temp.children())[:-1]))

    def forward(self, x):
        return self.head(x)


def np2th(weights, conv=False):
    """Possibly convert HWIO to OIHW."""
    if conv:
        weights = weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(weights)


def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {"gelu": torch.nn.functional.gelu, "relu": torch.nn.functional.relu, "swish": swish}


class MultiHeadAttention(nn.Module):
    def __init__(self, config, vis):
        super(MultiHeadAttention, self).__init__()
        self.vis = vis
        self.num_attention_heads = config.transformer["num_heads"]
        self.attention_head_size = int(config.hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = Linear(config.hidden_size, self.all_head_size)
        self.key = Linear(config.hidden_size, self.all_head_size)
        self.value = Linear(config.hidden_size, self.all_head_size)

        self.out = Linear(config.hidden_size, config.hidden_size)
        self.attn_dropout = Dropout(config.transformer["attention_dropout_rate"])
        self.proj_dropout = Dropout(config.transformer["attention_dropout_rate"])

        self.softmax = Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores)
        weights = attention_probs if self.vis else None
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        return attention_output, weights


class TransformerBlock(nn.Module):
    def __init__(self, config, vis):
        super(TransformerBlock, self).__init__()
        self.hidden_size = config.hidden_size
        self.attention_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn_norm = LayerNorm(config.hidden_size, eps=1e-6)
        # self.ffn = Mlp(config)
        self.ffn = nn.Sequential(
            nn.Linear(self.hidden_size, config.transformer["mlp_dim"]),
            nn.ReLU(inplace=True),
            nn.Linear(config.transformer["mlp_dim"], self.hidden_size)
        )

        self.attn = MultiHeadAttention(config, vis)

        self.dropout1 = nn.Dropout(config.transformer['dropout_rate'])
        self.dropout2 = nn.Dropout(config.transformer['dropout_rate'])

    def forward(self, x):
        h = x
        x = self.attention_norm(x)
        x, weights = self.attn(x)
        x = self.dropout1(x)
        x = x + h

        h = x
        x = self.ffn_norm(x)
        x = self.dropout2(self.ffn(x))
        x = x + h

        return x, weights

    def load_from(self, weights, n_block):
        ROOT = f"Transformer/encoderblock_{n_block}"
        with torch.no_grad():
            query_weight = np2th(weights[pjoin(ROOT, ATTENTION_Q, "kernel").replace('\\', '/')]).view(self.hidden_size,
                                                                                                      self.hidden_size).t()
            key_weight = np2th(weights[pjoin(ROOT, ATTENTION_K, "kernel").replace('\\', '/')]).view(self.hidden_size,
                                                                                                    self.hidden_size).t()
            value_weight = np2th(weights[pjoin(ROOT, ATTENTION_V, "kernel").replace('\\', '/')]).view(self.hidden_size,
                                                                                                      self.hidden_size).t()
            out_weight = np2th(weights[pjoin(ROOT, ATTENTION_OUT, "kernel").replace('\\', '/')]).view(self.hidden_size,
                                                                                                      self.hidden_size).t()

            query_bias = np2th(weights[pjoin(ROOT, ATTENTION_Q, "bias").replace('\\', '/')]).view(-1)
            key_bias = np2th(weights[pjoin(ROOT, ATTENTION_K, "bias").replace('\\', '/')]).view(-1)
            value_bias = np2th(weights[pjoin(ROOT, ATTENTION_V, "bias").replace('\\', '/')]).view(-1)
            out_bias = np2th(weights[pjoin(ROOT, ATTENTION_OUT, "bias").replace('\\', '/')]).view(-1)

            self.attn.query.weight.copy_(query_weight)
            self.attn.key.weight.copy_(key_weight)
            self.attn.value.weight.copy_(value_weight)
            self.attn.out.weight.copy_(out_weight)
            self.attn.query.bias.copy_(query_bias)
            self.attn.key.bias.copy_(key_bias)
            self.attn.value.bias.copy_(value_bias)
            self.attn.out.bias.copy_(out_bias)

            mlp_weight_0 = np2th(weights[pjoin(ROOT, FC_0, "kernel").replace('\\', '/')]).t()
            mlp_weight_1 = np2th(weights[pjoin(ROOT, FC_1, "kernel").replace('\\', '/')]).t()
            mlp_bias_0 = np2th(weights[pjoin(ROOT, FC_0, "bias").replace('\\', '/')]).t()
            mlp_bias_1 = np2th(weights[pjoin(ROOT, FC_1, "bias").replace('\\', '/')]).t()

            self.ffn.fc1.weight.copy_(mlp_weight_0)
            self.ffn.fc2.weight.copy_(mlp_weight_1)
            self.ffn.fc1.bias.copy_(mlp_bias_0)
            self.ffn.fc2.bias.copy_(mlp_bias_1)

            self.attention_norm.weight.copy_(np2th(weights[pjoin(ROOT, ATTENTION_NORM, "scale").replace('\\', '/')]))
            self.attention_norm.bias.copy_(np2th(weights[pjoin(ROOT, ATTENTION_NORM, "bias").replace('\\', '/')]))
            self.ffn_norm.weight.copy_(np2th(weights[pjoin(ROOT, MLP_NORM, "scale").replace('\\', '/')]))
            self.ffn_norm.bias.copy_(np2th(weights[pjoin(ROOT, MLP_NORM, "bias").replace('\\', '/')]))


class Mlp(nn.Module):
    def __init__(self, config):
        super(Mlp, self).__init__()
        self.n_quality_levels = config.n_quality_levels

        self.fc1 = Linear(config.hidden_size, config.transformer["mlp_dim"])
        # self.fc2 = Linear(config.transformer["mlp_dim"], config.hidden_size)
        self.fc2 = Linear(config.transformer["mlp_dim"], self.n_quality_levels)
        self.act_fn = ACT2FN["gelu"]
        self.dropout = Dropout(config.transformer["dropout_rate"])

        self.softmax = nn.Softmax(dim=1)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        # print(x)
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        # x = self.dropout(x)
        ### CrossEntropyLoss in pytorch not require softmax ####
        if self.n_quality_levels > 1:
            x = self.softmax(x)
        return x


class Transformer(nn.Module):
    def __init__(self, config, vis):
        super(Transformer, self).__init__()
        self.vis = vis
        self.layer = nn.ModuleList()
        self.encoder_norm = LayerNorm(config.hidden_size, eps=1e-6)
        for _ in range(config.transformer["num_layers"]):
            layer = TransformerBlock(config, vis)
            self.layer.append(copy.deepcopy(layer))

    def forward(self, hidden_states):
        attn_weights = []
        for layer_block in self.layer:
            hidden_states, weights = layer_block(hidden_states)
            if self.vis:
                attn_weights.append(weights)
        encoded = self.encoder_norm(hidden_states)

        if self.vis:
            return encoded, attn_weights
        else:
            return encoded


class TriQImageQualityTransformer(nn.Module):
    def __init__(
            self,
            config,
            zero_head=False,
            maximum_position_encoding=193,
            vis=False):
        super(TriQImageQualityTransformer, self).__init__()

        self.zero_head = zero_head
        self.classifier = config.classifier
        self.vis = vis
        self.n_quality_levels = config.n_quality_levels

        self.d_model = config.hidden_size

        self.backbone_model = BackBone(backbone=config.backbone)

        # Middle Part
        self.pos_emb = nn.Parameter(torch.empty(1, maximum_position_encoding, self.d_model))
        self.quality_emb = nn.Parameter(torch.empty(1, 1, self.d_model))

        self.feature_proj_conv = nn.Conv2d(2048, self.d_model, 1, 1)
        self.pooling_small = nn.MaxPool2d((2, 2))

        self.dropout = nn.Dropout(config.transformer['dropout_rate'])

        #### Transformer Part ####
        self.transformer = Transformer(config, vis)
        self.mlp_head = Mlp(config)

    def forward(self, x, labels=None):
        batch_size = x.shape[0]

        x = self.backbone_model(x)  # [-1, 2048, 1, 1]
        # print("Shape after ResNet: ", x.shape)

        x = self.feature_proj_conv(x)  # [-1, 32, 1, 1]
        if x.shape[2] >= 16:
            x = self.pooling_small(x)

        spatial_size = x.shape[2] * x.shape[3]
        x = x.view(batch_size, spatial_size, self.d_model)

        # Modify pos and quality emb
        # quality_emb = self.quality_emb.repeat(batch_size, 1, 1)
        # x = torch.cat([quality_emb, x], dim=1)

        # truncate the positional embedding for shorter videos
        # x = x + self.pos_emb[:, : spatial_size + 1, :]

        x = self.dropout(x)

        # print("Shape before transformer: ", x.shape)

        #### Feeding to transformer ####
        if self.vis:
            output, attn_weights = self.Transformer(x)
        else:
            output = self.transformer(x)

        # print("Shape after transformer: ", output.shape)

        logits = self.mlp_head(output[:, 0])

        # if labels is not None:
        #     loss_fct = CrossEntropyLoss()
        #     #tmp1 = logits.view(-1, self.n_quality_levels)
        #     #tmp2 = labels.view(-1)
        #     loss = loss_fct(logits.view(-1, self.n_quality_levels), labels.view(-1))
        #     return loss
        # else:
        #     return logits, attn_weights

        if self.vis:
            return logits, attn_weights
        else:
            return logits

    def load_from(self, weights):
        with torch.no_grad():
            if self.zero_head:
                nn.init.zeros_(self.head.weight)
                nn.init.zeros_(self.head.bias)
            else:
                self.head.weight.copy_(np2th(weights["head/kernel"]).t())
                self.head.bias.copy_(np2th(weights["head/bias"]).t())

            self.transformer.embeddings.patch_embeddings.weight.copy_(np2th(weights["embedding/kernel"], conv=True))
            self.transformer.embeddings.patch_embeddings.bias.copy_(np2th(weights["embedding/bias"]))
            self.transformer.embeddings.cls_token.copy_(np2th(weights["cls"]))
            self.transformer.encoder.encoder_norm.weight.copy_(np2th(weights["Transformer/encoder_norm/scale"]))
            self.transformer.encoder.encoder_norm.bias.copy_(np2th(weights["Transformer/encoder_norm/bias"]))

            posemb = np2th(weights["Transformer/posembed_input/pos_embedding"])
            posemb_new = self.transformer.embeddings.position_embeddings
            if posemb.size() == posemb_new.size():
                self.transformer.embeddings.position_embeddings.copy_(posemb)
            else:
                logger.info("load_pretrained: resized variant: %s to %s" % (posemb.size(), posemb_new.size()))
                ntok_new = posemb_new.size(1)

                if self.classifier == "token":
                    posemb_tok, posemb_grid = posemb[:, :1], posemb[0, 1:]
                    ntok_new -= 1
                else:
                    posemb_tok, posemb_grid = posemb[:, :0], posemb[0]

                gs_old = int(np.sqrt(len(posemb_grid)))
                gs_new = int(np.sqrt(ntok_new))
                print('load_pretrained: grid-size from %s to %s' % (gs_old, gs_new))
                posemb_grid = posemb_grid.reshape(gs_old, gs_old, -1)

                zoom = (gs_new / gs_old, gs_new / gs_old, 1)
                posemb_grid = ndimage.zoom(posemb_grid, zoom, order=1)
                posemb_grid = posemb_grid.reshape(1, gs_new * gs_new, -1)
                posemb = np.concatenate([posemb_tok, posemb_grid], axis=1)
                self.transformer.embeddings.position_embeddings.copy_(np2th(posemb))

            for bname, block in self.transformer.encoder.named_children():
                for uname, unit in block.named_children():
                    unit.load_from(weights, n_block=uname)

            if self.transformer.embeddings.hybrid:
                self.transformer.embeddings.hybrid_model.root.conv.weight.copy_(
                    np2th(weights["conv_root/kernel"], conv=True))
                gn_weight = np2th(weights["gn_root/scale"]).view(-1)
                gn_bias = np2th(weights["gn_root/bias"]).view(-1)
                self.transformer.embeddings.hybrid_model.root.gn.weight.copy_(gn_weight)
                self.transformer.embeddings.hybrid_model.root.gn.bias.copy_(gn_bias)

                for bname, block in self.transformer.embeddings.hybrid_model.body.named_children():
                    for uname, unit in block.named_children():
                        unit.load_from(weights, n_block=bname, n_unit=uname)

# if __name__ == '__main__':
#     base_config = get_triq_config()
#
#     input = torch.randn(16, 3, 224, 224)
#
#     model = TriQImageQualityTransformer(config=base_config)
#
#     output = model(input)
#     # print(model)
#     print("Output shape: ", output.shape)
