import torch
from fvcore.nn import FlopCountAnalysis, parameter_count_table
from model_hub import Google_net_modify, Resnet101_modify, Resnet18_modify, Alexnet_modify, Resnet50_modify
from model_hub import Inceptionresnetv2_modify, swin_vit_modify, Tres_modify,densent_modify


def print_model_parm_nums(model):
    total = sum([param.nelement() for param in model.parameters()])
    print('  + Number of params: %.2fM' % (total / 1e6))


model1 = densent_modify.Densenet_121(num_classes=5)
test_input = torch.rand((1, 3, 224, 224))
flops = FlopCountAnalysis(model1, test_input)
print("FLOPs: ", flops.total() / 1000000)
print_model_parm_nums(model1)
