import torch
import torchvision
from torch import nn
import torch.nn.functional as F


class Resnet(nn.Module):
    def __init__(self, region_num, v_embedding, weights=torchvision.models.ResNet101_Weights.DEFAULT):
        super(Resnet, self).__init__()
        resnet = torchvision.models.resnet101(weights=weights)
        self.region_num = region_num
        self.v_embedding = v_embedding
        self.resnet = nn.Sequential(*list(resnet.children())[:-2])
        self.pooling = nn.Sequential(*list(resnet.children())[-2:-1])

    def forward(self, x):
        x = self.resnet(x)
        local_feature = x
        local_feature = local_feature.reshape(-1, self.v_embedding, self.region_num).permute(0, 2, 1)
        x = self.pooling(x)
        x = x.reshape(-1, self.v_embedding)
        global_feature = x
        return local_feature, global_feature

