import timm
import torch
from matplotlib import pyplot as plt
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from network.Resnet import Resnet


class DFAN(nn.Module):
    def __init__(self, args):
        super(DFAN, self).__init__()
        self.args = args
        v_embedding = args.v_embedding
        region_num = args.region_num
        attr_num = args.attr_num

        self.v_encoder = Resnet(region_num, v_embedding)
        self.local_predictor = nn.Linear(v_embedding, attr_num, bias=False)
        self.global_predictor = nn.Linear(v_embedding, attr_num, bias=False)
        self.bias_learner = BiasLearner(v_embedding, attr_num)

    def forward(self, x):
        ########### Visual Feature Augmentation ###########
        local_feature, global_feature = self.v_encoder(x)
        local_feature = F.normalize(local_feature, dim=-1)
        local_predicted_attribute = self.local_predictor(local_feature)
        region_weight = F.softmax(local_predicted_attribute, dim=1)
        attr_v_feature = torch.einsum('bir, bil -> blr', local_feature, region_weight)
        attr_v_feature = F.normalize(attr_v_feature, dim=-1)

        ########### Visual to Semantic ###########
        w = self.local_predictor.weight
        local_predicted_results = torch.einsum('bir, ir -> bi', attr_v_feature, w)
        global_predicted_results = self.global_predictor(global_feature)

        ########### Semantic Feature Augmentation ###########
        local_predicted_bias = self.bias_learner(attr_v_feature).mean(dim=1)
        global_predicted_bias = self.bias_learner(global_feature)
        local_final_predicted_results = local_predicted_results + local_predicted_bias
        global_final_predicted_results = global_predicted_results + global_predicted_bias

        package = {'attr_v_feature': attr_v_feature,
                   'local_final_predicted_results': local_final_predicted_results,
                   'global_final_predicted_results': global_final_predicted_results}
        return package

class BiasLearner(nn.Module):
    def __init__(self, v_embedding, attr_num):
        super(BiasLearner, self).__init__()
        self.fc1 = nn.Linear(v_embedding, v_embedding // 2, bias=True)
        self.fc2 = nn.Linear(v_embedding // 2, v_embedding // 4, bias=True)
        self.fc3 = nn.Linear(v_embedding // 4, attr_num, bias=True)
        self.relu = torch.relu

    def forward(self, x):
        bias = self.relu(self.fc1(x))
        bias = self.relu(self.fc2(bias))
        bias = self.fc3(bias)

        return bias

