import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch import optim
from torchsummary.torchsummary import summary
from torch.utils.data import Dataset, random_split, DataLoader, TensorDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

classes = 84

class BuildModel(nn.Module):

    def __init__(self, net_arch):
        super(BuildModel, self).__init__()
        self.layers = []

        # ----------------------------------------------
        # 初始化模型的 layer (input size: 3 * 224 * 224)
        in_channels = classes

        layers = []
        
        for arch in net_arch:
            if arch == "FC":
                layers.append(nn.Sigmoid())
                layers.append(nn.Linear(in_channels, 1))
            else:
                layers.append(nn.Linear(in_channels, arch))
                layers.append(nn.ReLU(inplace=True))
                in_channels = arch

        self.layers = nn.ModuleList(layers)
        # ----------------------------------------------

    def forward(self, x):

        # ----------------------------------------------
        # Forward (最後輸出 20 個類別的機率值)
        view = False
        for layer in self.layers:
            x = layer(x)
        out = x
        # ----------------------------------------------
        return out



##### training #####
dtype = torch.float32

def L1Norm(model):
    L1_reg = torch.tensor(0., requires_grad=True)
    for name, param in model.named_parameters():
        if 'weight' in name:
            L1_reg = L1_reg + torch.norm(param, 1)*1e-5
    return L1_reg
 
model = BuildModel([16,32,64,'FC'])

C = model.to(device)  # 使用 model
optimizer_C = optim.Adam(C.parameters())  #  optimizer
summary(model, (1, classes))
torch.save(model.state_dict(), './model')