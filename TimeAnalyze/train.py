import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch import optim
from torchsummary.torchsummary import summary
from torch.utils.data import Dataset, random_split, DataLoader, TensorDataset

import os
import json
# import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

##### data #####

def split_Train_Val_Data():
    dataset = np.load('./data.npz')
    x = dataset['x'].tolist()
    y = dataset['y'].tolist()
    
    train_inputs = []
    train_labels = []
    test_inputs = []
    test_labels = []
    # -------------------------------------------
    # 將每一類都以 8:2 的比例分成訓練資料和測試資料
    # -------------------------------------------

    num_sample_train = int(0.8*len(x))
    num_sample_test = len(x)

    # 讀取每個類別中所有的測資 (i: label, data: filename)
    data = list(zip(x, y))
    
    for index, var in enumerate(data):  # 前 80% 資料存進 training list
        if index < num_sample_train:
            train_inputs.append(var[0])
            train_labels.append(var[1])
        else:
            test_inputs.append(var[0])
            test_labels.append(var[1])

    torch_dataset = TensorDataset(torch.Tensor(train_inputs), torch.Tensor(train_labels))
    train_dataloader = DataLoader(dataset= torch_dataset, batch_size=batch_size, shuffle=True)
    
    torch_dataset = TensorDataset(torch.tensor(test_inputs), torch.tensor(test_labels))
    test_dataloader = DataLoader(dataset= torch_dataset, batch_size=batch_size, shuffle=False)

    return train_dataloader, test_dataloader

###### Model #####

class BuildModel(nn.Module):

    def __init__(self):
        super(BuildModel, self).__init__()
        self.layers = []

        # ----------------------------------------------
        # 初始化模型的 layer (input size: 3 * 224 * 224)
        in_channels = 84

        layers = []

        net_arch = [128, 64, 32, 'FC']
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


batch_size = 64
lr = 1e-3
epochs = 500

train_dataloader, test_dataloader = split_Train_Val_Data()

ModelPath = './model'
model = torch.load(ModelPath) if os.path.exists(ModelPath) else BuildModel()

C = model.to(device)  # 使用 model
optimizer_C = optim.Adam(C.parameters(), lr=lr)  #  optimizer

# 利用 torchsummary 的 summary package 印出模型資訊，input size: (3 * 224 * 224)
classes = 84
summary(model, (1, classes))

# Loss function
criteron = nn.L1Loss()  # 選擇想用的 loss function

loss_epoch_C = []
train_acc, test_acc = [], []
best_acc, best_auc = 0.0, 0.0

epoch = 0
testing_acc = 0

##### training #####
dtype = torch.float32


if __name__ == '__main__':
    #for epoch in range(epochs):
    while testing_acc < 90 or epoch < 20:
        torch.save(C, ModelPath)
        
        iter = 0
        correct_train, total_train = 0, 0
        correct_test, total_test = 0, 0
        train_loss_C = 0.0

        C.train()  # 設定 train 或 eval

        print('epoch: ' + str(epoch + 1) + ' / ' + str(epochs))
        # ---------------------------
        # Training Stage
        # ---------------------------
        for i, (x, label) in enumerate(train_dataloader):
            x, label = x.to(device, dtype=torch.float), label.to(device, dtype=dtype)

            optimizer_C.zero_grad()  # 清空梯度
            output = C(x)  # 將訓練資料輸入至模型進行訓練

            loss = criteron(output, label)  # 計算 loss

            loss.backward()  # 將 loss 反向傳播
            optimizer_C.step()  # 更新權重

            # 計算訓練資料的準確度 (correct_train / total_train)
            predicted = torch.round(output.data)
            total_train += label.size(0)
            correct_train += (predicted == label.data).sum().item()
                
            train_loss_C += loss.item()
            iter += 1

        print('Training epoch: %d / loss_C: %.3f | acc: %.3f' %
              (epoch + 1, train_loss_C / iter, correct_train / total_train))
        # --------------------------
        # Testing Stage
        # --------------------------

        C.eval()  # 設定 train 或 eval

        for i, (x, label) in enumerate(test_dataloader):

            with torch.no_grad():  # 測試階段不需要求梯度
                x, label = x.to(device, dtype=torch.float), label.to(device, dtype=dtype)

                output = C(x)  # 將訓練資料輸入至模型進行訓練
                loss = criteron(output, label)  # 計算 loss
                print(output.T)
                print(label.T)
                predicted = torch.round(output.data)
                
                total_test += label.size(0)
                correct_test += (predicted == label.data).sum().item()

        print('Testing acc: %.3f' % (correct_test / total_test))

        train_acc.append(100 * (correct_train / total_train))  # training accuracy
        test_acc.append(100 * (correct_test / total_test))  # testing accuracy
        loss_epoch_C.append(train_loss_C / iter)  # loss
    
        with open('./record_mymodel.csv', 'a', encoding='utf8') as f:
            f.write(f"{100 * (correct_train / total_train)},{100 * (correct_test / total_test)},{train_loss_C / iter}\n")
            
        testing_acc = 100 * correct_test / total_test
        epoch += 1