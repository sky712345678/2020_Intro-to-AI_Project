import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributed import optim
from torchsummary.torchsummary import summary

import os
import json
# import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)
y = x.pow(2) + torch.rand(x.size())*torch.rand(x.size())*torch.rand(x.size())

##### data #####

def split_Train_Val_Data(data_dir):

    path = "./dataset.json"
    dataset = object
    
    with open(path, 'r', encoding='utf8') as f:
        dataset = json.load(f)

    # 讀取每個類別中所有的檔名 (i: label, data: filename)
    for i, data in enumerate(character):
        np.random.seed(42)
        np.random.shuffle(data)

        # -------------------------------------------
        # 將每一類都以 8:2 的比例分成訓練資料和測試資料
        # -------------------------------------------

        num_sample_train = int(0.8*len(data))
        num_sample_test = len(data)

        print(str(i) + ': ' + str(len(data)) + ' | ' +
              str(num_sample_train) + ' | ' + str(num_sample_test))

        for x in data[:num_sample_train]:  # 前 80% 資料存進 training list
            train_inputs.append(x)
            train_labels.append(i)

        for x in data[num_sample_train:num_sample_test]:  # 後 20% 資料存進 testing list
            test_inputs.append(x)
            test_labels.append(i)

    train_dataloader = DataLoader(DogDataset(train_inputs, train_labels, train_transformer),
                                  batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(DogDataset(test_inputs, test_labels, test_transformer),
                                 batch_size=batch_size, shuffle=False)

    return train_dataloader, test_dataloader

train_dataloader, test_dataloader = split_Train_Val_Data(data_dir)

###### Model #####

class BuildModel(nn.Module):

    def __init__(self):
        super(BuildModel, self).__init__()
        self.layers = []

        # ----------------------------------------------
        # 初始化模型的 layer (input size: 3 * 224 * 224)
        in_channels = 3
        num_classes = 20

        layers = []

        net_arch = [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 
                    512, 512, 'M5', 'FC1', 'FC2', 'FC']
        for arch in net_arch:
            if arch == 'M':
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            elif arch == 'M5':
                layers.append(nn.MaxPool2d(kernel_size=3, stride=1, padding=0))
                layers.append(nn.AdaptiveMaxPool2d((-1, 7)))
            elif arch == "FC1":
                layers.append(nn.Linear(512*7*7, 1024))
                layers.append(nn.ReLU(inplace=True))
            elif arch == "FC2":
                layers.append(nn.Linear(1024, 1024))
                layers.append(nn.ReLU(inplace=True))
            elif arch == "FC":
                layers.append(nn.Linear(1024, num_classes))
            else:
                layers.append(nn.Conv2d(in_channels=in_channels,
                                        out_channels=arch, kernel_size=3, padding=0))
                layers.append(nn.ReLU(inplace=True))
                in_channels = arch

        self.layers = nn.ModuleList(layers)
        # ----------------------------------------------

    def forward(self, x):

        # ----------------------------------------------
        # Forward (最後輸出 20 個類別的機率值)
        view = False
        for layer in self.layers:
            if layer.__module__ == 'torch.nn.modules.linear' and not view:
                x = x.view(-1, 512*4*4)
                view = True
            x = layer(x)
        out = x
        # ----------------------------------------------
        return out


batch_size = 16
lr = 1e-3
epochs = 200

ModelPath = './model'
model = torch.load(ModelPath) if os.path.exists(ModelPath) else BuildModel()

C = model.to(device)  # 使用 model
optimizer_C = optim.SGD(C.parameters(), lr=lr)  #  optimizer

# 利用 torchsummary 的 summary package 印出模型資訊，input size: (3 * 224 * 224)
summary(model, (3, 224, 224))

# Loss function
criteron = nn.CrossEntropyLoss()  # 選擇想用的 loss function

loss_epoch_C = []
train_acc, test_acc = [], []
best_acc, best_auc = 0.0, 0.0

epoch = 0
testing_acc = 0

##### training #####

if __name__ == '__main__':
    #for epoch in range(epochs):
    while testing_acc < 60 or epoch < 20:
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
            x, label = x.to(device), label.to(device, dtype=torch.long)

            optimizer_C.zero_grad()  # 清空梯度
            output = C(x)  # 將訓練資料輸入至模型進行訓練

            loss = criteron(output, label)  # 計算 loss

            loss.backward()  # 將 loss 反向傳播
            optimizer_C.step()  # 更新權重

            # 計算訓練資料的準確度 (correct_train / total_train)
            _, predicted = torch.max(output.data, 1)
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
                x, label = x.to(device), label.to(device, dtype=torch.long)

                output = C(x)  # 將訓練資料輸入至模型進行訓練
                loss = criteron(output, label)  # 計算 loss

                _, predicted = torch.max(output.data, 1)
                
                if i == 0:
                    print(f"label: {label}")
                    print(f"predicted: {predicted}")

                total_test += label.size(0)
                correct_test += (predicted == label.data).sum()

        print('Testing acc: %.3f' % (correct_test / total_test))

        train_acc.append(100 * (correct_train / total_train))  # training accuracy
        test_acc.append(100 * (correct_test / total_test))  # testing accuracy
        loss_epoch_C.append(train_loss_C / iter)  # loss
    
        with open('./record_mymodel.csv', 'a', encoding='utf8') as f:
            f.write(f"{100 * (correct_train / total_train)},{100 * (correct_test / total_test)},{train_loss_C / iter}\n")
            
        testing_acc = 100 * correct_test / total_test
        epoch += 1