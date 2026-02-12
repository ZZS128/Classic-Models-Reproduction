import torch
import torch.utils.data as data
import torch.nn as nn
from torchvision.datasets import FashionMNIST
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
from model import LeNet
import copy
import time

# 数据预处理，划分训练集和验证集，加载数据
def data_process():
    train_dataset = FashionMNIST(
        root='./LeNet/Dataset',
        train=True,
        transform=transforms.Compose([transforms.Resize(size=28),transforms.ToTensor()]),
        download=True)
    train_data, val_data = data.random_split(train_dataset, 
                                             [round(len(train_dataset)*0.8), 
                                              len(train_dataset)-round(len(train_dataset)*0.8)])

    train_data_loader = data.DataLoader(train_data, 
                                        batch_size=128, 
                                        shuffle=True, 
                                        num_workers=8)

    val_data_loader = data.DataLoader(val_data, 
                                      batch_size=128, 
                                      shuffle=False, 
                                      num_workers=8)
    return train_data_loader, val_data_loader


# 训练函数
def train(model, train_data_loader, val_data_loader, num_epochs):
    # 选择设备并导入模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    # 损失函数
    criterion = torch.nn.CrossEntropyLoss()
    # 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # 最优模型
    best_model = copy.deepcopy(model.state_dict())
    # 最高准确度
    best_acc = 0.0
    # 训练集损失列表
    train_loss_list = []
    # 验证集损失列表
    val_loss_list = []
    # 训练集准确度列表
    train_acc_list = []
    # 验证集准确度列表
    val_acc_list = []
    # 当前时间
    since = time.time()


