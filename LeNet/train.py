from torchvision.datasets import FashionMNIST
from torchvision import transforms
import torch.utils.data as data
import numpy as np
import matplotlib.pyplot as plt
from model import LeNet


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