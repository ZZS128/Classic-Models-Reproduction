import torch
import torch.utils.data as data
import torch.nn as nn
from torchvision.datasets import FashionMNIST
from torchvision import transforms
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from model import AlexNet
import copy
import time

# 数据预处理，划分训练集和验证集，加载数据
def data_process():
    train_dataset = FashionMNIST(
        root='./AlexNet/Dataset',
        train=True,
        transform=transforms.Compose([transforms.Resize(size=227),transforms.ToTensor()]),
        download=True)
    train_data, val_data = data.random_split(train_dataset, 
                                             [round(len(train_dataset)*0.8), 
                                              len(train_dataset)-round(len(train_dataset)*0.8)])

    train_data_loader = data.DataLoader(train_data, 
                                        batch_size=64, 
                                        shuffle=True, 
                                        num_workers=4)

    val_data_loader = data.DataLoader(val_data, 
                                      batch_size=64, 
                                      shuffle=False, 
                                      num_workers=4)
    return train_data_loader, val_data_loader


# 训练函数
def train(model, train_data_loader, val_data_loader, num_epochs):
    # 选择设备并导入模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    # 损失函数
    criterion = torch.nn.CrossEntropyLoss()
    # 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
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

    for epoch in range(num_epochs):
        print("-" * 10)
        print(f"Epoch {epoch+1}/{num_epochs}")
        train_num = 0 # 训练集样本数
        train_loss = 0.0 # 训练集损失
        train_corrects = 0 # 训练集正确分类的样本数
        val_num = 0 # 验证集样本数
        val_loss = 0.0 # 验证集损失
        val_corrects = 0 # 验证集正确分类的样本数

        # 训练部分
        for step, (b_x, b_y) in enumerate(train_data_loader):
            # 前向传播部分
            b_x = b_x.to(device)
            b_y = b_y.to(device)
            model.train()# 训练模式

            outputs = model(b_x) # 前向传播
            pre_res = torch.argmax(outputs, dim=1) # 预测结果
            loss = criterion(outputs, b_y) # 计算损失
            
            # 反向传播部分
            optimizer.zero_grad()
            loss.backward() # 反向传播
            optimizer.step() # 更新参数

            # 统计损失与精确度
            train_loss += loss.item() * b_x.size(0) # 累加损失
            train_corrects += torch.sum(pre_res == b_y.data) # 累加正确
            train_num += b_x.size(0) # 累加样本数
        
        # 验证部分
        for step, (b_x, b_y) in enumerate(val_data_loader):
            b_x = b_x.to(device)
            b_y = b_y.to(device)
            model.eval() # 验证模式

            output = model(b_x) # 前向传播
            pre_res = torch.argmax(output, dim=1) # 预测结果
            loss = criterion(output, b_y) # 计算损失

            val_loss += loss.item() * b_x.size(0) # 累加损失
            val_corrects += torch.sum(pre_res == b_y.data) # 累加正确
            val_num += b_x.size(0) # 累加样本数
        
        train_loss_list.append(train_loss / train_num) # 计算平均损失
        train_acc_list.append(train_corrects.double().item() / train_num) # 计算平均准确度
        val_loss_list.append(val_loss / val_num) # 计算平均损失
        val_acc_list.append(val_corrects.double().item() / val_num) # 计算平均准确度

        # 输出结果
        print(f"train loss:{train_loss_list[-1]:.4f}, train acc: {train_acc_list[-1]:.4f}, val loss: {val_loss_list[-1]:.4f}, val acc: {val_acc_list[-1]:.4f}")

        # 更新最优模型
        if val_acc_list[-1] > best_acc:
            best_acc = val_acc_list[-1]
            best_model = copy.deepcopy(model.state_dict())

        time_use = time.time() - since
        print(f"time use: {time_use//60:.0f}m {time_use%60:.0f}s")
    
    # 保存最优模型及训练数据
    torch.save(best_model, './AlexNet/alexnet.pth')
    train_process = pd.DataFrame(data={"epoch":range(num_epochs),
                                       "train_loss_list":train_loss_list,
                                       "train_acc_list":train_acc_list,
                                       "val_loss_list":val_loss_list,
                                       "val_acc_list":val_acc_list})
    return train_process

# 可视化
def matplot_process(train_process):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_process["epoch"], train_process["train_loss_list"], label="train loss")
    plt.plot(train_process["epoch"], train_process["val_loss_list"], label="val loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("loss curve")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(train_process["epoch"], train_process["train_acc_list"], label="train acc")
    plt.plot(train_process["epoch"], train_process["val_acc_list"], label="val acc")
    plt.xlabel("epoch")
    plt.ylabel("acc")
    plt.title("acc curve")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    AlexNet = AlexNet()
    train_data_loader, val_data_loader = data_process()
    train_process = train(AlexNet, train_data_loader, val_data_loader, num_epochs=20)
    matplot_process(train_process)

