# -*- coding: utf-8 -*-
"""
@File     : gcn_gru_model.py
@Date     : 2022-12-07
@Author   : Terry_Li     --穷且益坚，不坠青云之志。
IDE       : PyCharm
@Mail     : terry.ljq.dev@foxmail.com
"""
import torch
import torch.nn as nn


# =============================   GCN_GRU ================================= #
# =======================================#
#                    GCN_GRU                #
# =======================================#
# 导入必要的库
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


# 定义GCN-GRU模型
class GCNGRU(nn.Module):
    def __init__(self, n_features, n_hidden, n_classes, n_layers, dropout):
        super(GCNGRU, self).__init__()
        self.gcn_layers = nn.ModuleList([nn.Linear(n_features, n_hidden)])
        self.gcn_layers.extend([nn.Linear(n_hidden, n_hidden) for i in range(n_layers - 1)])
        self.gru = nn.GRU(n_hidden, n_hidden, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(n_hidden, n_classes)

    def forward(self, x, adj):
        # GCN
        for layer in self.gcn_layers:
            x = layer(x)
            x = torch.matmul(adj, x)
            x = torch.relu(x)
            x = self.dropout(x)
        # GRU
        _, x = self.gru(x)
        x = x[-1, :, :]
        # 输出
        x = self.fc(x)
        return x


# 定义数据集
class TrafficDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


# 训练模型
def train_model(model, device, train_loader, optimizer, criterion, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data, adj)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))


# 测试模型
def test_model(model, device, test_loader, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data, adj)
            test_loss += criterion(output, target).item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

# 准备数据
# network_topo: 网络拓扑结构
# traffic_data: 历史流量数据
# window_size: 流量预测的时间窗口大小
# ...

# 构建图形
# adj: 邻接矩阵
# ...

# 分割数据
#

