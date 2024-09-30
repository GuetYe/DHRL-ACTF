import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from stellargraph import StellarGraph
from stellargraph.layer import GCN
from stellargraph.mapper import FullBatchNodeGenerator
from stellargraph.data import BiasedRandomWalk
from sklearn.metrics import mean_squared_error, mean_absolute_error

# 准备数据
# network_topo: 网络拓扑结构
# traffic_data: 历史流量数据
# window_size: 流量预测的时间窗口大小
# ...

# 构建图形
G = StellarGraph.from_networkx(network_topo)
rw = BiasedRandomWalk(G)
walks = rw.run(
    nodes=list(G.nodes()),  # root nodes
    length=window_size,  # maximum length of a random walk
    n=10,  # number of random walks per root node
    p=0.5,  # Defines (unormalised) probability, 1/p, of returning to source node
    q=2.0,  # Defines (unormalised) probability, 1/q, for moving away from source node
)

generator = FullBatchNodeGenerator(G, method="gcn")
train_data = generator.flow(walks, targets=traffic_data, batch_size=64)


# 定义模型
class GCN_GRU(nn.Module):
    def __init__(self, in_feats, hidden_size):
        super(GCN_GRU, self).__init__()
        self.gcn = GCN(
            layer_sizes=[hidden_size],
            activations=["relu"],
            generator=generator,
            bias=True,
            dropout=0.5,
        )
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, inputs):
        x, A = inputs
        x = self.gcn([x, A])
        x = self.gru(x)[1][0]
        x = self.fc(x)
        return x


model = GCN_GRU(in_feats=generator.features_size(), hidden_size=32)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

# 训练模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

for epoch in range(50):
    running_loss = 0.0
    for i, batch in enumerate(train_data):
        batch = [b.to(device) for b in batch]
        inputs, targets = batch
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print("Epoch %d, loss: %.4f" % (epoch + 1, running_loss / len(train_data)))

# 预测流量
model.eval()
with torch.no_grad():
    test_data = generator.flow(walks, targets=traffic_data, batch_size=1)
    y_pred = []
    y_true = []
    for inputs, targets in test_data:
        inputs = [b.to(device) for b in inputs]
        outputs = model(inputs)
        y_pred.append(outputs.cpu().numpy())
        pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
