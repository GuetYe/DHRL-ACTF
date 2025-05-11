# -*- coding: utf-8 -*-
import config
import numpy as np
import torch.nn as nn
import torch.optim as optim
from utils import *
from gpu import gpu
from dataSet import *
from data1 import *
from transformer_network import *

# 初始化模型
# 初始化模型（假设输入展平后维度为58*3=174）
model = Transformer(n_encoder_inputs=1, n_decoder_inputs=1)

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.05)

# 存放损失函数与精确度的路径
loss_graph_path, accuracy_graph_path = create_graph_path()


# 训练模型
def train():
    # 训练模型
    loss_list = []
    epochs_lsit = []
    epochs = 10
    for epoch in range(1, epochs + 1):

        topology_graph = parse_xml_topology(config.XML_TOPOLOGIES_PATH)
    
        # 创建数据加载器
        # train_loader = create_data_loader(
        #         data_dir=config.DATASET_PATH,
        #         batch_size=4,
        #         seq_length=5,
        #     )
        train_loader = read_file_to_datasets(DATASET_PATH, 5, 1, 250)
        # 验证数据加载
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            # 创建解码器初始输入(全零或起始token)
            target_in = torch.zeros_like(inputs[:, :1, :])  # 取第一个时间步
            optimizer.zero_grad()
            outputs = model(inputs,target_in)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            epochs_lsit.append(epoch)
            loss_list.append(loss.item())
            print(f'Epoch [{epoch}/{epochs}], Loss: {loss.item()}')
        if epoch % 1 == 0:
            torch.save(model.state_dict(), './model_weight/{}_transformer_model.pth'.format(epoch))  # 保存模型权重
    plot_episode_data(loss_graph_path, epochs_lsit, loss_list, "episode", "loss", "loss")


if __name__ == "__main__":
    train()
