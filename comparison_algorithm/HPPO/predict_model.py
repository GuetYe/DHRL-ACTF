# 用训练好的模型去预测结果
# -*- coding: utf-8 -*-
"""
@File     predict_model.py
@Date     : 2022-10-07
@Author   : Terry_Li     --我以为我的奋不顾身会换来你为我遮风挡雨，可后来所有的大风大浪都是你给的。
IDE       : PyCharm
@Mail     : terry.ljq.dev@foxmail.com
"""
import matplotlib.pyplot as plt
import numpy as np

# Generate some data
# x = np.linspace(0, 10, 100)
# y1 = np.sin(x)
# y2 = np.cos(x)
# y3 = np.sin(x) + np.cos(x)
# y4 = np.sin(x) * np.cos(x)
#
# # Set figure size and style
# plt.figure(figsize=(8, 6))
# plt.style.use('ggplot')
#
# # Set plot title, axis labels, and tick labels
# plt.title('Title', fontsize=18, fontweight='bold')
# plt.xlabel('X Label', fontsize=14, fontweight='bold')
# plt.ylabel('Y Label', fontsize=14, fontweight='bold')
# plt.xticks(fontsize=12, fontweight='bold')
# plt.yticks(fontsize=12, fontweight='bold')
#
# # Plot the data with custom colors and line styles
# plt.plot(x, y1, color='#0072B2', linestyle='-', linewidth=2, label='Line 1')
# plt.plot(x, y2, color='#009E73', linestyle='--', linewidth=2, label='Line 2')
# plt.plot(x, y3, color='#CC79A7', linestyle='-.', linewidth=2, label='Line 3')
# plt.plot(x, y4, color='#D55E00', linestyle=':', linewidth=2, label='Line 4')
#
# # Add grid lines and legend
# plt.grid(True, linestyle='--', alpha=0.25, color='gray', linewidth=1)
# plt.legend(loc='best', fontsize=12, frameon=False)
#
# # Add padding and show the plot
# plt.tight_layout()
# plt.show()

import torch
import torch.nn as nn
import torch.optim as optim

# 示例数据
data = [
    (3, 1, {'free_bw': 38.9996164602477, 'delay': 3.2192468643188477, 'loss': 0.0, 'used_bw': 0.0003835397522972434,
            'pkt_err': 0.0, 'pkt_drop': 0, 'distance': 90.0}),
    # ... (其他元组)
]


# 数据预处理
def prepare_data(data):
    inputs = []
    targets = []

    for source, target, attributes in data:
        # 将属性字典中的信息整合成一个输入序列
        input_sequence = [attributes['free_bw'], attributes['delay'], attributes['loss'], attributes['used_bw'],
                          attributes['pkt_err'], attributes['pkt_drop'], attributes['distance']]
        target_value = attributes['delay']  # 以延迟为例，你可以根据实际问题选择其他属性作为目标值

        inputs.append(input_sequence)
        targets.append(target_value)

    return torch.tensor(inputs, dtype=torch.float32), torch.tensor(targets, dtype=torch.float32)


# 定义Transformer模型
class TransformerModel(nn.Module):
    def __init__(self, input_size, d_model, nhead, num_layers):
        super(TransformerModel, self).__init__()
        self.transformer = nn.Transformer(d_model=d_model, nhead=nhead, num_encoder_layers=num_layers)
        self.fc = nn.Linear(d_model, 1)

    def forward(self, src, tgt):
        x = self.transformer(src, tgt)
        x = self.fc(x[-1, :, :])  # 取最后一个时刻的输出
        return x


# 初始化模型
input_size = 7  # 输入特征的维度，根据属性字典的长度确定
d_model = 64
nhead = 2
num_layers = 2
model = TransformerModel(input_size, d_model, nhead, num_layers)

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 准备数据
inputs, targets = prepare_data(data)

# 训练模型
epochs = 50
for epoch in range(epochs):
    optimizer.zero_grad()
    outputs = model(inputs, inputs)  # src和tgt都使用相同的输入数据，这里简单示例，实际问题中可能需要更复杂的设置
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()

    print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item()}')


def predict(model, input_data):
    model.eval()
    with torch.no_grad():
        input_tensor = torch.tensor(input_data, dtype=torch.float32).unsqueeze(0)
        # 注意：这里的unsqueeze(0)用于在第0维添加一个维度，以适应模型的输入形状
        output = model(input_tensor, input_tensor)  # 这里假设使用相同的输入作为预测的输入，实际问题中可能需要不同的设置
        return output.item()


# 示例用法
input_data = [38.9996164602477, 3.2192468643188477, 0.0, 0.0003835397522972434, 0.0, 0, 90.0]
prediction = predict(model, input_data)
print(f'Predicted Delay: {prediction}')
