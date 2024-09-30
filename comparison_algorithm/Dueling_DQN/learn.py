# -*- coding: utf-8 -*-
"""
@File     : learn.py
@Date     : 2022-09-07 20:14
@Author   : Terry_Li  - 既然选择了远方，便只顾风雨兼程。
IDE       : PyCharm
@Mail     : 1419727833@qq.com
"""
import numpy as np
import torch
import torch.nn as nn
from net import DQN
import config
import torch.optim as optim
from replay_buffer import ReplayBuffer

current_model = DQN()
target_model = DQN()
optimizer = optim.Adam(current_model.parameters(), lr=config.LEARNING_RATE)
loss_fun = nn.MSELoss()
replay_buffer = ReplayBuffer(config.MAX_BUFFER_CAPACITY)


def learn(batch_size):
    # 目标网络参数更新
    learn_step_counter = 0
    if learn_step_counter % config.Q_NETWORK_ITERATION == 0:
        target_model.load_state_dict(current_model.state_dict())
    learn_step_counter += 1
    state, action, reward, next_state, done = replay_buffer.sample(batch_size)

    state = torch.FloatTensor(np.float32(state))
    next_state = torch.FloatTensor(np.float32(next_state))
    action = torch.LongTensor(action)
    reward = torch.FloatTensor(reward)
    done = torch.FloatTensor(done)

    q_values = current_model(state)
    next_q_values = current_model(next_state)
    next_q_state_values = target_model(next_state)

    q_value = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
    next_q_value = next_q_state_values.gather(1, torch.max(next_q_values, 1)[1].unsqueeze(1)).squeeze(1)
    expected_q_value = reward + config.GAMMA * next_q_value * (1 - done)

    # loss = (q_value - expected_q_value.data.to(gpu())).pow(2).mean()
    loss = loss_fun(q_value, expected_q_value)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss
