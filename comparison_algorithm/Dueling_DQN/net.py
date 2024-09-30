# -*- coding: utf-8 -*-
"""
@File     : net.py
@Date     : 2022-09-07 15:19
@Author   : Terry_Li  - 既然选择了远方，便只顾风雨兼程。
IDE       : PyCharm
@Mail     : 1419727833@qq.com
"""
import random
import torch
import config
import torch.nn as nn


class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(config.STATE_NUM, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, config.ACTION_NUM)
        )

    def forward(self, x):
        return self.layers(x)

    def act(self, state, epsilon):
        """
        动作
        :param state: 状态
        :param epsilon: 贪婪策略的概率
        :return: 动作的值
        """
        if random.random() > epsilon:
            state = torch.FloatTensor(state).unsqueeze(0)
            q_value = self.forward(state)
            action = q_value.max(1)[1].data[0].item()
        else:
            action = random.randrange(config.ACTION_NUM)
        return action
