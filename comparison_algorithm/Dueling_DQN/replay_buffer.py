# -*- coding: utf-8 -*-
"""
@File     : replay_buffer.py
@Date     : 2022-09-07 20:14
@Author   : Terry_Li  - 既然选择了远方，便只顾风雨兼程。
IDE       : PyCharm
@Mail     : 1419727833@qq.com
"""
import random
from collections import deque
from collections import namedtuple
import numpy as np

# 使用具名元组 快速建立一个类
Transition = namedtuple('Transition',
                        ('state', 'action', 'reward', 'next_state'))


class ReplayBuffer(object):
    """
    经验回放函数
    """

    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        state = np.expand_dims(state, 0)
        next_state = np.expand_dims(next_state, 0)
        # print(next_state.shape)
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return np.concatenate(state), action, reward, np.concatenate(next_state), done

    def __len__(self):
        return len(self.buffer)

