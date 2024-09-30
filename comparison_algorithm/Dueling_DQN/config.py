# -*- coding: utf-8 -*-
"""
@File     : config.py
@Date     : 2022-10-09
@Author   : Terry_Li     --总有地上的生灵，敢于面对雷霆的威光。
IDE       : PyCharm
@Mail     : terry.ljq.dev@foxmail.com
"""
# 参数配置
import numpy
import torch
from pathlib import Path
from dataSet import DataSet

# ---------------强化学习训练使用的参数------------------------#
BATCH_SIZE = 128  # 批量大小
LEARNING_RATE = 0.001  # 学习率大小
GAMMA = 0.99  # 折扣率的大小
TAU = 0.001  # 参数更新的权重值
MAX_EPISODES = 100  # 迭代的次数
MAX_STEPS = 1000  # 最大的步长
MAX_BUFFER_CAPACITY = 3000  # 经验池的大小
MAX_TOTAL_REWARD = 300  # 总奖励值
Q_NETWORK_ITERATION = 500  # Q网络更新的次数

# ------------------数据类型dtype--------------------------#
NUMPY_TYPE = numpy.float32
TORCH_TYPE = torch.float32

# -------------------归一化参数设置-----------------------------#
# -------------归一化默认参数取值为（0-1）------------------------#
A_NORMAL = 0
B_NORMAL = 1

# ----------------------训练开始的起点-终点---------------------------#
START = 1  # 起点
END = [14]  # 终点

# ---------------------奖励函数参数的权重值--------------------------#
BETA1 = 0.7  # free_bw 剩余带宽的权重值
BETA2 = 0.3  # delay 时延的权重值
BETA3 = 0.1  # loss 权重值
BETA4 = 0.1  # packet drop 权重值
BETA5 = 0.1  # distance 权重值

# --------------------奖励函数惩罚值---------------------------#
DISCOUNT1 = -0.5  # 回路惩罚值
DISCOUNT2 = -0.1  # 不是边的惩罚值

# 数据集地址以及topo信息的地址
WORK_DIR = Path.cwd().parent  # 获取当前路径
XML_TOPOLOGIES_PATH = WORK_DIR / 'PPOGG/topologies/topology_node_14.xml'  # 链路拓扑路径
DATASET_PATH = WORK_DIR / 'PPOGG/dataSet/weight_pickle'  # 数据集的路径
# -----------------从数据集中加载一些参数----------------------#
dataset = DataSet(XML_TOPOLOGIES_PATH, DATASET_PATH)
dataset.set_init_topology()

# -------------------动作个数和状态个数--------------------------#
ACTION_NUM = dataset.nodes_num
STATE_NUM = dataset.nodes_num
