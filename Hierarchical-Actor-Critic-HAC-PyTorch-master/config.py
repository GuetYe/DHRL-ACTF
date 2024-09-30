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

import yaml

file = open('./paramSet/config.yaml', 'r', encoding="utf-8")
# 读取文件中的所有数据
file_data = file.read()
file.close()

# 指定Loader
params = yaml.load(file_data, Loader=yaml.FullLoader)

# ---------------强化学习训练使用的参数------------------------#
EPOCH = params["Train"]["EPOCH"]
BATCH_SIZE = 128  # 批量大小
LEARNING_RATE = 0.001  # 学习率大小
GAMMA = 0.99  # 折扣率的大小
TAU = 0.001  # 参数更新的权重值
MAX_EPISODES = 100  # 迭代的次数
MAX_STEPS = 1000  # 最大的步长
MAX_BUFFER_CAPACITY = 3000  # 经验池的大小
MAX_TOTAL_REWARD = 300  # 总奖励值

# HAC parameters:
k_level = 2  # num of levels in hierarchy
H = 20  # time horizon to achieve subgoal
lamda = 0.3  # subgoal testing parameter

# ------------------数据类型dtype--------------------------#
NUMPY_TYPE = numpy.float32
TORCH_TYPE = torch.float32

# -------------------归一化参数设置-----------------------------#
# -------------归一化默认参数取值为（0-1）------------------------#
A_NORMAL = 0
B_NORMAL = 1

# ----------------------训练开始的起点-终点---------------------------#
START_END = params["SOURCE_DESTINATION"]  # 起点-终点
# ----------------------域间链路信息---------------------------#
DOMAIN_LINKS = params["DOMAIN_LINKS"]

# ---------------------每个域最后一个交换机的标号，用于区分域与域之间的信息-------------#
DOMAIN_END_NODE1 = params["Initialize"]["DOMAIN_END_NODE1"]
DOMAIN_END_NODE2 = params["Initialize"]["DOMAIN_END_NODE2"]
DOMAIN_END_NODE3 = params["Initialize"]["DOMAIN_END_NODE2"]

# --------------------故障节点和故障边设置-------------------------------#
FAULT_NODES = params["NODES_FAULT"]  # 也可以写成集合或列表的方式(如果有多个故障节点的时候)
FAULT_EDGES = params["EDGES_FAULT"]  # 故障边设置
# ---------------------奖励函数参数的权重值--------------------------#
BETA1 = params["Initialize"]["BETA1"]  # free_bw 剩余带宽的权重值
BETA2 = params["Initialize"]["BETA2"]  # delay 时延的权重值
BETA3 = params["Initialize"]["BETA3"]  # loss 权重值
BETA4 = params["Initialize"]["BETA4"]  # packet drop 权重值
BETA5 = params["Initialize"]["BETA5"]  # distance 权重值

# --------------------奖励函数惩罚值---------------------------#
DISCOUNT1 = params["Initialize"]["DISCOUNT1"]  # 回路惩罚值
DISCOUNT2 = params["Initialize"]["DISCOUNT2"]  # 不是边的惩罚值

# 数据集地址以及topo信息的地址
WORK_DIR = Path.cwd().parent  # 获取当前路径
XML_TOPOLOGIES_PATH = WORK_DIR / 'Hierarchical-Actor-Critic-HAC-PyTorch-master/topologies/topology_node_30.xml'  #
# 链路拓扑路径
DATASET_PATH = WORK_DIR / 'Hierarchical-Actor-Critic-HAC-PyTorch-master/dataSet/weight_pickle30'  # 数据集的路径

# -----------------从数据集中加载一些参数----------------------#
dataset = DataSet(XML_TOPOLOGIES_PATH, DATASET_PATH)
dataset.set_init_topology()
# -------------------动作个数和状态个数--------------------------#
ACTION_NUM = params["Initialize"]["ACTION_NUM"]
STATE_NUM = params["Initialize"]["STATE_NUM"]
