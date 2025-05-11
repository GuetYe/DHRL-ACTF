# -*- coding: utf-8 -*-
"""
@File     : utils.py
@Date     : 2022-10-08
@Author   : Terry_Li     --落霞与孤鹜齐飞，秋水共长天一色。
IDE       : PyCharm
@Mail     : terry.ljq.dev@foxmail.com
"""
import shutil
import time
import os
from pathlib import Path

import matplotlib.pyplot as plt

import config
import numpy as np
import torch


def soft_update(target, source, tau):
    """
    Copies the parameters from source network (x) to target network(y) using the below update
    y = TAU*x + (1 - TAU)*y
    :param target: Target network (PyTorch)
    :param source: Source network (PyTorch)
    :param tau:  更新比例权重
    :return:
    """
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - tau) + param.data * tau
        )


def hard_update(target, source):
    """
    Copies the parameters from source network to target network
    :param target: Target network (PyTorch)
    :param source: Source network (PyTorch)
    :return:
    """
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


def save_training_checkpoint(state, is_best, episode_count):
    """
    Saves the models, with all trainning parameters intact
    保存模型
    :param state:
    :param is_best:
    :param episode_count:
    :return:
    """
    filename = str(episode_count) + 'checkpoint.path.rar'
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.rar')


def get_non_minus_one_max_min(matrix):
    """
        返回除去-1的最大最小值
    :param matrix: 要求最大最小的矩阵
    :return: max, _min, matrix 【最大最小值， matrix将-1改为0】
    """
    non_minus_one_mask = np.where(matrix != -1)
    _max = matrix[non_minus_one_mask].max()
    _min = matrix[non_minus_one_mask].min()
    matrix[np.where(matrix == -1)] = 0
    return _max, _min, matrix


def normalize_matrix(matrix, max_value, min_value, nodes_num):
    """
       将矩阵按最大最小正则 值在0到1之间
    :param nodes_num: 网络节点个数
    :param matrix:矩阵
    :param max_value:矩阵中最大的值
    :param min_value:矩阵中最小的值
    :return:归一化后的矩阵
    归一化公式: mij = a + (mij-min(TM))*(b-a)/(max(TM)-min(TM)+1e-6)
    """
    # normal_m = (matrix - min_value) / (max_value - min_value)
    normal_m = (matrix - min_value) / (max_value - min_value + 1e-6)  # 加一个很小的数避免分母为零时计算出错
    normal_m -= normal_m * np.identity(len(nodes_num))  # 乘以一个对角线全为1的方阵
    normal_m = config.A_NORMAL + normal_m * (config.B_NORMAL - config.A_NORMAL)
    return normal_m.astype(config.NUMPY_TYPE)


def node_to_index(node):
    """
        节点从1起， 索引从0起，将节点号转为索引号
    :param node: 节点号
    :return: 索引号
    """
    if isinstance(node, list):
        return [i - 1 for i in node]
    else:
        return node - 1


def index_to_node(index):
    """
        索引号转为节点号
    :param index: 索引号
    :return: 节点号
    """
    if isinstance(index, list):
        return [i + 1 for i in index]
    else:
        return index + 1


def get_adj_edges(agent_node, adj_node: list):
    """
    :param agent_node: 当前智能体所在的节点位置 1
    :param adj_node:   与智能体相邻的节点列表 [3,4,5,11]
    :return: 智能体的邻接边 [(1,3),(1,4),(1,5),(1,11)]
    """
    adj_list = [agent_node]
    adj_list_tup = []
    for i in adj_node:
        adj_list.append(i)

        adj_list_tup.append(tuple(adj_list))
        adj_list.pop(1)
    return adj_list_tup


def compare_link(link1: tuple, link2: list):
    """
    比较动作产生的链路与智能体邻接边的链路是否一致
    :param link1: 智能体随机动作的链路tuple(11,14)
    :param link2: 智能体邻接边的链路[(1, 3), (1, 4), (1, 5), (1, 11)]
    :return: True or False
    """
    if link1 in link2:
        return True
    else:
        return False


def update_state(agent_index, next_jump_index, _state):
    """
    根据索引值更改list中的值
    :param _state: 当前的状态矩阵
    :param agent_index: 当前智能体的索引值
    :param next_jump_index: 智能体下一跳的索引值
    :return: 新的状态矩阵
    """
    _state[agent_index], _state[next_jump_index] = _state[next_jump_index], _state[agent_index]
    return _state


def create_picture_path():
    """
    在文件夹下创建文件
    """
    local_time = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
    WORK_DIR = Path.cwd().parent
    picture_path = WORK_DIR / "Dueling_DQN/result/result_{}".format(local_time)

    if not os.path.exists(picture_path):
        os.mkdir(picture_path)

    return picture_path

def plot_episode_data(picture_path, x, y, x_label: str, y_label: str, file_name):
    """
    画一个图
    picture_path:存放图片的路径
    x:x坐标的数据
    y:y坐标的数据
    x_label:x的标签
    y_label:y的标签
    file_name:文件名字
    """
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.plot(x, y)
    plt.savefig("{}/{}.pdf".format(picture_path, file_name))
    plt.show()


if __name__ == "__main__":
    x = np.array([[3., -1., 2.],
                  [2., 0., 0.],
                  [0., 1., -1.], ])

    agent_node = 1
    adj_node = [3, 4, 5, 11]
    link1 = (1, 6)
    link2 = [(1, 3), (1, 4), (1, 5), (1, 11)]
    # a = compare_link(link1, link2)
    # print(a)
