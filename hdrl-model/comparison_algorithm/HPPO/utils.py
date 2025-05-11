# -*- coding: utf-8 -*-
"""
@File     : utils.py
@Date     : 2023-1-04
@Author   : Terry_Li     --落霞与孤鹜齐飞，秋水共长天一色。
IDE       : PyCharm
@Mail     : terry.ljq.dev@foxmail.com
"""
import os
import shutil
import time
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.lines as mlines

import config
import numpy as np
import torch
import pickle

# 设置中文字体
plt.rcParams['font.sans-serif'] = [u'simHei']  # 显示中文
plt.rcParams['axes.unicode_minus'] = False  # 解决负号问题


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
    _state = _state.squeeze(1)  # 降维度
    index = (
        torch.LongTensor([agent_index, next_jump_index]),
        torch.LongTensor([agent_index, next_jump_index]),
    )
    new_value = torch.FloatTensor([0, 1])
    new_state = _state.index_put(index, new_value)
    new_state = new_state.unsqueeze(1)  # 升维度
    return new_state


def combine_state(state):
    if state is not None:
        _combine_state = torch.stack(
            [torch.from_numpy(state)
             ], dim=1)
        return _combine_state
    else:
        return None


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
    # Set figure size and style
    plt.figure(figsize=(8, 6))
    plt.style.use('ggplot')

    # Set plot title, axis labels, and tick labels
    plt.xlabel(x_label, fontsize=14, fontweight='bold')
    plt.ylabel(y_label, fontsize=14, fontweight='bold')
    plt.xticks(fontsize=12, fontweight='bold')
    plt.yticks(fontsize=12, fontweight='bold')

    # Plot the data with custom colors and line styles
    plt.plot(x, y, color='#0072B2', linestyle='-', linewidth=2)

    # Add grid lines and legend
    plt.grid(True, linestyle='--', alpha=0.25, color='gray', linewidth=1)
    plt.legend(loc='best', fontsize=12, frameon=False)

    # Add padding and show the plot
    plt.tight_layout()
    plt.savefig("{}/{}.pdf".format(picture_path, file_name))
    plt.show()


def plot_compare_data(title_name, data_path, x, y1, y2, y3, y4, x_label, y_label, file_name):
    """
    画两条曲线的图
    title_name:图的标题
    picture_path:存放图片的路径
    x:x坐标的数据
    y:y坐标的数据
    x_label:x的标签
    y_label:y的标签
    file_name:文件名字
    """

    # plt.title(title_name)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.plot(x, y1, color='#0072B2', linestyle='-', label="PPONSA")  # "#0848ae"
    plt.plot(x, y2, color='#009E73', linestyle='-.', label="Dueling DQN")  # linestyle='--'
    # plt.plot(x, y3, color='#CC79A7', linestyle='-.',label="ξ1:ξ2=0.5:1.5")
    # plt.plot(x, y4, color='#D55E00', linestyle='-',label="ξ1:ξ2=0.5:0.8")  # "#e8710a"
    # 在两条曲线之间做一个填充
    # plt.fill_between(x, y1, y2)
    # 显示图例
    plt.legend()  # 默认loc = Best
    plt.grid(True, linestyle='--', alpha=0.25)
    plt.savefig("{}/{}.pdf".format(data_path, file_name), dpi=300, bbox_inches='tight', pad_inches=0)
    plt.savefig("{}/{}.jpeg".format(data_path, file_name), dpi=300, bbox_inches='tight', pad_inches=0)
    plt.show()


def autolabel(rects):
    """
    定义函数来显示柱子上的数值
    """
    for rect in rects:
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width() / 2. - 0.08, 1.02 * height, '%.2f' % height, size=4,
                 family="Times new roman")  # rotation=270 数值旋转270度


def plot_traffic_graph(data_path, x, y1, y2, y3, y4, y5, x_label, y_label, file_name, label_name):
    """
    画出与OSPF、DVRP、LSRP的流量对比图，柱状图
    """
    # 设置xy标签的值
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    autolabel(plt.bar(x, y1, width=0.15, color="#0848ae", label="PPONSA$_{%s}$" % label_name))  # "#0848ae"
    autolabel(plt.bar([i + 0.15 for i in x], y2, width=0.15, color="#00C0C0", label="Dueling DQN$_{%s}$" % label_name))
    autolabel(plt.bar([i + 0.3 for i in x], y3, width=0.15, color="#df208c", label="OSPF$_{%s}$" % label_name))
    autolabel(plt.bar([i + 0.45 for i in x], y4, width=0.15, color="#e8710a",
                      label="DVRP$_{%s}$" % label_name))  # "#e8710a"
    autolabel(plt.bar([i + 0.6 for i in x], y5, width=0.15, color="#E1F190",
                      label="LSRP$_{%s}$" % label_name))  # "#e8710a"
    # 显示图例和网格
    plt.legend(fontsize=10)  # 默认loc = Best # ncol=4让数据标签横向排列
    plt.grid(True, linestyle='--', alpha=0.5)
    # 修改x刻度名字
    plt.xticks([i + 0.3 for i in x], ['0:00', '3:00', '6:00', '9:00', '12:00', '15:00', '18:00', '21:00'])
    plt.savefig("{}/{}.pdf".format(data_path, file_name), dpi=300, bbox_inches='tight', pad_inches=0.2)
    plt.savefig("{}/{}.jpeg".format(data_path, file_name), dpi=300, bbox_inches='tight', pad_inches=0.2)
    plt.show()


def create_picture_path():
    """
    在文件夹下创建文件
    """
    local_time = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
    WORK_DIR = Path.cwd().parent
    picture_path = WORK_DIR / "HPPO/result/result_{}".format(local_time)
    experimental_data_path = WORK_DIR / "HPPO/save_experimental_data/data_{}".format(local_time)

    if not os.path.exists(picture_path):
        os.mkdir(picture_path)
    if not os.path.exists(experimental_data_path):
        os.mkdir(experimental_data_path)
    return picture_path, experimental_data_path


def create_compare_path():
    """
    创建对比实验的路径
    """
    local_time = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
    WORK_DIR = Path.cwd().parent
    compare_path = WORK_DIR / "HPPO/compare_picture/result_{}".format(local_time)
    if not os.path.exists(compare_path):
        os.mkdir(compare_path)
    return compare_path


def save_experimental_data(data, filepath, name: str):
    """
    将实验数据存储成pickle的文件
    data:将要存储的数据
    filepath:存储数据的路径
    name:文件的名字
    """
    filename = "{}/{}.pkl".format(filepath, name)
    f = open(filename, 'wb')
    pickle.dump(data, f)
    f.close()
    return filename


def load_experimental_data(filename):
    """
    获取实验的数据
    filepath:存储数据的路径
    """
    f = open(filename, 'rb')
    data = pickle.load(f)
    f.close()
    return data


def calculate_average_percentage(a: list, b: list, c: list, d: list, e: list):
    """
    用于计算对比实验中的数值平均百分比
    输入的值为列表
    输出的值为百分比
    """
    ppo_ddqn_percentage = (sum(b) - sum(a)) / sum(b)
    ppo_ospf_percentage = (sum(c) - sum(a)) / sum(c)
    ppo_dvrp_percentage = (sum(d) - sum(a)) / sum(d)
    ppo_lsrp_percentage = (sum(e) - sum(a)) / sum(e)
    # ppo_ddqn_percentage = (sum(a) - sum(b)) / sum(a)
    # ppo_ospf_percentage = (sum(a) - sum(c)) / sum(a)
    # ppo_dvrp_percentage = (sum(a) - sum(d)) / sum(a)
    # ppo_lsrp_percentage = (sum(a) - sum(e)) / sum(a)
    # max_ppo_ddqn_percentage = (max(a) - min(b)) / max(a)
    return ppo_ddqn_percentage, ppo_ospf_percentage, ppo_dvrp_percentage, ppo_lsrp_percentage


def draw_fold_line(x_data, parameter1_data, parameter2_data):
    # 假设有两个参数的数据，分别存储在 parameter1_data 和 parameter2_data 中
    # 假设还有 x 轴上的数据，存储在 x_data 中

    # 绘制折线图
    line1, = plt.plot(x_data, parameter1_data, color='blue', label='DRL-PPO')
    line2, = plt.plot(x_data, parameter2_data, color='red', label='Dijkstra')

    # 绘制数据点
    scatter1 = plt.scatter(x_data, parameter1_data, color='blue', marker='*')
    scatter2 = plt.scatter(x_data, parameter2_data, color='red', marker='s')

    # 添加标签和标题
    plt.xlabel('时间')
    plt.ylabel('丢包率(%)')
    plt.title('DRL-PPO与Dijkstra的丢包率对比曲线')

    # 创建自定义图例标记
    custom_legend = [
        mlines.Line2D([], [], color='blue', marker='*', linestyle='None', label='DRL-PPO'),
        mlines.Line2D([], [], color='red', marker='s', linestyle='None', label='Dijkstra')
    ]

    # 添加图例，并设置图例标记样式
    plt.legend(handles=custom_legend)

    # 添加网格线
    plt.grid(True)

    # 显示图形
    plt.show()


if __name__ == "__main__":
    # x = np.array([[3., -1., 2.],
    #               [2., 0., 0.],
    #               [0., 1., -1.], ])
    #
    # agent_node = 1
    # adj_node = [3, 4, 5, 11]
    # link1 = (1, 6)
    # link2 = [(1, 3), (1, 4), (1, 5), (1, 11)]
    x_data = ['0:00', '3:00', '6:00', '9:00', '12:00', '15:00', '18:00', '21:00']
    x1 = [0.02, 0.01, 0.01, 0.01, 0.02, 0.01, 0.01, 0.03]
    x2 = [0.02, 0.14, 0.25, 0.25, 0.36, 0.68, 0.14, 0.29]
    draw_fold_line(x_data, x1, x2)

    # a = compare_link(link1, link2)
    # print(a)
