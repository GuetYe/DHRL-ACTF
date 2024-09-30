# -*- coding: utf-8 -*-
"""
@File     : performance_comparison.py
@Date     : 2024-2-13
@Author   : Terry_Li     --我不想让你的悲哀成为这个世界的悲哀。
IDE       : PyCharm
@Mail     : terry.ljq.dev@foxmail.com
"""
import config
import time
import os
import plot
from pathlib import Path
from utils import *
from env import MultiDomainEnv


def get_path(name):
    """
    通过txt文件获取各算法生成的路径信息
    返回路径信息表
    """
    route = []
    with open("./path/{}.txt".format(name), 'r', encoding="utf-8") as f:
        lines = f.readlines()
    for line in lines:
        path = line.strip()
        route.append(eval(path))
    return route


def data_to_average(data):
    """
    每隔十个数据做一次平均，并把平均后的数据放回原列表中
    """
    # 每隔10个数据进行平均，并放回列表中
    averaged_data = [sum(data[i:i + 10]) / 10 for i in range(0, len(data), 10)]

    return averaged_data


def analysis_path_data(route):
    used_bw = 0
    free_bw = 0
    delay = 0
    loss = 1
    drop = 0
    distance = 0

    # 瓶颈带宽
    bottleneck_bandwidth = []
    index_list = []
    total_free_bw = []
    total_delay = []
    total_loss = []
    total_drop = []
    total_distance = []
    bottleneck_bandwidth_list = []
    total_used_bw = []
    for index, pkl_path in enumerate(config.dataset.pickle_file_path_yield()):
        env.read_pickle_and_modify(pkl_path)  # 重置图数据
        env.set_graph_params()  # 将图数据中的权重写入矩阵当中
        path_nodes = route[index]
        path_edges = [(path_nodes[i], path_nodes[i + 1]) for i in range(len(path_nodes) - 1)]  # 通过路径去生成对应的边
        for (i, j) in path_edges:
            used_bw += env.use_bw_matrix[i - 1][j - 1]
            free_bw += env.free_bw_matrix[i - 1][j - 1]
            bottleneck_bandwidth.append(env.free_bw_matrix[i - 1][j - 1])
            delay += env.delay_matrix[i - 1][j - 1]
            loss *= (1 - env.loss_matrix[i - 1][j - 1])
            drop += env.pkt_drop_matrix[i - 1][j - 1]
            distance += env.distance_matrix[i - 1][j - 1]

        index_list.append(index)
        total_used_bw.append(used_bw)
        total_free_bw.append(free_bw)  # 将累计的剩余添加到列表中
        total_delay.append(delay)
        total_loss.append(1 - loss)
        total_drop.append(drop)
        total_distance.append(distance)
        bottleneck_bandwidth_list.append(min(bottleneck_bandwidth))

        used_bw = 0
        free_bw = 0  # 迭代后重新置零
        delay = 0  # 迭代后重新置零
        loss = 1
        drop = 0
        distance = 0
        bottleneck_bandwidth.clear()
    return index_list[1:len(data_to_average(total_free_bw))], data_to_average(total_free_bw)[0:len(
        data_to_average(total_free_bw)) - 1], data_to_average(
        total_delay)[0:len(data_to_average(total_free_bw)) - 1], data_to_average(
        total_loss)[0:len(data_to_average(total_free_bw)) - 1], data_to_average(total_drop)[0:len(
        data_to_average(total_free_bw)) - 1], data_to_average(total_distance)[
                                              0:len(data_to_average(total_free_bw)) - 1], data_to_average(
        total_used_bw)[0:len(data_to_average(total_free_bw)) - 1], data_to_average(
        bottleneck_bandwidth_list)[0:len(data_to_average(total_free_bw)) - 1]


def draw_performance_cure():
    """
    通过强化学习得到的路径去解析路径的权重信息
    """
    # 算法1 ospf
    osfp_route = get_path("ospf_path")
    index_list, ospf_free_bw, ospf_delay, ospf_loss, ospf_drop, ospf_distance, ospf_use_bw, ospf_bottleneck_bandwidth = analysis_path_data(
        osfp_route)

    # 算法2 drhl
    drhl_route = get_path("dhrl_path")
    _, dhrl_free_bw, dhrl_delay, dhrl_loss, dhrl_drop, dhrl_distance, dhrl_use_bw, dhrl_bottleneck_bandwidth = analysis_path_data(
        drhl_route)

    # 算法3 q-learning
    q_learnnig_route = get_path("q_learning_path")
    _, q_learnnig_free_bw, q_learnnig_delay, q_learnnig_loss, q_learnnig_drop, q_learnnig_distance, q_learnnig_use_bw, q_learnnig_bottleneck_bandwidth = analysis_path_data(
        q_learnnig_route)
    # 算法4 DRL-PPONSA
    drl_pponsa_route = get_path("drl_pponsa_path")
    _, drl_pponsa_free_bw, drl_pponsa_delay, drl_pponsa_loss, drl_pponsa_drop, drl_pponsa_distance, drl_pponsa_use_bw, drl_pponsa_bottleneck_bandwidth = analysis_path_data(
        drl_pponsa_route)
    # 算法5 DRL-DQN
    dqn_route = get_path("drl_dqn_path")
    _, dqn_free_bw, dqn_delay, dqn_loss, dqn_drop, dqn_distance, dqn_use_bw, dqn_bottleneck_bandwidth = analysis_path_data(
        dqn_route)

    # 算法6 BGP
    bgp_route = get_path("bgp_path")
    _, bgp_free_bw, bgp_delay, bgp_loss, bgp_drop, bgp_distance, bgp_use_bw, bgp_bottleneck_bandwidth = analysis_path_data(
        bgp_route)

    plot.draw_fold_line(picture_path, '网络信息状态/时刻', '平均丢包率(%)',
                        index_list, dhrl_loss,
                        ospf_loss, dqn_loss, q_learnnig_loss,
                        drl_pponsa_loss, bgp_loss)
    # print(calculate_average_percentage(dhrl_bottleneck_bandwidth,drl_pponsa_bottleneck_bandwidth))


if __name__ == "__main__":
    env = MultiDomainEnv(config.dataset.graph)
    local_time = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
    WORK_DIR = Path.cwd().parent
    picture_path = WORK_DIR / "Hierarchical-Actor-Critic-HAC-PyTorch-master/comparison_results/result_{}".format(
        local_time)
    if not os.path.exists(picture_path):
        os.mkdir(picture_path)
    draw_performance_cure()
