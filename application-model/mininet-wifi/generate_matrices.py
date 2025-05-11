# -*- coding: utf-8 -*-
"""
@File     : network_shorest_path.py
@Date     : 2022-10-28
@Author   : Terry_Li  -- 穷且益坚，不坠青云之志。
IDE       : VS Code
@Mail     : terry.ljq.dev@foxmail.com
"""
import argparse
import numpy as np
import numpy.random

from pathlib import Path
from tmgen.models import modulated_gravity_tm
import matplotlib.pyplot as plt


def ste_seed():
    """
    设置随机的因子
    """
    numpy.random.seed(args.seed)


def generate_tm():
    """
    创建流量矩阵
    """
    tm = modulated_gravity_tm(args.num_nodes, args.num_tms, args.mean_traffic, args.pm_ratio, args.t_ratio,
                              args.diurnal_freq, args.spatial_variance, args.temporal_variance)
    mean_time_tm = []
    for t in range(args.num_tms):
        mean_time_tm.append(tm.at_time(t).mean())
        print(f"time:  {t} h, mean traffic : {mean_time_tm[-1]}")

    # 构造一个（num_nodes, num_nodes, num_tms)的0-1均匀分布生成的矩阵
    _size = (args.num_nodes,) * 2
    _size += (args.num_tms,)
    temp = np.random.random(_size)

    mask = temp < args.communicate_ratio  # 做一个判断处理，返回的是True 或者是 False
    communicate_tm = tm.matrix * mask  # 把False的地方变成0

    mean_communicate_tm = []
    for t in range(args.num_tms):
        mean_communicate_tm.append(communicate_tm[:, :, t].mean())
        print(f"time： {t} h, mean communicate nodes traffic :{mean_communicate_tm[-1]}")

    np_save(tm.matrix, "traffic_matrix")
    np_save(mean_time_tm, "mean_time_tm")

    np_save(communicate_tm, "communicate_tm")
    np_save(mean_communicate_tm, "mean_communicate_tm")

    plot_tm_mean(mean_time_tm, title="mean_time_tm")
    plot_tm_mean(mean_communicate_tm, title="mean_communicate_tm")


def np_save(file_data, file_name):
    """
    file_data:文件数据
    file_name：文件名字
    """
    Path('./tm_statistic').mkdir(exist_ok=True)
    np.save(f'./tm_statistic/{file_name}.npy', file_data)  # 将数据保存成npy数据格式
    print(f"save {file_name}")


def plot_tm_mean(mean_list, x_label='time', y_label='mean_traffic', title='mean'):
    """
    画图
    """
    fig = plt.figure()  # 画图
    plt.xlabel(x_label)  # 横坐标为时间
    plt.ylabel(y_label)  # 纵坐标为流量均值
    plt.title(title)  # 标题
    x = list(range(len(mean_list)))
    y = mean_list
    plt.bar(x, y)
    Path("./figure").mkdir(exist_ok=True)
    plt.savefig(f"./figure/{title}.pdf", dpi=400)
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate traffic matrices")
    parser.add_argument("--seed", default=2022, type=int, help="random seed")
    parser.add_argument("--num_nodes", default=14, type=int, help="number of nodes of network")
    parser.add_argument("--num_tms", default=24, type=int, help="total number of matrices")
    # 1.55*1e3*0.750
    parser.add_argument("--mean_traffic", default=200, type=int, help="mean volume of traffic (Mbps/s)")
    parser.add_argument("--pm_ratio", default=2, type=float, help="peak-to-mean ratio")
    parser.add_argument("--t_ratio", default=0.2, type=float, help="trough-to-mean ratio")
    parser.add_argument("--diurnal_freq", default=1 / 24, type=float, help="Frequency of modulation")
    parser.add_argument("--spatial_variance", default=20,
                        help="Variance on the volume of traffic between origin-destination pairs")
    parser.add_argument("--temporal_variance", default=0.03, type=float, help="Variance on the volume in time")
    parser.add_argument("--communicate_ratio", default=0.7, help="percentage of nodes to communicate")

    args = parser.parse_args()
    ste_seed()

    generate_tm()
