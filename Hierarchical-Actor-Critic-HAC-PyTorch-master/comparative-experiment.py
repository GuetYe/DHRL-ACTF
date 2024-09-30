# -*- coding: utf-8 -*-
"""
@File     : config.py
@Date     : 2024-02-17
@Author   : Terry_Li     --剑修一生痴绝处，无梦到此登城头。
IDE       : PyCharm
@Mail     : terry.ljq.dev@foxmail.com
"""
import os
import utils
import time
import plot
import matplotlib.pyplot as plt
from pathlib import Path

WORK_DIR = Path.cwd().parent
experimental_data_path = WORK_DIR / "Hierarchical-Actor-Critic-HAC-PyTorch-master/save_experimental_data"
names = os.listdir(experimental_data_path)


def parsing_data(idx):
    """
    idx: 所要解析的数据id
    返回值：迭代次数的列表值，奖励值，步长
    """
    data_path = experimental_data_path / str(names[idx])
    episodes = utils.load_experimental_data(data_path / "episode.pkl")
    episode_reward = utils.load_experimental_data(data_path / "mean_reward.pkl")
    step_num = utils.load_experimental_data(data_path / "step_num.pkl")
    return episodes, episode_reward, step_num


# # ppo_update_time  10
# data_path1 = experimental_data_path / str(names[10]) # 4
# episode_reward1 = utils.load_experimental_data(data_path1 / "episode_reward.pkl")
# step_num1 = utils.load_experimental_data(data_path1 / "step_num.pkl")
#
# # ppo_update_time 100
# data_path2 = experimental_data_path / str(names[7]) # 0
# episode_reward2 = utils.load_experimental_data(data_path2 / "episode_reward.pkl")
# step_num2 = utils.load_experimental_data(data_path2 / "step_num.pkl")
#
# # ppo_update_time 1000
# data_path3 = experimental_data_path / str(names[8]) # 1
# episode_reward3 = utils.load_experimental_data(data_path3 / "episode_reward.pkl")
# step_num3 = utils.load_experimental_data(data_path3 / "step_num.pkl")

# episode_reward1 = (np.array(episode_reward1) - 6).tolist()
# step_num1 = (np.array(step_num1) + 6).tolist()
# 画出对比的图
# compare_path = utils.create_compare_path()
# utils.plot_compare_data("compare ppo_update_time data", compare_path, episode, episode_reward0, episode_reward1,
#                         episode_reward2, episode_reward3,
#                         "episode", "reward", "compare_ppo_reward")
# utils.plot_compare_data("compare ppo_step_num data", compare_path, episode, step_num0, step_num1, step_num2, step_num3,
#                         "episode", "stemp_num", "compare_ppo_step_num")
# print("over")

if __name__ == "__main__":
    local_time = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
    WORK_DIR = Path.cwd().parent
    picture_path = WORK_DIR / "Hierarchical-Actor-Critic-HAC-PyTorch-master/comparison_results/result_{}".format(
        local_time)
    if not os.path.exists(picture_path):
        os.mkdir(picture_path)
    # 本文件主要是对比不同算法以及修改部分参数的奖励收敛情况，同时包括不同预测方法的收敛情况
    # ppo_update_time 1
    episode, episode_reward1, step_num1 = parsing_data(37)
    _, episode_reward2, step_num2 = parsing_data(30)
    _, episode_reward3, step_num3 = parsing_data(9)
    # _, episode_reward4, step_num4 = parsing_data(33)
    # 绘制主图
    fig, ax = plt.subplots(1, 1, figsize=(12, 7))
    plt.xlabel("训练代数", fontsize=15)
    plt.ylabel("最终奖励值", fontsize=15)
    # ax.plot(episode, episode_reward1, color='cornflowerblue', linewidth=1.5,
    #         alpha=0.7)
    ax.plot(episode, episode_reward1, color='blue', label='discount = [1,-0.1,-0.2,-0.3]', linewidth=1.5, linestyle='--',
            alpha=0.7)

    ax.plot(episode, episode_reward2, color='c', label='discount = [1,-0.3,-0.4,-0.6]',
            linewidth=1.5, linestyle='-', alpha=0.7)

    ax.plot(episode, episode_reward3, color='red', label='discount = [1,-0.5,-0.6,-0.9]', linestyle='-.', alpha=0.7)
    # ax.plot(episode, episode_reward4, color='orange', label='batch size = 128', linestyle=':', alpha=0.7)

    ax.legend(loc='right', fontsize=15)

    # plt.show()

    # 绘制缩放图
    axins = ax.inset_axes((0.4, 0.2, 0.4, 0.3))

    # 在缩放图中也绘制主图所有内容，然后根据限制横纵坐标来达成局部显示的目的
    axins.plot(episode, episode_reward1, color='blue', label='trick-1', linestyle='--', alpha=0.7)
    axins.plot(episode, episode_reward2, color='c', label='trick-2', linestyle='-', alpha=0.7)
    axins.plot(episode, episode_reward3, color='red', label='trick-3', linestyle='-.', alpha=0.7)
    # axins.plot(episode, episode_reward4, color='orange', label='trick-4', linestyle=':', alpha=0.7)

    # 局部显示并且进行连线
    plot.zone_and_linked(ax, axins, 50, 150, episode,
                         [episode_reward1, episode_reward2, episode_reward3], 'right')
    plt.savefig("{}/{}.png".format(picture_path, "学习率"), dpi=400, bbox_inches='tight', pad_inches=0.1)

    # 设置图例，指定字体大小和字体类型
    # ax.legend(fontsize=9, fontfamily='Times New Roman')  # 字体大小对应五号字体

    plt.show()
