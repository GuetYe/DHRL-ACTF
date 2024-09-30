# -*- coding: utf-8 -*-
"""
@File     : config.py
@Date     : 2022-11-30
@Author   : Terry_Li     --剑修一生痴绝处，无梦到此登城头。
IDE       : PyCharm
@Mail     : terry.ljq.dev@foxmail.com
"""
import os
import utils
from pathlib import Path

WORK_DIR = Path.cwd().parent
experimental_data_path = WORK_DIR / "PPOGG/save_experimental_data"
names = os.listdir(experimental_data_path)

# ppo_update_time 1
data_path0 = experimental_data_path / str(names[5]) # 1
episode = utils.load_experimental_data(data_path0 / "episode.pkl")
episode_reward0 = utils.load_experimental_data(data_path0 / "episode_reward.pkl")
step_num0 = utils.load_experimental_data(data_path0 / "step_num.pkl")

# ppo_update_time  10
data_path1 = experimental_data_path / str(names[10]) # 4
episode_reward1 = utils.load_experimental_data(data_path1 / "episode_reward.pkl")
step_num1 = utils.load_experimental_data(data_path1 / "step_num.pkl")

# ppo_update_time 100
data_path2 = experimental_data_path / str(names[7]) # 0
episode_reward2 = utils.load_experimental_data(data_path2 / "episode_reward.pkl")
step_num2 = utils.load_experimental_data(data_path2 / "step_num.pkl")

# ppo_update_time 1000
data_path3 = experimental_data_path / str(names[8]) # 1
episode_reward3 = utils.load_experimental_data(data_path3 / "episode_reward.pkl")
step_num3 = utils.load_experimental_data(data_path3 / "step_num.pkl")

# episode_reward1 = (np.array(episode_reward1) - 6).tolist()
# step_num1 = (np.array(step_num1) + 6).tolist()
# 画出对比的图
compare_path = utils.create_compare_path()
utils.plot_compare_data("compare ppo_update_time data", compare_path, episode, episode_reward0, episode_reward1,
                        episode_reward2, episode_reward3,
                        "episode", "reward", "compare_ppo_reward")
# utils.plot_compare_data("compare ppo_step_num data", compare_path, episode, step_num0, step_num1, step_num2, step_num3,
#                         "episode", "stemp_num", "compare_ppo_step_num")
# print("over")
