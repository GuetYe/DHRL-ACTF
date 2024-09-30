# -*- coding: utf-8 -*-
"""
@File     : double_dqn.py
@Date     : 2022-09-07 15:09
@Author   : Terry_Li  - 既然选择了远方，便只顾风雨兼程。
IDE       : PyCharm
@Mail     : 1419727833@qq.com
"""
import math
import utils
from learn import *
from tensorboardX import SummaryWriter
from env import UnicastEnv

seed = 1
torch.manual_seed(seed)


def update_target(current_model, target_model):
    target_model.load_state_dict(current_model.state_dict())


def epsilon_greedy_exploration():
    """
    返回贪婪策略的概率值
    """
    epsilon_start = 1.0
    epsilon_final = 0.001
    epsilon_decay = 500

    epsilon_by_frame = lambda frame_idx: epsilon_final + (epsilon_start - epsilon_final) * math.exp(
        -1. * frame_idx / epsilon_decay)
    # plt.plot([epsilon_by_frame(i) for i in range(10000)])
    return epsilon_by_frame


def train():
    batch_size = 32
    step_num = []
    episode = []
    episode_reward = []
    picture_path = utils.create_picture_path()
    writer1 = SummaryWriter('./DQN/logs')  # 保存训练日志
    writer2 = SummaryWriter('./DQN/logs1')  # 保存训练日志

    for i_epoch in range(1000):
        env = UnicastEnv(config.dataset.graph)
        for index, pkl_path in enumerate(config.dataset.pickle_file_path_yield()):
            if index == 1:
                env.read_pickle_and_modify(pkl_path)  # 将pkl_graph的值覆盖graph
                state = env.reset(config.START, config.END)  # 获取当前的状态信息
                reward_temp = 0
                while True:
                    epsilon_by_frame = epsilon_greedy_exploration()
                    epsilon = epsilon_by_frame(i_epoch)
                    action = current_model.act(state, epsilon)
                    action_node = utils.index_to_node(action)  # 将动作转成节点
                    next_state, reward, done, info = env.step(action_node, state)
                    replay_buffer.push(state, action, reward, next_state, done)

                    reward_temp += reward
                    state = next_state
                    if done:
                        if len(replay_buffer) > batch_size:  learn(batch_size)
                        break
                step_num.append(info.get("step_num"))
                episode_reward.append(reward_temp)
                episode.append(i_epoch)
                print(info)
                print(reward_temp)
                print("\n")

    utils.plot_episode_data(picture_path, episode, list(reversed(episode_reward)), "episode", "episode_reward",
                            "episode_reward")
    utils.plot_episode_data(picture_path, episode, list(reversed(step_num)), "episode", "step_num", "step_num")


if __name__ == "__main__":
    train()
