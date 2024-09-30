# -*- coding: utf-8 -*-
"""
@File     : train.py
@Date     : 2023-12-13
@Author   : Terry_Li     --古有曳影之剑，腾空而舒，克伐四方，历史我的源代码。
IDE       : PyCharm
@Mail     : terry.ljq.dev@foxmail.com
"""

import utils
from train_model import *
from utils import *
from env import UnicastEnv
from collections import namedtuple

# 打印环境的状态空间和动作空间
print('State Dimensions :', config.STATE_NUM)
print('Action Dimensions :', config.ACTION_NUM)

render = True
seed = 10
torch.manual_seed(seed)
Transition = namedtuple('Transition', ['state', 'action', 'a_log_prob', 'reward', 'next_state'])


def print_param(epoch, pickle_num, total_reward, path_information, timeSpent):
    """
    用于打印参数
    timeSpent: 统计时间
    epoch: 迭代次数
    pickle_num: 网络流量地图个数
    total_reward: 总奖励值
    path_information: 路径信息
    """
    print("***----->既然选择了远方，便只顾风雨兼程<----***")
    print("***----->当前迭代次数:{}<----***".format(epoch))
    print("***----->执行第 {} 个地图中的参数<----***".format(pickle_num))
    print("***----->路径奖励: {}<----***".format(total_reward))
    print("***----->路径信息: {}<----***".format(path_information))
    print("***----->共需要 {}s 时间<----***".format(timeSpent))
    print("***--------------------end--------------------------***\n")


def train():
    agent = PPO()
    pickle_reward = []  # 每个pickle文件的奖励值
    episode_reward = []  # 每次迭代后的平均奖励值
    episode = []  # 迭代值
    pickle_step_num = []  # 每个pickle文件的步长情况
    episode_step_num = []  # 每次迭代后的平均步长
    picture_path, experimental_data_path = utils.create_picture_path()
    for i_epoch in range(1, config.EPOCH + 1):
        start_time = time.time()  # 起始时间
        env = UnicastEnv(config.dataset.graph)
        for index, pkl_path in enumerate(config.dataset.pickle_file_path_yield()):
            if index == 1:
                env.read_pickle_and_modify(pkl_path)  # 将pkl_graph的值覆盖graph
                for src, dst in config.START_END:
                    state = env.reset(src, dst)  # 获取当前的状态信息
                    state = utils.combine_state(state)
                    state = torch.FloatTensor(state)
                    reward_temp = 0
                    while True:
                        action, action_prob = agent.select_action(state)
                        action_node = utils.index_to_node(action)  # 将动作转成节点
                        # print("agent_index----->", agent_index)
                        next_state, reward, done, info = env.step(action_node, state)
                        trans = Transition(state, action, action_prob, reward, next_state)
                        agent.store_transition(trans)
                        reward_temp += reward

                        if done:
                            if len(agent.buffer) >= agent.batch_size: agent.update(i_epoch)
                            # agent.writer.add_scalar('liveTime/livestep', i_epoch, global_step=i_epoch)
                            break
                        state = next_state
                    pickle_step_num.append(info.get("step_num"))
                    pickle_reward.append(reward_temp)

                    end_time = time.time()  # 结束的时间
                    print_param(i_epoch, index, reward_temp, info, end_time - start_time)  # 打印出每一步的路径信息

        episode.append(i_epoch)
        episode_reward.append(np.mean(pickle_reward))
        episode_step_num.append((np.mean(pickle_step_num)))

        pickle_reward.clear()  # 将其置空处理
        pickle_step_num.clear()  # 将其置空处理

    utils.save_experimental_data(episode, experimental_data_path, "episode")
    utils.save_experimental_data(episode_reward, experimental_data_path, "mean_reward")
    utils.save_experimental_data(episode_step_num, experimental_data_path, "step_num")
    utils.plot_episode_data(picture_path, episode, episode_reward, "episode", "mean_reward", "mean_reward")
    utils.plot_episode_data(picture_path, episode, episode_step_num, "episode", "step_num", "step_num")


if __name__ == '__main__':
    train()
    print("training over")
