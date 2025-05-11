import time
import torch
import config
import utils
import plot
import numpy as np
from model import PPO
from HAC import HAC
from env import MultiDomainEnv
from collections import namedtuple

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
Transition = namedtuple('Transition', ['state', 'action', 'a_log_prob', 'reward', 'next_state'])


def train(count, draw_flag):
    """
    count:迭代
    draw_flag:画图的标志，默认为False
    """
    #################### Hyperparameters ####################
    random_seed = 0
    render = False

    pickle_reward = []  # 每个pickle文件的奖励值
    episode_reward = []  # 每次迭代后的平均奖励值
    episode = []  # 迭代值
    pickle_step_num = []  # 每个pickle文件的步长情况
    episode_step_num = []  # 每次迭代后的平均步长

    state_dim = config.STATE_NUM  # 获取环境状态空间的维度，即观察的特征数量。
    action_dim = config.ACTION_NUM  # 获取环境动作空间的维度，即动作的数量。

    subgoal_action = [7, 8, 10, 11, 12, 16, 17, 18, 19, 20, 21, 22, 23, 24]  # sub objectives for each layer

    #########################################################

    if random_seed:
        print("Random Seed: {}".format(random_seed))
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)

    # creating HAC agent and setting parameters
    agent = PPO()
    # 加载分层强化学习参数
    hierarchical_policy = HAC(config.k_level, config.H, state_dim, action_dim, subgoal_action, render)

    # logging file:
    log_f = open("./route/routing.txt", 'a', encoding='utf-8')

    # training procedure
    for i_episode in range(1, config.EPOCH + 1):
        start_time = time.time()  # 起始时间
        hierarchical_policy.reward = 0
        hierarchical_policy.timestep = 0
        env = MultiDomainEnv(config.dataset.graph)
        for index, pkl_path in enumerate(config.dataset.pickle_file_path_yield()):
            if index == count:
                env.read_pickle_and_modify(pkl_path)  # 将pkl_graph的值覆盖graph
                for src, dst in config.START_END:
                    state = env.reset(src, dst)  # 获取当前的状态信息
                    state = utils.combine_state(state)
                    state = torch.FloatTensor(state)
                    reward_temp = 0
                    while True:
                        # 加载分层强化学习高层策略,标记子目标的状态
                        state = hierarchical_policy.run_HAC(env, config.k_level - 1, state,
                                                            False)
                        action, action_prob = agent.select_action(state)
                        action_node = utils.index_to_node(action)  # 将动作转成节点
                        # print("agent_index----->", agent_index)
                        # 底层策略，选择动作
                        next_state, reward, done, info = env.step(action_node, state)
                        trans = Transition(state, action, action_prob, reward, next_state)
                        agent.store_transition(trans)
                        reward_temp += reward

                        state = next_state
                        if done:
                            if len(agent.buffer) >= agent.batch_size: agent.update(i_episode)
                            # agent.writer.add_scalar('liveTime/livestep', i_epoch, global_step=i_epoch)
                            break
                    end_time = time.time()  # 结束的时间
                    utils.print_param(i_episode, index, reward_temp, info, end_time - start_time)  # 打印出每一步的路径信息
                    pickle_step_num.append(info.get("step_num"))  # 添加每一个pickle的步长
                    pickle_reward.append(reward_temp)  # 添加每一个pickle的奖励值
                    # # 将每个地图的路径进行保存
                    if i_episode == config.EPOCH:
                        log_f.write('{}\n'.format(info.get('path')[(src, dst)]))
                        log_f.flush()
        # 保存迭代次数、pickle奖励值、以及步长
        episode.append(i_episode)
        episode_reward.append(np.mean(pickle_reward))
        episode_step_num.append((np.mean(pickle_step_num)))

        pickle_reward.clear()  # 将其置空处理
        pickle_step_num.clear()  # 将其置空处理

    if draw_flag:
        picture_path, experimental_data_path = utils.create_picture_path()  # 存放实验数据和图片的路径
        utils.save_experimental_data(episode, experimental_data_path, "episode")
        utils.save_experimental_data(episode_reward, experimental_data_path, "mean_reward")
        utils.save_experimental_data(episode_step_num, experimental_data_path, "step_num")
        plot.draw_episode_data(picture_path, episode, episode_reward, "episode", "mean_reward", "mean_reward")
        plot.draw_episode_data(picture_path, episode, episode_step_num, "episode", "step_num", "step_num")


if __name__ == '__main__':
    for count in range(1, 2):
        train(count, True)
