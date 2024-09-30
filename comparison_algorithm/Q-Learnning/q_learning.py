import os
import sys
import plot

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from env import MultiDomainEnv
import config
import numpy as np
from collections import defaultdict


class QLearning:
    def __init__(self, num_states, num_actions, learning_rate=0.01, discount_factor=0.9, exploration_rate=0.1):
        self.num_states = num_states
        self.num_actions = num_actions
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.q_table = defaultdict(lambda: [0]*self.num_states)

    def choose_action(self, state):
        if np.random.uniform(0, 1) < self.exploration_rate:
            # Explore: choose a random action
            action = np.random.choice(self.num_actions)
        else:
            # Exploit: choose the action with the highest Q-value
            action = np.argmax(self.q_table[state])
            # print(action)
        return action

    def update_q_table(self, state, action, reward, next_state):
        current_q = self.q_table[state][action]
        # 贝尔曼方程更新
        new_q = reward + self.discount_factor * np.max(self.q_table[next_state])
        self.q_table[state][action] += self.learning_rate * (new_q - current_q)


if __name__ == "__main__":
    # Initialize Q-Learning agent
    agent = QLearning(num_states=30, num_actions=30)
    episode_list = []
    episode_reward = []
    # Initialize maze environment
    env = MultiDomainEnv(config.dataset.graph)
    # Run episodes of training
    for episode in range(1000):
        for index, pkl_path in enumerate(config.dataset.pickle_file_path_yield()):
            if index == 2:
                env.read_pickle_and_modify(pkl_path)  # 将pkl_graph的值覆盖graph
                for src, dst in config.START_END:
                    state = env.reset(src, dst)  # 获取当前的状态信息
                    total_reward = 0
                    while True:
                        action = agent.choose_action(str(state))
                        next_state, reward, done, route = env.step(action, state)
                        agent.update_q_table(str(state), action, reward, str(next_state))
                        state = next_state
                        total_reward += reward
                        # print(route)
                        # 当到达终点就终止游戏开始新一轮训练
                        if done:
                            break
                    episode_list.append(episode)
                    episode_reward.append(total_reward)
                    print(route)
                    print("Episode {}: Total Reward = {}".format(episode, total_reward))
    plot.draw_episode_data(episode_list, episode_reward, "episode", "mean_reward")
