# -*- coding: utf-8 -*-
"""
@File     : train.py
@Date     : 2023-12-25
@Author   : Terry_Li     --既然选择了远方，便只顾风雨兼程。
IDE       : PyCharm
@Mail     : terry.ljq.dev@foxmail.com
"""
import utils
import config
import torch
import numpy as np
from model import PPO
from collections import namedtuple

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class HAC:
    def __init__(self, k_level, H, state_dim, action_dim, goals, render):
        """
         k_level:强化学习层数
         H: 在H时间内判断所选动作是不是子目标
         state_dim: 状态维度
         action_dim: 动作维度
         render: 是否渲染环境
         lr: 学习率
        """
        self.replay_buffer = []
        # adding bottom level strategy 添加底层策略
        self.HAC = [PPO()]
        self.transition = namedtuple('Transition', ['state', 'action', 'a_log_prob', 'reward', 'next_state'])

        # adding remaining levels strategy 添加其余层策略
        for _ in range(k_level - 1):
            self.HAC.append(PPO())

        # set some parameters
        self.k_level = k_level
        self.H = H
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.render = render

        # logging parameters
        self.goals = goals  # 子目标对象
        self.height_reward = 0  # 高层奖励值
        self.timestep = 0

    def check_goal(self, state, sub_goal: list):
        """
        state: 当前智能体的状态
        sub_goal: 子目标集合
        是否为子目标的状态
        """
        if state is not None:
            agent_index = np.argwhere(state == 1).flatten()[0].item()  # 根据状态获取当前智能体的索引值
            # print("111--->",agent_index)
            agent_node = utils.index_to_node(agent_index)  # 将索引切换成节点
            if agent_node in sub_goal:
                return sub_goal.index(agent_node)  # 返回对应目标的索引
            return - 1
        else:
            return None

    def run_HAC(self, env, i_level, state, is_subgoal_test):
        """
        env:强化学习环境
        i_level: 分层强化学习的层数
        state: 智能体的状态
        goal: 目标集合
        is_subgoal_test: 判断是否为子目标的标志位
        目的：输出最优的子目标状态

        ---- 分层强化学习的核心函数 ---
        """

        #   <================ 高层策略 ================>
        if i_level > 0:
            sub_goal_index = self.check_goal(state, self.goals)
            if sub_goal_index >= 0:
                next_state = utils.update_subgoal_state(self.goals[sub_goal_index] - 1, state)  # 将子目标的位置进行标记
            else:
                next_state = state  # 如果不是子目标，那么就直接用之前的状态即可

        next_state = state

        # take random action if not subgoal testing
        # if not is_subgoal_test:
        #     action = np.random.randint(1, config.ACTION_NUM + 1)  # 随机选取一个动作

        # Determine whether to test subgoal (action)
        # if np.random.random_sample() < config.lamda:
        #     is_next_subgoal_test = True

        # Pass subgoal to lower level
        # 通过子目标抵达底层策略
        # next_state, done = self.run_HAC(env, i_level - 1, state, is_next_subgoal_test)

        # if subgoal was tested but not achieved, add subgoal testing transition
        # if is_next_subgoal_test and not self.check_goal(state, self.goals):
        #     PPO().store_transition(self.transition(state, action, action_prob, 0, next_state))  # 未达到子目标给予0的奖励值
        # if is_next_subgoal_test and self.check_goal(state, self.goals):
        #     PPO().store_transition(self.transition(state, action, action_prob, 2, next_state))  # 达到子目标后给予+2的奖励值

        # for hindsight action transition
        # state = next_state

        #   <================ 低层策略 ================>
        # else:
        #     # take random action if not subgoal testing
        #     # if not is_subgoal_test:
        #     #     action = np.random.randint(1, config.ACTION_NUM + 1)  # 随机选取一个动作
        #
        #     # take primitive action
        #
        #     next_state, reward, done, info = env.step(action, state)  # 与环境交互
        #
        #     trans = self.transition(state, action, action_prob, reward, next_state)
        #     PPO().store_transition(trans)
        #     self.reward += reward  # 与环境进行交互

        # print(self.reward)
        # print(next_state)

        # <================ finish one step/transition ================>

        # check if goal is achieved
        # goal_achieved = self.check_goal(state, self.goals)
        # print(goal_achieved)
        #
        # # hindsight action transition
        # if goal_achieved:
        #     PPO().store_transition(self.transition(state, action, action_prob, 10, next_state))
        # else:
        #     PPO().store_transition(self.transition(state, action, action_prob, 0, next_state))

        # copy for goal transition
        # goal_transitions.append([state, action, -1.0, next_state, None, 0.99, float(done)])

        #   <================ finish H attempts ================>

        # hindsight goal transition
        # last transition reward and discount is 0
        # goal_transitions[-1][2] = 0.0
        # goal_transitions[-1][5] = 0.0
        # for transition in goal_transitions:
        #     # last state is goal for all transitions
        #     transition[4] = next_state
        #     self.replay_buffer[i_level].add(tuple(transition))

        return next_state
