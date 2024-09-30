# import numpy as np
#
# # Create a 3x3 matrix with symbolic elements
# matrix = np.array([
#     ['e_{1,1}', 'e_{1,2}', 'e_{1,3}'],
#     ['e_{2,1}', 'e_{2,2}', 'e_{2,3}'],
#     ['e_{3,1}', 'e_{3,2}', 'e_{3,3}']
# ])
#
# # Access the element e_{2,3}^{1,2}
# element = matrix[1, 2]
#
# print(matrix)

# str1 = "destination"
# print(str1.upper())

# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.distributions import Categorical
# import gym
# import numpy as np
#
#
# class Policy(nn.Module):
#     def __init__(self, input_size, output_size):
#         super(Policy, self).__init__()
#         self.fc = nn.Linear(input_size, output_size)
#
#     def forward(self, state):
#         return torch.softmax(self.fc(state), dim=-1)
#
#
# class HierarchicalPolicy(nn.Module):
#     def __init__(self, high_level_policy, low_level_policies):
#         super(HierarchicalPolicy, self).__init__()
#         self.high_level_policy = high_level_policy
#         self.low_level_policies = low_level_policies
#
#     def forward(self, state):
#         high_level_action_probs = self.high_level_policy(state)
#         high_level_action = Categorical(high_level_action_probs).sample()
#         low_level_action_probs = self.low_level_policies[high_level_action](state)
#         low_level_action = Categorical(low_level_action_probs).sample()
#         return high_level_action, low_level_action
#
#
# # Initialize environment
# env = gym.make('CartPole-v1')
# state_size = env.observation_space.shape[0]
# high_level_action_size = 2  # Assuming 2 high-level actions
# low_level_action_size = env.action_space.n
#
# # Initialize policies
# high_level_policy = Policy(state_size, high_level_action_size)
# low_level_policies = [Policy(state_size, low_level_action_size) for _ in range(high_level_action_size)]
# hierarchical_policy = HierarchicalPolicy(high_level_policy, low_level_policies)
#
# # PPO hyperparameters
# epochs = 1000
# max_steps = 500
# gamma = 0.99
# epsilon_clip = 0.2
# lr = 0.001
#
# optimizer = optim.Adam(hierarchical_policy.parameters(), lr=lr)
#
# for epoch in range(epochs):
#     state = env.reset()
#     high_level_actions = []
#     low_level_actions = []
#     rewards = []
#     log_probs = []
#
#     for step in range(max_steps):
#         high_level_action, low_level_action = hierarchical_policy(torch.tensor(state, dtype=torch.float32))
#         high_level_actions.append(high_level_action)
#         low_level_actions.append(low_level_action)
#
#         state, reward, done, _ = env.step(low_level_action.item())
#         rewards.append(reward)
#
#         high_level_action_probs = hierarchical_policy.high_level_policy(torch.tensor(state, dtype=torch.float32))
#         low_level_action_probs = hierarchical_policy.low_level_policies[high_level_action.item()](
#             torch.tensor(state, dtype=torch.float32))
#
#         high_level_log_prob = torch.log(high_level_action_probs[high_level_action])
#         low_level_log_prob = torch.log(low_level_action_probs[low_level_action])
#         log_probs.append(high_level_log_prob + low_level_log_prob)
#
#         if done:
#             break
#
#     discounted_rewards = []
#     running_add = 0
#     for r in reversed(rewards):
#         running_add = running_add * gamma + r
#         discounted_rewards.insert(0, running_add)
#
#     discounted_rewards = torch.tensor(discounted_rewards, dtype=torch.float32)
#
#     old_log_probs = torch.stack(log_probs).detach()
#     advantage = discounted_rewards - old_log_probs
#
#     for _ in range(3):  # Number of optimization steps
#         high_level_action_probs = hierarchical_policy.high_level_policy(torch.tensor(state, dtype=torch.float32))
#         low_level_action_probs = hierarchical_policy.low_level_policies[high_level_action.item()](
#             torch.tensor(state, dtype=torch.float32))
#
#         high_level_entropy = Categorical(high_level_action_probs).entropy()
#         low_level_entropy = Categorical(low_level_action_probs).entropy()
#
#         high_level_log_prob = torch.log(high_level_action_probs[high_level_action])
#         low_level_log_prob = torch.log(low_level_action_probs[low_level_action])
#
#         high_level_ratio = torch.exp(high_level_log_prob - old_log_probs[0])
#         low_level_ratio = torch.exp(low_level_log_prob - old_log_probs[1])
#
#         high_level_advantage = advantage[0].unsqueeze(-1)
#         low_level_advantage = advantage[1].unsqueeze(-1)
#
#         high_level_surrogate1 = high_level_ratio * high_level_advantage
#         high_level_surrogate2 = torch.clamp(high_level_ratio, 1 - epsilon_clip, 1 + epsilon_clip) * high_level_advantage
#         high_level_surrogate = -torch.min(high_level_surrogate1, high_level_surrogate2)
#
#         low_level_surrogate1 = low_level_ratio * low_level_advantage
#         low_level_surrogate2 = torch.clamp(low_level_ratio, 1 - epsilon_clip, 1 + epsilon_clip) * low_level_advantage
#         low_level_surrogate = -torch.min(low_level_surrogate1, low_level_surrogate2)
#
#         loss = high_level_surrogate + low_level_surrogate - 0.01 * (high_level_entropy + low_level_entropy)
#
#         optimizer.zero_grad()
#         loss.sum().backward()
#         optimizer.step()
#
#     if epoch % 1 == 0:
#         print(f'Epoch [{epoch}/{epochs}], Total Reward: {sum(rewards)}')
#
# env.close()

# import numpy as np
#
# # 假设 _state 是您的矩阵
# state = np.zeros((30, 30), dtype=int)  # 设置一个全0的矩阵
#
# # 使用条件判断检查是否存在值为 2 的元素
# print(state.shape)
# if np.any(state == 2):
#     print("矩阵中存在值为 2 的元素")
# else:
#     print("矩阵中不存在值为 2 的元素")
import torch

# 假设有9个30*30的矩阵
matrix1 = torch.rand(30, 30)
matrix2 = torch.rand(30, 30)
matrix3 = torch.rand(30, 30)
matrix4 = torch.rand(30, 30)
matrix5 = torch.rand(30, 30)
matrix6 = torch.rand(30, 30)
matrix7 = torch.rand(30, 30)
matrix8 = torch.rand(30, 30)
matrix9 = torch.rand(30, 30)

# 将这些矩阵按照所需的方式进行拼接
concatenated_tensor = torch.cat((matrix1.unsqueeze(0), matrix2.unsqueeze(0), matrix3.unsqueeze(0),
                                 matrix4.unsqueeze(0), matrix5.unsqueeze(0), matrix6.unsqueeze(0),
                                 matrix7.unsqueeze(0), matrix8.unsqueeze(0), matrix9.unsqueeze(0)), dim=0)

# 打印结果
print(concatenated_tensor)




