# -*- coding: utf-8 -*-
"""
@File     : config.py
@Date     : 2022-12-25
@Author   : Terry_Li     --剑修一生痴绝处，无梦到此登城头。
IDE       : PyCharm
@Mail     : terry.ljq.dev@foxmail.com
"""
import utils

compare_path = utils.create_compare_path()  # 存放对比实验的路径
# delay
# x = range(8)
# y1 = [7.26, 7.56, 7.36, 7.76, 7.20, 7.25, 7.20, 7.56]  # PPONSA
# y2 = [7.78, 7.56, 7.36, 8.32, 7.25, 8.25, 8.20, 8.26]  # Dueling DQN
# y3 = [15.50, 10.56, 8.36, 10.32, 13.25, 10.86, 10.20, 18.86]  # OSPF
# y4 = [18.96, 17.56, 15.36, 30.32, 20.25, 19.96, 11.20, 19.96]  # DVRA
# y5 = [17.96, 11.56, 9.36, 24.32, 18.25, 15.96, 13.20, 15.96]  # LSRP
# ppo_ddqn_percentage, ppo_ospf_percentage, ppo_dvrp_percentage, ppo_lsrp_percentage = utils.calculate_average_percentage(
#     y1, y2, y3, y4, y5)
# print(ppo_ddqn_percentage, ppo_ospf_percentage, ppo_dvrp_percentage, ppo_lsrp_percentage)
# utils.plot_traffic_graph(compare_path, x, y1, y2, y3, y4,y5, "Hour", "Delay(ms)", "Mean delay throughout the day", "delay")

# throughput 吞吐量
# x = range(8)
# y1 = [133.76, 160.36, 170.26, 175.26, 175.26, 165.26, 165.25, 145.25]  # PPONSA
# y2 = [120.36, 150.26, 152.26, 165.26, 156.26, 145.26, 156.32, 136.25]  # Dueling DQN
# y3 = [100.35, 126.59, 98.56, 99.26, 120.35, 100.32, 99.58, 100.26]  # OSPF
# y4 = [95.25, 89.65, 99.25, 80.25, 56.25, 58.25, 95.25, 65.48] # DVRA
# y5 =  [105.25, 99.65, 89.25, 89.25, 107.25, 99.25, 95.25, 90.48]  # LSRP
# ppo_ddqn_percentage, ppo_ospf_percentage, ppo_dvrp_percentage, ppo_lsrp_percentage = utils.calculate_average_percentage(
#     y1, y2, y3, y4, y5)
# print(ppo_ddqn_percentage, ppo_ospf_percentage, ppo_dvrp_percentage, ppo_lsrp_percentage)
# utils.plot_traffic_graph(compare_path, x, y1, y2, y3, y4,y5, "时间", "平均吞吐量(Mbps)", "Mean  throughput throughout the day",
#                          "throughput")
# # Loss 丢包率
# x = range(8)
# y1 = [0.02, 0.01, 0.01, 0.01, 0.02, 0.01, 0.01, 0.03]  # PPOGG
# y2 = [0.01, 0.02, 0.00, 0.03, 0.07, 0.05, 0.02, 0.04]  # DDQN
# y3 = [0.02, 0.14, 0.25, 0.25, 0.36, 0.68, 0.14, 0.29]  # OSPF
# y4 = [0.04, 0.36, 0.18, 0.35, 0.23, 1.0, 0.38, 0.38]  # DVRA
# y5 = [0.24, 0.36, 0.26, 0.45, 0.44, 0.56, 0.38, 0.48]  # LSRP
# ppo_ddqn_percentage, ppo_ospf_percentage, ppo_dvrp_percentage, ppo_lsrp_percentage = utils.calculate_average_percentage(
#     y1, y2, y3, y4, y5)
# print(ppo_ddqn_percentage, ppo_ospf_percentage, ppo_dvrp_percentage, ppo_lsrp_percentage)
# utils.plot_traffic_graph(compare_path, x, y1, y2, y3, y4, y5, "时间", "平均丢包率(%)", "Mean loss throughout the day",
#                          "loss")

# Loss 错包率
# x = range(8)
# y1 = [0.02, 0.01, 0.01, 0.01, 0.12, 0.01, 0.01, 0.03]  # PPOGG
# y2 = [0.11, 0.02, 0.12, 0.33, 0.27, 0.05, 0.02, 0.04]  # Dueling DQN
# y3 = [0.44, 0.54, 0.35, 0.55, 0.26, 0.28, 0.24, 0.29]  # OSPF
# y4 = [0.36, 0.26, 0.28, 0.55, 0.49, 1.0, 0.28, 0.48]  # DVRA
# y5 = [0.24, 0.36, 0.56, 0.45, 0.64, 0.56, 0.38, 0.48]  # LSRP
# ppo_ddqn_percentage, ppo_ospf_percentage, ppo_dvrp_percentage, ppo_lsrp_percentage = utils.calculate_average_percentage(
#     y1, y2, y3, y4, y5)
# print(ppo_ddqn_percentage, ppo_ospf_percentage, ppo_dvrp_percentage, ppo_lsrp_percentage)
# utils.plot_traffic_graph(compare_path, x, y1, y2, y3, y4, y5, "时间", "平均错包率(%)",
#                          "Mean Pkt_error throughout the day",
#                          "pkt_{error}")

# Distance 距离
x = range(8)
y1 = [296.39, 286.25, 285.26, 278.59, 296.35, 286.59, 296.58, 300.25]  # PPONSA
y2 = [301.39, 299.75, 289.96, 279.79, 299.65, 296.99, 299.48, 305.95]  # DDQN
y3 = [361.84, 371.94, 391.84, 381.29, 451.84, 355.28, 351.74, 455.54]  # OSPF
y4 = [451.94, 391.84, 561.64, 391.84, 751.84, 651.74, 451.84, 759.84]  # DVRA
y5 = [351.54, 381.04, 461.59, 371.04, 651.29, 551.54, 351.29, 659.29]  # LSRP
# ppo_ddqn_percentage, ppo_ospf_percentage, ppo_dvrp_percentage, ppo_lsrp_percentage = utils.calculate_average_percentage(
#     y1, y2, y3, y4, y5)
# print(ppo_ddqn_percentage, ppo_ospf_percentage, ppo_dvrp_percentage, ppo_lsrp_percentage)
utils.plot_traffic_graph(compare_path, x, y1, y2, y3, y4, y5, "时间", "平均链路长度(m)", "Mean distance throughout the day",
                         "distance")
