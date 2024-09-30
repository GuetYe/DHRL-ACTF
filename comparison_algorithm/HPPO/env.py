# -*- coding: utf-8 -*-
"""
@File     : env.py
@Date     : 2023-12-13
@Author   : Terry_Li     --古有言胜不骄败不馁，何况本剑千年未尝一败。
IDE       : PyCharm
@Mail     : terry.ljq.dev@foxmail.com
"""
from matplotlib import pyplot as plt

import config
import utils
import numpy as np
import copy
import networkx as nx
from dataSet import DataSet


class UnicastEnv(object):
    """
    类的初始化需要，一个Graphd的图
    修改图时，调用self.modify_graph(graph),修改self.graph属性
    调用环境前，使用self.reset(start,ends),重置环境的属性、列表、矩阵等
    使用self.step(link)前进一步，修改列表、添加链路信息

    ---
    env = UnicastEnv(graph)
    env.reset(start_node, end_nodes)
    env.step(link)
    """

    def __init__(self, graph, numpy_type=config.NUMPY_TYPE, normalize=True):
        """
        单播路由环境设计
        :param graph: 创建一个无向图，存放链路的信息
        :param numpy_type: 数据类型
        :param normalize: 正则化标志位
        """

        self.graph = graph
        if graph is not None:  # 如果图不是空的
            self.nodes = sorted(list(graph.nodes()))  # 对图中的节点进行从小到大排序
            self.edges = sorted(graph.edges, key=lambda x: (x[0], x[1]))  # 对边的信息做一个处理
        else:
            self.nodes = None  # 将节点置空
            self.edges = None  # 将边置空处理

        self.numpy_type = numpy_type  # 数据类型

        self.Intelligent_routing_path = {}  # 用来存放搜索得到的路径

        self.start = None  # 起点
        self.ends = None  # 终点
        self.step_num = 1  # 步数

        # --------------------------存储链路信息的字典-----------------------------------#
        self.info_dict = {}  # 用于存放链路的一些信息

        # -----------------------------------所有矩阵置空处理-----------------------------#
        # self.adj_matrix = None  # 邻接矩阵
        self.free_bw_matrix = None  # 剩余带宽矩阵
        self.delay_matrix = None  # 时延矩阵
        self.loss_matrix = None  # 丢包率矩阵
        self.use_bw_matrix = None  # 使用带宽矩阵
        self.pkt_err_matrix = None  # 错包率矩阵
        self.pkt_drop_matrix = None  # 弃包个数矩阵
        self.distance_matrix = None  # 距离矩阵

        # ------------------------------------矩阵归一化处理-------------------------------#
        self.normal_free_bw_matrix = None  # 带宽矩阵归一化
        self.normal_delay_matrix = None  # 时延矩阵归一化
        self.normal_loss_matrix = None  # 丢包率矩阵归一化
        self.normal_use_bw_matrix = None  # 使用带宽矩阵归一化
        self.normal_pkt_err_matrix = None  # 错包率矩阵归一化
        self.normal_pkt_drop_matrix = None  # 弃包个数矩阵归一化
        self.normal_distance_matrix = None  # 距离矩阵归一化

        # ----------------------------------找出最大、最小带宽、时延、丢包率、距离的值-------------------#
        self.max_free_bw = np.finfo(np.float32).eps  # eps是取非负的最小值,避免计算溢出出错
        self.min_free_bw = 0

        self.max_delay = np.finfo(np.float32).eps
        self.min_delay = 0

        self.max_loss = np.finfo(np.float32).eps
        self.min_loss = 0

        self.max_use_bw = np.finfo(np.float32).eps  # eps是取非负的最小值,避免计算溢出出错
        self.min_use_bw = 0

        self.max_pkt_err = np.finfo(np.float32).eps  # eps是取非负的最小值,避免计算溢出出错
        self.min_pkt_err = 0

        self.max_pkt_drop = np.finfo(np.float32).eps  # eps是取非负的最小值,避免计算溢出出错
        self.min_pkt_drop = 0

        self.max_distance = np.finfo(np.float32).eps
        self.min_distance = 0

    def reset_set_params(self):
        """
           将所有的矩阵都置为空,重置所有的属性
        :return: None
        """
        self.Intelligent_routing_path = {}  # 用来存放搜索得到的路径

        self.step_num = 1  # 步数
        self.start = None  # 起点
        self.ends = None  # 终点

        # --------------------------存储链路信息的字典-----------------------------------#
        self.info_dict = {}  # 用于存放链路的一些信息

        # -----------------------------------所有矩阵置空处理-----------------------------#
        # self.adj_matrix = None  # 邻接矩阵
        self.free_bw_matrix = None  # 剩余带宽矩阵
        self.delay_matrix = None  # 时延矩阵
        self.loss_matrix = None  # 丢包率矩阵
        self.use_bw_matrix = None  # 使用带宽矩阵
        self.pkt_err_matrix = None  # 错包率矩阵
        self.pkt_drop_matrix = None  # 弃包个数矩阵
        self.distance_matrix = None  # 距离矩阵

        # ------------------------------------矩阵归一化处理-------------------------------#
        self.normal_free_bw_matrix = None  # 带宽矩阵归一化
        self.normal_delay_matrix = None  # 时延矩阵归一化
        self.normal_loss_matrix = None  # 丢包率矩阵归一化
        self.normal_use_bw_matrix = None  # 使用带宽矩阵归一化
        self.normal_pkt_err_matrix = None  # 错包率矩阵归一化
        self.normal_pkt_drop_matrix = None  # 弃包个数矩阵归一化
        self.normal_distance_matrix = None  # 距离矩阵归一化

        # ----------------------------------找出最大、最小带宽、时延、丢包率、距离的值-------------------#
        self.max_free_bw = np.finfo(np.float32).eps  # eps是取非负的最小值,避免计算溢出出错
        self.min_free_bw = 0

        self.max_delay = np.finfo(np.float32).eps
        self.min_delay = 0

        self.max_loss = np.finfo(np.float32).eps
        self.min_loss = 0

        self.max_use_bw = np.finfo(np.float32).eps  # eps是取非负的最小值,避免计算溢出出错
        self.min_use_bw = 0

        self.max_pkt_err = np.finfo(np.float32).eps  # eps是取非负的最小值,避免计算溢出出错
        self.min_pkt_err = 0

        self.max_pkt_drop = np.finfo(np.float32).eps  # eps是取非负的最小值,避免计算溢出出错
        self.min_pkt_drop = 0

        self.max_distance = np.finfo(np.float32).eps
        self.min_distance = 0

    def set_graph_params(self):
        """
           设置值
        :return: None
        """
        self.set_nodes()  # 设置节点,对节点进行排序处理
        self.set_edges()  # 调整边的序号，保证节点序号小的在左边
        self.set_adj_matrix()  # 设置邻接矩阵
        self.parse_graph_weight_data()  # 解析图的信息,并把权重信息存进矩阵之中,以字典的方式返回
        self.normalize_matrix_weight()  # 对矩阵中的参数进行归一化处理，并以字典的方式返回

    def set_nodes(self):
        """
        修改并返回nodes的值
        :return: self.nodes
        """
        self.nodes = sorted(list(self.graph.nodes()))  # 对节点进行从小到大排序

    def set_edges(self):
        """
           设置 边的属性
        :return: None
        """
        # [(1, 3), (1, 4), (1, 5), (1, 11),
        # (2, 4), (2, 5), (2, 6), (3, 6),
        # (4, 5), (4, 6), (4, 9), (4, 11), (4, 13),
        # (5, 6), (5, 7), (5, 8), (7, 9), (8, 9), (8, 14), (9, 10), (9, 13),
        # (10, 13), (11, 14), (12, 13), (12, 14)]
        edges = [(e[0], e[1]) if e[0] < e[1] else (e[1], e[0]) for e in self.graph.edges]  # 保证节点序号小的在左边
        self.edges = sorted(edges, key=lambda x: (x[0], x[1]))

    def set_adj_matrix(self):
        """
            设置并返回邻接矩阵,表示顶点之间相邻关系的矩阵
        :return:adj_m
        """
        adj_m = nx.adjacency_matrix(self.graph, self.nodes).todense()
        self.adj_matrix = np.array(adj_m, dtype=self.numpy_type)
        return self.adj_matrix

    def parse_graph_weight_data(self):
        """
           解析边的 bw, delay, loss ,distance数据存入矩阵中
        :return: bw_matrix, delay_matrix, loss_matrix, distance_matri。。。。。。。。
        """
        # 14*14的全 -1 矩阵
        _num = len(self.nodes)
        self.free_bw_matrix = -np.ones((_num, _num), dtype=self.numpy_type)
        self.delay_matrix = -np.ones((_num, _num), dtype=self.numpy_type)
        self.loss_matrix = -np.ones((_num, _num), dtype=self.numpy_type)
        self.use_bw_matrix = -np.ones((_num, _num), dtype=self.numpy_type)
        self.pkt_err_matrix = -np.ones((_num, _num), dtype=self.numpy_type)
        self.pkt_drop_matrix = -np.ones((_num, _num), dtype=self.numpy_type)
        self.distance_matrix = -np.ones((_num, _num), dtype=self.numpy_type)
        # ------------------ 将数据存储到矩阵中--------------------------#
        for edge in self.graph.edges.data():
            _start, _end, _data = edge
            #  先将剩余带宽的数据存储到矩阵中
            self.free_bw_matrix[_start - 1][_end - 1] = _data["free_bw"]

            self.free_bw_matrix[_end - 1][_start - 1] = _data["free_bw"]
            # 时延矩阵数据存储
            self.delay_matrix[_start - 1][_end - 1] = _data["delay"]
            self.delay_matrix[_end - 1][_start - 1] = _data["delay"]
            # loss 数据处理
            self.loss_matrix[_start - 1][_end - 1] = _data["loss"]
            self.loss_matrix[_end - 1][_start - 1] = _data["loss"]
            # 使用带宽数据处理
            self.use_bw_matrix[_start - 1][_end - 1] = _data["used_bw"]
            self.use_bw_matrix[_end - 1][_start - 1] = _data["used_bw"]
            # 包错误率数据处理
            self.pkt_err_matrix[_start - 1][_end - 1] = _data["pkt_err"]
            self.pkt_err_matrix[_end - 1][_start - 1] = _data["pkt_err"]
            # 弃包矩阵
            self.pkt_drop_matrix[_start - 1][_end - 1] = _data["pkt_drop"]
            self.pkt_drop_matrix[_end - 1][_start - 1] = _data["pkt_drop"]
            # 距离矩阵
            self.distance_matrix[_start - 1][_end - 1] = _data["distance"]
            self.distance_matrix[_end - 1][_start - 1] = _data["distance"]
        # 矩阵寻找最大和最小的值，同时把-1 的地方用0替换
        self.max_free_bw, self.min_free_bw, self.free_bw_matrix = utils.get_non_minus_one_max_min(self.free_bw_matrix)
        self.max_delay, self.min_delay, self.delay_matrix = utils.get_non_minus_one_max_min(self.delay_matrix)
        self.max_loss, self.min_loss, self.loss_matrix = utils.get_non_minus_one_max_min(self.loss_matrix)
        self.max_use_bw, self.min_use_bw, self.use_bw_matrix = utils.get_non_minus_one_max_min(self.use_bw_matrix)
        self.max_pkt_err, self.min_pkt_err, self.pkt_err_matrix = utils.get_non_minus_one_max_min(self.pkt_err_matrix)
        self.max_pkt_drop, self.min_pkt_drop, self.pkt_drop_matrix = utils.get_non_minus_one_max_min(
            self.pkt_drop_matrix)
        self.max_distance, self.min_distance, self.distance_matrix = utils.get_non_minus_one_max_min(
            self.distance_matrix)

        # 将计算得到的矩阵以字典的方式返回
        traffic_matrix = {"free_bw": self.free_bw_matrix, "delay": self.delay_matrix, "loss": self.loss_matrix,
                          "used_bw": self.use_bw_matrix, "pkt_err": self.pkt_err_matrix,
                          "pkt_drop": self.pkt_drop_matrix,
                          "distance": self.distance_matrix}

        return traffic_matrix

    def normalize_matrix_weight(self):
        """
        对矩阵中的数据进行归一化处理
        :return:归一化后的矩阵字典
        """
        self.normal_free_bw_matrix = utils.normalize_matrix(self.free_bw_matrix, self.max_free_bw,
                                                            self.min_free_bw, self.nodes)
        self.normal_delay_matrix = utils.normalize_matrix(self.delay_matrix, self.max_delay, self.min_delay, self.nodes)
        self.normal_loss_matrix = utils.normalize_matrix(self.loss_matrix, self.max_loss, self.min_loss, self.nodes)
        self.normal_use_bw_matrix = utils.normalize_matrix(self.use_bw_matrix, self.max_use_bw, self.min_use_bw,
                                                           self.nodes)
        self.normal_pkt_err_matrix = utils.normalize_matrix(self.pkt_err_matrix, self.max_pkt_err, self.min_pkt_err,
                                                            self.nodes)
        self.normal_pkt_drop_matrix = utils.normalize_matrix(self.pkt_drop_matrix, self.max_pkt_drop, self.min_pkt_drop,
                                                             self.nodes)
        self.normal_distance_matrix = utils.normalize_matrix(self.distance_matrix, self.max_distance, self.min_distance,
                                                             self.nodes)
        # 将计算得到的归一化矩阵以字典的方式返回
        traffic_normal_matrix = {"free_bw": self.normal_free_bw_matrix, "delay": self.normal_delay_matrix,
                                 "loss": self.normal_loss_matrix,
                                 "used_bw": self.normal_use_bw_matrix, "pkt_err": self.normal_pkt_err_matrix,
                                 "pkt_drop": self.normal_pkt_drop_matrix,
                                 "distance": self.normal_distance_matrix}

        return traffic_normal_matrix

    def update_graph(self, graph: nx.Graph):
        """
            修改 属性 并返回 graph
        :param graph: networkx的图
        :return: self.graph
        """
        self.graph = graph
        return self.graph

    def read_pickle_and_modify(self, pkl_graph_path):
        """
            读取图graph的pickle文件
            初始化所有参数
        :param pkl_graph_path: pickle文件路径
        :return: 图 nx.graph
        """
        pkl_graph = nx.read_gpickle(pkl_graph_path)  # 将pickle中的图读取出来
        self.update_graph(pkl_graph)
        return pkl_graph

    def get_node_adj_index_list(self, index):
        """
            根据当前节点获得邻居节点的索引号
        :param index: 节点索引
        :return: 邻居节点的索引号列表
        """
        adj_ids = np.nonzero(self.adj_matrix[index])  # return tuple
        adj_index_list = adj_ids[0].tolist()
        return adj_index_list

    def action_change_link(self, action):
        """
        根据动作转换成链路信息
        :param action: 0 ---> (1,2)
        :return:每个动作的边 (1,2)
        """
        edge = self.edges[action]
        return edge

    def reward_single_link_score(self, link):
        """
        计算每条链路的奖励
        :param link:链路信息(1,2) 需要进行一个简单的转换link(node1,node2)--->matrix(index1,index2)
        :return: reward
        reward = beta1*free_bw - beta2*delay-beta3*loss-beta4*pkt_drop-beta5*distance
        """
        e0, e1 = utils.node_to_index(link[0]), utils.node_to_index(link[1])
        free_bw = self.normal_free_bw_matrix[e0][e1]
        delay = self.normal_delay_matrix[e0][e1]
        loss = self.normal_loss_matrix[e0][e1]
        pkt_drop = self.normal_pkt_drop_matrix[e0][e1]
        distance = self.normal_distance_matrix[e0][e1]
        reward = config.BETA1 * free_bw - config.BETA2 * delay - \
                 config.BETA3 * loss - config.BETA4 * pkt_drop - \
                 config.BETA5 * distance

        return reward

    def mean_step_reward(self, score):
        """
        单步平均奖励
        """
        total_reward = 0
        total_reward += score
        mean_reward = total_reward / self.step_num
        return mean_reward

    def discount_reward(self, score):
        """
          对reward 进行discount 处理 score = score * discount ** step_num
        :param score: reward
        :return: score
        """
        score *= config.DISCOUNT1
        return score

    def _discount_reward(self, score):
        """
        对reward 进行discount 处理 score = score * discount ** step_num
        :param score: reward
        :return: score
        """
        score *= config.DISCOUNT2
        return score

    def reward_total_link_score(self, link):
        """
        计算整条链路的奖励值
        :param link:链路信息
        :return: total_reward
        """
        pass

    def judge_to_end(self, next_node, _state):
        """

        :param _state:
        :param next_node:
        :return:
        """
        if np.argwhere(_state == -1).flatten()[0].item() == next_node - 1:  # 抵达终点
            return True
        else:
            return False

    def reset(self, start, ends):
        """
        重置环境信息，初始化路径参数
        :param start: 开始节点
        :param ends: 终点的位置
        :return:当前环境的状态
        """
        self.reset_set_params()  # 重置参数
        self.set_graph_params()  # 重置图的边、节点信息、以及将权重存进矩阵中
        if self.graph is None:
            raise Exception("graph is None")
        self.start = copy.deepcopy(start)  # 开始的节点位置
        self.ends = copy.deepcopy(ends)  # 结束的终点位置
        self.Intelligent_routing_path[(start, ends)] = [self.start, ]  # 设置智能搜索路径
        _num = len(self.nodes)  # 获取节点的个数
        _state = np.zeros((_num, _num), dtype=self.numpy_type)  # 设置一个全0的矩阵
        idx = utils.node_to_index(self.start)
        _state[idx, idx] = 1  # 设置起点为1
        _idx = utils.node_to_index(self.ends)
        _state[_idx, _idx] = -1  # 把终点设置为-1
        return _state

    def step(self, action, _state):
        """
        更新环境
        :param _state
        :param action: 根据动作选择的链路 如1、2、3、4、5、6、7、8、9、10、11、12、13、14
        :return: next_state:set(), reward_score:float, terminal_done:bool
        """
        # 对下一跳节点做个转换，变成索引值
        agent_index = np.argwhere(_state == 1).flatten()[0].item()  # 获取智能体当前位置的索引
        next_jump_index = utils.node_to_index(action)
        # 先获取与当前状态相关的邻接节点
        agent_node = utils.index_to_node(agent_index)  # 获取智能体节点
        state_adj_index_list = self.get_node_adj_index_list(agent_index)  # 获取智能体邻接索引列表
        state_adj_nodes_list = utils.index_to_node(state_adj_index_list)  # 获取智能体邻接节点列表
        state_adj_edges_list = utils.get_adj_edges(agent_node, state_adj_nodes_list)  # 获取智能体邻接边列表
        link_flag = utils.compare_link((agent_node, action), state_adj_edges_list)  # 获取进行比对之后的标志位
        if link_flag:  # 所选的动作在邻接边中的情况
            self.Intelligent_routing_path[(self.start, self.ends)].append(action)
            # 判断是结束了还是往前进了一步
            judge_flag = self.judge_to_end(action, _state)
            if not judge_flag:
                route_done = False
                if action in self.Intelligent_routing_path.get((self.start, self.ends))[:-1]:  # 链路存在环路的情况
                    reward_score = self.reward_single_link_score((agent_node, action))  # 计算链路的奖励值
                    reward_score = self.discount_reward(reward_score)  # 进行奖励惩罚
                    next_state = utils.update_state(agent_index, next_jump_index, _state)  # 更新下一个状态
                    # next_state = self.reset(config.START, config.END)
                    # reward_score = -1
                else:
                    route_done = False
                    reward_score = self.reward_single_link_score((agent_node, action))   # 计算链路的奖励值
                    next_state = utils.update_state(agent_index, next_jump_index, _state)  # 更新下一个状态

            else:  # 抵达终点之后的情况
                route_done = True
                reward_score = self.reward_single_link_score((agent_node, action))  # 计算链路的奖励值
                reward_score += 10
                next_state = None  # 这个动作后状态，无动作

        else:  # 所选的动作不在邻接边的情况
            route_done = False
            # reward_score = self.reward_single_link_score((agent_node, action))  # 计算链路的奖励值
            reward_score = 1
            reward_score = self._discount_reward(reward_score)  # 进行奖励惩罚
            next_state = utils.update_state(agent_index, agent_index, _state)  # 不跟新状态 做一个惩罚值
            # print("惩罚之后的奖励------>:", reward_score)
        self.step_num += 1
        self.info_dict["step_num"] = self.step_num
        self.info_dict["path"] = self.Intelligent_routing_path
        return next_state, reward_score, route_done, self.info_dict


if __name__ == "__main__":
    # 调用数据集中的类进行处理
    test_dataset = DataSet(config.XML_TOPOLOGIES_PATH, config.DATASET_PATH)
    test_dataset.set_init_topology()

    test_env = UnicastEnv(test_dataset.graph)
    # test_env.set_graph_params()
    # for index, pkl_path in enumerate(test_dataset.pickle_file_path_yield()):
    #     # print(pkl_path)
    #     if index == 1:
    #         state = test_env.reset(config.START, config.END)
    #         test_env.read_pickle_and_modify(pkl_path)

    # print(test_env.edges)
    # print(test_env.adj_matrix)

    nx.draw(test_env.graph, with_labels=True)
    plt.show()
