# -*- coding: utf-8 -*-
"""
@File     : env.py
@Date     : 2023-12-28
@Author   : Terry_Li     --欲买桂花同载酒，终不似少年游。
IDE       : PyCharm
@Mail     : terry.ljq.dev@foxmail.com
"""

import config
import utils
import numpy as np
import copy
import networkx as nx
from dataSet import DataSet


if __name__ == "__main__":
    # 调用数据集中的类进行处理
    test_dataset = DataSet(config.XML_TOPOLOGIES_PATH, config.DATASET_PATH)
    test_dataset.set_init_topology()
    test_env = MultiDomainEnv(test_dataset.graph)

    for index, pkl_path in enumerate(test_dataset.pickle_file_path_yield()):
        # print(pkl_path)
        if index == 1:
            test_env.read_pickle_and_modify(pkl_path)
            test_env.reset(1, 29)
    edges = test_env.edges

    # print(test_env.adj_matrix)
    #
    # nx.draw(test_env.graph, with_labels=True)
    # plt.show()
