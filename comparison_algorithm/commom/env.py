# -*- coding: utf-8 -*-
"""
@File     : env.py
@Date     : 2022-10-13
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

if __name__ == "__main__":
    # 调用数据集中的类进行处理
    test_dataset = DataSet(config.XML_TOPOLOGIES_PATH, config.DATASET_PATH)
    test_dataset.set_init_topology()

    test_env = UnicastEnv(test_dataset.graph)
    # test_env.set_graph_params()
    for index, pkl_path in enumerate(test_dataset.pickle_file_path_yield()):
        # print(pkl_path)
        if index == 1:
            state = test_env.reset(config.START, config.END)
            test_env.read_pickle_and_modify(pkl_path)

    # print(test_env.edges)
    # print(test_env.adj_matrix)

    nx.draw(test_env.graph, with_labels=True)
    plt.show()
