# -*- coding: utf-8 -*-
"""
@File     : network_shortest_path.py
@Date     : 2022-11-14
@Author   : Terry_Li  -- 人生就是在不断结交新的朋友。
IDE       : VS Code
@Mail     : terry.ljq.dev@foxmail.com
"""
import os
import pickle
import networkx as nx
from pathlib import Path

def create_pickle_path():
    # 创建一个用来存放pickle文件的路径
    pickle_path = Path(r"D:\rerouting-drl\hdrl-model\Hierarchical-Actor-Critic-HAC-PyTorch-master\构造数据集\pickle")
    if not pickle_path.exists():
        pickle_path.mkdir(parents=True, exist_ok=True)
    return pickle_path

def convert_file_to_pickle(input_dir, output_dir):
    """
    将文本文件转成pickle文件
    """
    for filename in os.listdir(input_dir):
        if filename.endswith(".txt"):
            # 构造完整的文件路径
            txt_path = os.path.join(input_dir, filename)
            pkl_path = os.path.join(output_dir, filename.replace(".txt", ".pkl"))

            # 创建图对象
            G = nx.Graph()

            # 读取文本文件并解析为图的边和权重
            with open(txt_path, 'r') as ft:
                lines = ft.readlines()
                for line in lines:
                    line = line.strip()
                    if line:
                        try:
                            # 解析元组
                            link_data = eval(line)
                            node1, node2, attrs = link_data
                            G.add_edge(node1, node2, **attrs)
                        except Exception as e:
                            print(f"Error parsing line: {line}")
                            print(f"Error: {e}")

            # 保存为pickle文件
            with open(pkl_path, 'wb') as fp:
                pickle.dump(G, fp)
            print(f"Converted {filename} to {pkl_path}")

if __name__ == "__main__":
    input_dir = r"D:\rerouting-drl\hdrl-model\Hierarchical-Actor-Critic-HAC-PyTorch-master\构造数据集\modified_data"
    output_dir = create_pickle_path()
    convert_file_to_pickle(input_dir, output_dir)