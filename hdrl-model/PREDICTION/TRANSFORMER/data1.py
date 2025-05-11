import os
from pathlib import Path
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import networkx
from itertools import islice
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET  # 解析xml树形结构
from gpu import gpu
from utils import normalize_value
from config import DATASET_PATH

def normalize_matrix(matrix):
    """
    对输入矩阵按列进行Min-Max归一化
    :param matrix: numpy数组，形状为(N_samples, N_features)
    :return: 归一化后的矩阵
    """
    # 计算每列的最小值和最大值
    min_vals = np.min(matrix, axis=0, keepdims=True)  # 形状(1, N_features)
    max_vals = np.max(matrix, axis=0, keepdims=True)
    
    # 防止除零，添加极小值
    epsilon = 1e-8
    normalized = (matrix - min_vals) / (max_vals - min_vals + epsilon)
    return normalized


def get_node_neighbors(graph):
    """
    找出所有节点相对应的邻居节点
    :return:
    """
    m_graph = graph
    nodes = sorted(m_graph.nodes())
    neighbors = {}  # 所有节点的邻居节点
    for node in nodes:
        neighbors[node] = list(graph.neighbors(node))
    return neighbors


def file_path_yield(file_dir, ista=0, iend=100, step=1):
    """
    生成迭代器，自定义起始点、终点和步长
    :param file_dir: 数据目录
    :param ista: 开始文件index
    :param iend: 结束文件index
    :param step: 读取步长
    :return:
    """
    _dir = os.listdir(file_dir)
    # print('pkl num={}'.format(len(_dir)))
    assert iend <= len(_dir), "n should small than len(a)"  # 判断是否超出数据文件总数
    file_names = sorted(_dir, key=lambda x: int(x.split('-')[0]))  # 按文件序号排序
    for name in file_names[ista:iend:step]:
        yield file_dir / name

def seq_file_path_yield(file_dir, ista=0, iend=100, step=1):
    """
    生成迭代器，自定义起始点、终点和步长
    :param file_dir: 数据目录
    :param ista: 开始文件index
    :param iend: 结束文件index
    :param step: 读取步长
    :return:
    """
    _dir = os.listdir(file_dir)
    # print('pkl num={}'.format(len(_dir)))
    assert iend <= len(_dir), "n should small than len(a)"  # 判断是否超出数据文件总数
    file_names = sorted(_dir, key=lambda x: int(x.split('-')[0]))  # 按文件序号排序
    return batch_iterator(file_names[ista:iend],step)

def batch_iterator(data, batch_size):
    """
    生成一个迭代器，每次从列表中读取 batch_size 个元素。
    :param data: 输入列表
    :param batch_size: 每次读取的元素数量
    :return: 每次返回 batch_size 个元素的迭代器
    """
    it = iter(data)  # 将列表转换为迭代器
    while True:
        batch = list(islice(it, batch_size))  # 每次提取 batch_size 个元素
        if not batch or len(batch) < batch_size:  # 如果 batch 为空，说明已经遍历完所有数据
            break
        yield batch

def read_pickle(pickle_path):
    """
    读取pickle并转化为graph
    :param pickle_path:
    :return:
    """
    pkl_graph = networkx.read_gpickle(pickle_path)
    # print(pkl_graph.edges.data())
    # networkx.draw(pkl_graph, with_labels=True)
    # plt.show()
    return pkl_graph

def prepare_data(pkl_graph):
    data = []
    for weight in pkl_graph.edges.data():
        data.append(weight)
    # 提取 "free_bw" 和 "delay" 中的参数
    free_bw_values = [item[2]['free_bw'] for item in data]
    # bw = [item[2]['free_bw'] for item in data]
    delay_values = [item[2]['delay'] for item in data]
    loss_value = [item[2]['loss'] for item in data]
    used_bw_value = [item[2]['used_bw'] for item in data]
    pkt_err_value = [item[2]['pkt_err'] for item in data]
    pkt_drop_value = [item[2]['pkt_drop'] for item in data]
    distance_value = [item[2]['distance'] for item in data]
    signal_strength_value = [item[2]['distance'] - 100 for item in data]
    input_sequence = [free_bw_values, delay_values, loss_value, used_bw_value, pkt_err_value,
                      pkt_drop_value, distance_value, signal_strength_value]
    # input_sequence = [bw]  
    target_sequence =  [free_bw_values, delay_values, loss_value, used_bw_value, pkt_err_value,
                      pkt_drop_value, distance_value, signal_strength_value]
    # 创建矩阵
    input_matrix_data = np.array(input_sequence).T
    output_matrix_data = np.array(target_sequence).T
    # 转换为 PyTorch tensor
    input_tensor_data = torch.tensor(input_matrix_data, dtype=torch.float32)
    output_tensor_data = torch.tensor(output_matrix_data, dtype=torch.float32)

    return  normalize_matrix(input_matrix_data)

def prepare_seq_data(seq_pkl_graph):
    seq_inputs = ''
    target = ''
    for index, graph in enumerate(seq_pkl_graph):
        input_tensor_data , _ = prepare_data(graph)
        if  index == len(seq_pkl_graph)-1:
            target, _ = prepare_data(graph)
        else :
            if torch.is_tensor(seq_inputs) :
                seq_inputs = torch.cat((seq_inputs, input_tensor_data.unsqueeze(0)), dim=0)
            else:
                # seq_inputs = input_tensor_data.unsqueeze(0)
                seq_inputs = input_tensor_data

    return seq_inputs, target

def read_file_to_datasets(file_path, seq_lenth, ista, iend):
    graph_data = [read_pickle(pkl_path) for pkl_path in  file_path_yield(file_path, ista=ista, iend=iend, step=1)]
    data = [prepare_data(graph) for graph in graph_data]
    loader = SequenceDataset(data, seq_lenth)
    return loader


class SequenceDataset(Dataset):
    def __init__(self, data, seq_length):
        """
        Args:
            data (list or numpy array): 输入数据，通常是一个时间序列。
            seq_length (int): 每个序列的长度。
        """
        self.data = data
        self.seq_length = seq_length

    def __len__(self):
        # 返回数据集中序列的数量
        return len(self.data) - self.seq_length

    def __getitem__(self, idx):
        # 返回一个序列和对应的目标值

        sequence = self.data[idx:idx + self.seq_length]
        target = self.data[idx + self.seq_length]
        return torch.tensor(sequence, dtype=torch.float32), torch.tensor(target, dtype=torch.float32)



if __name__ == '__main__':

    a = read_file_to_datasets(DATASET_PATH, 5, 1, 250)

    for batch_idx, (inputs, targets) in enumerate(a):
        print(f"Batch {batch_idx}:")
        print(f"Input shape: {inputs}")  # 应为 [batch_size, seq_len, num_features]
        print(f"Target shape: {targets.shape}") # 应为 [batch_size, num_features]

