import config
import xml.etree.ElementTree as ET
import os
import torch
from gpu import gpu
import numpy as np
import networkx
from utils import normalize_value


def prepare_data(pickle_graph):
    data = []
    for weight in pickle_graph.edges.data():
        data.append(weight)
    # 提取 "free_bw" 和 "delay" 中的参数
    free_bw_values = [item[2]['free_bw'] for item in data]
    delay_values = [item[2]['delay'] for item in data]
    loss_value = [item[2]['loss'] for item in data]
    used_bw_value = [item[2]['used_bw'] for item in data]
    pkt_err_value = [item[2]['pkt_err'] for item in data]
    pkt_drop_value = [item[2]['pkt_drop'] for item in data]
    distance_value = [item[2]['distance'] for item in data]
    signal_strength_value = [item[2]['distance'] - 100 for item in data]
    input_sequence = [free_bw_values, delay_values, loss_value, used_bw_value, pkt_err_value,
                      pkt_drop_value, distance_value, signal_strength_value]

    # 创建矩阵
    matrix_data = np.array(input_sequence).T
    # 转换为 PyTorch tensor
    tensor_data = torch.tensor(matrix_data, dtype=torch.float32)

    # 进行归一化处理
    normalized_data = normalize_value(tensor_data)
    return normalized_data.to(gpu())


class DataSet(object):
    def __init__(self, xml_topology_path, dataset_path):
        """
        先对数据进行预处理
        :param xml_topology_path: 搭建topo的路径
        :param dataset_path: 数据集的路径
        """
        self.edges_num = None  # 设置边为空
        self.nodes_num = None  # 设置节点为空
        self.graph = None  # 设置图为空
        self.xml_topology_path = xml_topology_path  # xml_topology地址
        self.dataset_path = dataset_path  # 数据集的地址

    def set_init_topology(self):
        """
        1.解析xml文件
        2.设置self.graph
             self.nodes_num
             self.edges_num
        :return: None
        """
        graph, nodes_num, edges_num = self.parse_xml_topology()
        self.graph = graph
        self.nodes_num = nodes_num
        self.edges_num = edges_num

    def parse_xml_topology(self):
        """
        parse topology from topology.xml
        :return: topology graph, networkx.Graph()
                 nodes_num,  int
                 edges_num, int
        """
        tree = ET.parse(self.xml_topology_path)
        root = tree.getroot()
        topo_element = root.find("topology")
        graph = networkx.Graph()
        for child in topo_element.iter():
            # parse nodes
            if child.tag == 'node':
                node_id = int(child.get('id'))
                graph.add_node(node_id)
            # parse link
            elif child.tag == 'link':
                from_node = int(child.find('from').get('node'))
                to_node = int(child.find('to').get('node'))
                graph.add_edge(from_node, to_node)

        nodes_num = len(graph.nodes)
        edges_num = len(graph.edges)

        # print('nodes: ', nodes_num, '\n', graph.nodes, '\n',
        #       'edges: ', edges_num, '\n', graph.edges)
        return graph, nodes_num, edges_num

    def pickle_file_path_yield(self, s=5, n: int = 630, step: int = 2):
        """
        生成保存的txt文件的路径, 按序号递增的方式生成
        :param s: 开始index
        :param n: 结束index
        :param step: 间隔
        """
        a = os.listdir(self.dataset_path)
        assert n < len(a), "n should small than len(a)"
        # print(len([x.split('-')[0] for x in a]))  # 切割文件名字第一个数值标号
        b = sorted(a, key=lambda x: int(x.split("-")[0]))  # 对文件名首字标号进行排序处理,返回的是文件名列表
        for p in b[s:n:step]:
            yield self.dataset_path / p

    def prepare_data(self, graph):
        """
        对实验数据进行解析，将数据存放在矩阵当中
        """
        data = []
        for weight in graph.edges.data():
            data.append(weight)
        # 提取 "free_bw" 和 "delay" 中的参数
        free_bw_values = [item[2]['free_bw'] for item in data]
        delay_values = [item[2]['delay'] for item in data]
        loss_value = [item[2]['loss'] for item in data]
        used_bw_value = [item[2]['used_bw'] for item in data]
        pkt_err_value = [item[2]['pkt_err'] for item in data]
        pkt_drop_value = [item[2]['pkt_drop'] for item in data]
        distance_value = [item[2]['distance'] for item in data]
        # input_sequence = [free_bw_values, delay_values, loss_value, used_bw_value, pkt_err_value,
        #                   pkt_drop_value, distance_value]
        input_sequence = [free_bw_values]
        target_sequence = [free_bw_values]
        # 创建矩阵
        input_matrix_data = np.array(input_sequence).T
        output_matrix_data = np.array(target_sequence).T
        # 转换为 PyTorch tensor
        input_tensor_data = torch.tensor(input_matrix_data, dtype=torch.float32)
        output_tensor_data = torch.tensor(output_matrix_data, dtype=torch.float32)
        return input_tensor_data.to(gpu()), output_tensor_data.to(gpu())


if __name__ == "__main__":
    test_dataset = DataSet(config.XML_TOPOLOGIES_PATH, config.DATASET_PATH)
    test_dataset.set_init_topology()
    test_dataset.pickle_file_path_yield()
    print(test_dataset.nodes_num)
