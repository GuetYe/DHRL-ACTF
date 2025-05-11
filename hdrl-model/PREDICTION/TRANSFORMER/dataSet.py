import os
from pathlib import Path
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import networkx as nx
from itertools import islice
import xml.etree.ElementTree as ET
from gpu import gpu
from utils import normalize_value
import config

           
# 辅助函数部分
def parse_xml_topology(xml_path):
    """解析XML拓扑文件生成网络图"""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    graph = nx.Graph()
    
    for elem in root.iter():
        if elem.tag == 'node':
            node_id = int(elem.get('id'))
            graph.add_node(node_id)
        elif elem.tag == 'link':
            from_node = int(elem.find('from').get('node'))
            to_node = int(elem.find('to').get('node'))
            graph.add_edge(from_node, to_node)
    
    print(f"Parsed topology with {len(graph.nodes)} nodes and {len(graph.edges)} edges")
    return graph

def file_path_generator(data_dir, start=0, end=100, step=1):
    """生成文件路径迭代器"""
    files = sorted(os.listdir(data_dir), key=lambda x: int(x.split('-')[0]))
    for f in files[start:end:step]:
        yield Path(data_dir) / f

def load_graph_data(pkl_path):
    """从pickle文件加载图数据"""
    return nx.read_gpickle(pkl_path)

def prepare_features(graph, fixed_edges=58, num_features=3):
    """提取边特征并展平为向量"""
    # 提取原始边特征（每条边3个特征）
    edge_features = []
    for _, _, d in graph.edges.data():
        # 添加特征筛选和组合
        feat = [
            d['free_bw'],
            # d['delay'] * 0.01,  # 延迟特征缩放
            # np.log(d['loss'] + 1e-6)  # 对loss取对数防止数值爆炸
        ]
        edge_features.append(feat)
    
    # 填充/截断
    if len(edge_features) < fixed_edges:
        edge_features += [[0.0]] * (fixed_edges - len(edge_features))
    else:
        edge_features = edge_features[:fixed_edges]
    
    # 转换为Tensor
    features = torch.tensor(edge_features, dtype=torch.float32)  # [58,3]
    
    # 按特征维度归一化
    mean = features.mean(dim=0, keepdim=True)
    std = features.std(dim=0, keepdim=True)
    normalized = (features - mean) / (std + 1e-6)
    flattened =normalized.view(-1)  # [58*3=174]
    
    return flattened.to(gpu())  # 归一化后形状 [174]

class TopologyDataset(Dataset):
    def __init__(self, data_dir, seq_length=5):
        self.file_paths = sorted(Path(data_dir).glob("*.pkl"))[:1000]
        self.seq_length = seq_length
        
        # 加载所有时间步的特征向量 [total_steps, 174]
        self.feature_sequence = torch.stack([
            prepare_features(nx.read_gpickle(p))
            for p in self.file_paths
        ])
    
    def __len__(self):
        return len(self.feature_sequence) - self.seq_length
    
    def __getitem__(self, idx):
        return (
            self.feature_sequence[idx:idx+self.seq_length],  # [seq_len, 174]
            self.feature_sequence[idx+self.seq_length]       # [174]
        )

def create_data_loader(data_dir, batch_size=4, seq_length=5):
    return DataLoader(
        TopologyDataset(data_dir, seq_length),
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True
    )

# 使用示例
if __name__ == "__main__":
    # 初始化拓扑图
    topology_graph = parse_xml_topology(config.XML_TOPOLOGIES_PATH)
    
    # 创建数据加载器
    loader = create_data_loader(
        data_dir=config.DATASET_PATH,
        batch_size=32,
        seq_length=5
    )
    
    # 验证数据加载
    for batch_idx, (inputs, targets) in enumerate(loader):
        print(f"Batch {batch_idx}:")
        print(f"Input shape: {inputs.shape}")  # 应为 [batch_size, seq_len, num_features]
        print(f"Target shape: {targets.shape}") # 应为 [batch_size, num_features]
        
        # if batch_idx > 2:  # 仅查看前几个批次
        #     break