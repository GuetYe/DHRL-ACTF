from pathlib import Path
from dataSet import DataSet

d_model = 8  # 模型中的隐藏状态将是8维向量
nhead = 8  # 该模型在多头注意力机制中使用了8个头。
num_layers = 8  # 变压器模型由16个相同的层组成。

# 数据集地址以及topo信息的地址
WORK_DIR = Path.cwd().parent  # 获取当前路径

XML_TOPOLOGIES_PATH = WORK_DIR / 'TRANSFORMER/topologies/topology_node_30.xml'  # 链路拓扑路径
DATASET_PATH = WORK_DIR / 'TRANSFORMER/dataSet/weight_pickle'  # 数据集的路径

# -----------------从数据集中加载一些参数----------------------#
dataset = DataSet(XML_TOPOLOGIES_PATH, DATASET_PATH)
dataset.set_init_topology()
