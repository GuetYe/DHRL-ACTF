import config
from env import MultiDomainEnv
import networkx as nx

env = MultiDomainEnv(config.dataset.graph)  # 加载环境变量


def create_txt_file(_pathList):
    with open("bgp_path.txt", 'w+', encoding='utf-8') as f:
        for path in _pathList:
            f.write(f"{path}\n")


def heuristic(u, v):
    return 0


if __name__ == "__main__":
    # 采用迪杰斯特拉计算最短路径
    pathList = []
    for index, pkl_path in enumerate(config.dataset.pickle_file_path_yield()):
        env.read_pickle_and_modify(pkl_path)
        shortest_path = nx.astar_path(env.graph, source=1, target=29, heuristic=heuristic,weight='free_bw')
        pathList.append(shortest_path)
        # print("最短路径:", shortest_path)
    create_txt_file(pathList)
