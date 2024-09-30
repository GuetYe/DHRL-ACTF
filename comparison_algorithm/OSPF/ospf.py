import config
from env import MultiDomainEnv
import networkx as nx

env = MultiDomainEnv(config.dataset.graph)  # 加载环境变量


def create_txt_file(_pathList):
    with open("./ospf_path.txt", 'w+', encoding='utf-8') as f:
        for path in _pathList:
            f.write(f"{path}\n")


if __name__ == "__main__":

    pathList = []
    #
    # for index, pkl_path in enumerate(config.dataset.pickle_file_path_yield()):
    #     env.read_pickle_and_modify(pkl_path)
    #     # 使用多目标优化算法找到最优路径
    #     shortest_path = nx.multi_objective_shortest_path(env.graph, source=1, target=29,
    #                                                      weight=('bandwidth', 'delay', 'packet_loss', 'packet_drop'),
    #                                                      criterion=multi_objective_fitness)
    #     pathList.append(shortest_path)
    #
    # create_txt_file(pathList)

    # Define the objectives and their corresponding weights
    objectives = {'free_bw': 0.5, 'delay': 0.5, 'loss': 0.1, 'pkt_drop': 0.1, 'distance': 0.1}

    # Iterate over the pickle files
    for index, pkl_path in enumerate(config.dataset.pickle_file_path_yield()):
        env.read_pickle_and_modify(pkl_path)


        # Define a function to calculate the combined weight of multiple objectives
        def combined_weight(u, v, data):
            weight = 0
            for obj, obj_weight in objectives.items():
                weight += obj_weight * data.get(obj, 0)  # Multiply weight by the value of each objective
            return weight


        # Update edge weights in the graph based on multiple objectives
        for u, v, data in env.graph.edges(data=True):
            data['weight'] = combined_weight(u, v, data)

        # Find the shortest path considering multiple objectives
        shortest_path = nx.dijkstra_path(env.graph, source=1, target=29, weight='loss')
        pathList.append(shortest_path)

    # Create a text file containing the paths
    create_txt_file(pathList)
