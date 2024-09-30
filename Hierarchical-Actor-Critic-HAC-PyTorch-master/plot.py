import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import networkx as nx
import numpy as np
from matplotlib.patches import ConnectionPatch

# 设置中文字体
plt.rcParams['font.sans-serif'] = [u'simHei']  # 显示中文
plt.rcParams['axes.unicode_minus'] = False  # 解决负号问题


def plot_roadmap(edges):
    # 创建一个简单的图
    G = nx.Graph()
    G.add_edges_from(edges)

    # 故障链路为9-10 22-25 两条
    # 无故障链路的路径 [1, 9, 10, 16, 17, 22, 25, 28, 29]
    # 故障链路恢复后的路径[1, 2, 8, 11, 16, 17, 21, 25, 28, 29]
    path_nodes1 = [1, 9, 10, 16, 17, 22, 25, 28, 29]  # 节点
    path_edges1 = [(path_nodes1[i], path_nodes1[i + 1]) for i in range(len(path_nodes1) - 1)]  # 边

    path_nodes2 = [1, 2, 8, 11, 16, 17, 21, 25, 28, 29]  # 节点
    path_edges2 = [(path_nodes2[i], path_nodes2[i + 1]) for i in range(len(path_nodes2) - 1)]  # 边

    # 定义节点的颜色
    node_colors = ['blue' for _ in G.nodes()]  # 默认所有节点为蓝色
    node_list = list(G.nodes())
    for node in path_nodes1[1:-1]:
        if node in path_nodes2[1:-1]:
            node_colors[node_list.index(node)] = 'purple'  # 共同点为紫色
        else:
            node_colors[node_list.index(node)] = 'green'  # 路径1的节点为绿色
    for node in path_nodes2[1:-1]:
        if node in path_nodes1[1:-1]:
            node_colors[node_list.index(node)] = 'purple'  # 共同点为紫色
        else:
            node_colors[node_list.index(node)] = 'red'  # 路径2的节点为红色

    # 源节点和目的节点颜色
    node_colors[0] = 'yellow'  # 源节点设置为黄色
    node_colors[-1] = 'orange'  # 目的节点设置为橙色

    # 定义边的颜色
    edge_colors = ['black' for _ in G.edges()]  # 默认所有边为黑色
    edge_list = list(G.edges())  # 将边的视图转换为列表
    for edge in path_edges1:
        if edge in path_edges2:
            edge_colors[edge_list.index(edge)] = 'purple'  # 公共边为紫色
        else:
            edge_colors[edge_list.index(edge)] = 'green'  # 路径1的边为绿色
    for edge in path_edges2:
        if edge in path_edges1:
            edge_colors[edge_list.index(edge)] = 'purple'  # 公共边为紫色
        else:
            edge_colors[edge_list.index(edge)] = 'red'  # 路径2的边为红色

    # 绘制图形
    pos = nx.spring_layout(G)  # 计算节点位置
    nx.draw(G, pos, with_labels=True, node_color=node_colors, edge_color=edge_colors)

    # 显示图形
    plt.show()


def draw_episode_data(picture_path, x, y, x_label: str, y_label: str, file_name):
    """
    画一个图
    picture_path:存放图片的路径
    x:x坐标的数据
    y:y坐标的数据
    x_label:x的标签
    y_label:y的标签
    file_name:文件名字
    """
    # Set figure size and style
    plt.figure(figsize=(8, 6))
    plt.style.use('ggplot')

    # Set plot title, axis labels, and tick labels
    plt.xlabel(x_label, fontsize=14, fontweight='bold')
    plt.ylabel(y_label, fontsize=14, fontweight='bold')
    plt.xticks(fontsize=12, fontweight='bold')
    plt.yticks(fontsize=12, fontweight='bold')

    # Plot the data with custom colors and line styles
    plt.plot(x, y, color='#0072B2', linestyle='-', linewidth=2)

    # Add grid lines and legend
    plt.grid(True, linestyle='--', alpha=0.25, color='gray', linewidth=1)
    plt.legend(loc='best', fontsize=12, frameon=False)

    # Add padding and show the plot
    plt.tight_layout()
    plt.savefig("{}/{}.pdf".format(picture_path, file_name))
    plt.show()


def plot_compare_data(title_name, data_path, x, y1, y2, y3, y4, x_label, y_label, file_name):
    """
    画两条曲线的图
    title_name:图的标题
    picture_path:存放图片的路径
    x:x坐标的数据
    y:y坐标的数据
    x_label:x的标签
    y_label:y的标签
    file_name:文件名字
    """

    # plt.title(title_name)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.plot(x, y1, color='#0072B2', linestyle='-', label="PPONSA")  # "#0848ae"
    plt.plot(x, y2, color='#009E73', linestyle='-.', label="Dueling DQN")  # linestyle='--'
    # plt.plot(x, y3, color='#CC79A7', linestyle='-.',label="ξ1:ξ2=0.5:1.5")
    # plt.plot(x, y4, color='#D55E00', linestyle='-',label="ξ1:ξ2=0.5:0.8")  # "#e8710a"
    # 在两条曲线之间做一个填充
    # plt.fill_between(x, y1, y2)
    # 显示图例
    plt.legend()  # 默认loc = Best
    plt.grid(True, linestyle='--', alpha=0.25)
    plt.savefig("{}/{}.pdf".format(data_path, file_name), dpi=300, bbox_inches='tight', pad_inches=0)
    plt.savefig("{}/{}.jpeg".format(data_path, file_name), dpi=300, bbox_inches='tight', pad_inches=0)
    plt.show()


def autolabel(rects):
    """
    定义函数来显示柱子上的数值
    """
    for rect in rects:
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width() / 2. - 0.08, 1.02 * height, '%.2f' % height, size=4,
                 family="Times new roman")  # rotation=270 数值旋转270度


def plot_traffic_graph(data_path, file_name, x, y1, y2, y3, y4, y5, y6, x_label, y_label, label_name):
    """
    画出与PPO、DQN、Q-learning、BGP、OSPF的流量对比图，柱状图
    """
    # 设置xy标签的值
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    autolabel(plt.bar(x, y1, width=0.1, color="#0848ae", label="DHRL-ACTF$_{%s}$" % label_name))  # "#0848ae"
    autolabel(plt.bar([i + 0.15 for i in x], y2, width=0.1, color="#00C0C0", label="DRL-PPONSA$_{%s}$" % label_name))
    autolabel(plt.bar([i + 0.3 for i in x], y3, width=0.1, color="#df208c", label="DRL-DQN$_{%s}$" % label_name))
    autolabel(plt.bar([i + 0.45 for i in x], y4, width=0.1, color="#e8710a",
                      label="Q_learning$_{%s}$" % label_name))  # "#e8710a"
    autolabel(plt.bar([i + 0.6 for i in x], y5, width=0.1, color="#E1F190",
                      label="BGP$_{%s}$" % label_name))  # "#e8710a"
    autolabel(plt.bar([i + 0.75 for i in x], y6, width=0.1, color="m",
                      label="OSPF$_{%s}$" % label_name))  # "#e8710a"
    # 显示图例和网格
    plt.legend(fontsize=10)  # 默认loc = Best # ncol=4让数据标签横向排列
    plt.grid(True, linestyle='--', alpha=0.5)
    # 修改x刻度名字
    plt.xticks([i + 0.3 for i in x], ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11'])
    # plt.savefig("{}/{}.pdf".format(data_path, file_name), dpi=300, bbox_inches='tight', pad_inches=0)
    plt.savefig("{}/{}.jpeg".format(data_path, file_name), dpi=300, bbox_inches='tight', pad_inches=0)
    plt.show()


def draw_fold_line(path, xLable, yLable, x_data, parameter1_data, parameter2_data, parameter3_data, parameter4_data,
                   parameter5_data,
                   parameter6_data):
    # 假设有两个参数的数据，分别存储在 parameter1_data 和 parameter2_data 中
    # 假设还有 x 轴上的数据，存储在 x_data 中

    # 绘制折线图
    plt.plot(x_data, parameter1_data, color='blue', linestyle='-', label='DHRL-ACTF')
    plt.plot(x_data, parameter2_data, color='red', linestyle='-', label='DRL-PPONSA')
    plt.plot(x_data, parameter3_data, color='green', linestyle='-', label='DRL-DQN')
    plt.plot(x_data, parameter4_data, color='orange', linestyle='-', label='Q_learning')
    plt.plot(x_data, parameter5_data, color='c', linestyle='-', label='OSPF')
    plt.plot(x_data, parameter6_data, color='m', linestyle='-', label='BGP')

    # 绘制数据点
    scatter1 = plt.scatter(x_data, parameter1_data, color='blue', marker='*')
    scatter2 = plt.scatter(x_data, parameter2_data, color='red', marker='+')
    scatter3 = plt.scatter(x_data, parameter3_data, color='green', marker='^')
    scatter1 = plt.scatter(x_data, parameter4_data, color='orange', marker='o')
    scatter2 = plt.scatter(x_data, parameter5_data, color='c', marker='d')
    scatter3 = plt.scatter(x_data, parameter6_data, color='m', marker='p')

    # 添加标签和标题
    plt.xlabel('{}'.format(xLable))
    plt.ylabel('{}'.format(yLable))
    # plt.title('DRL-PPO与Dijkstra的丢包率对比曲线')

    for i in range(len(x_data)):
        plt.text(x_data[i], parameter1_data[i], '{:.2f}'.format(parameter1_data[i]), fontsize=8, color='blue',
                 ha='center', va='bottom')
        plt.text(x_data[i], parameter2_data[i], '{:.2f}'.format(parameter2_data[i]), fontsize=8, color='red',
                 ha='center', va='bottom')
        plt.text(x_data[i], parameter3_data[i], '{:.2f}'.format(parameter3_data[i]), fontsize=8, color='green',
                 ha='center', va='bottom')
        plt.text(x_data[i], parameter4_data[i], '{:.2f}'.format(parameter4_data[i]), fontsize=8, color='orange',
                 ha='center', va='bottom')
        plt.text(x_data[i], parameter5_data[i], '{:.2f}'.format(parameter5_data[i]), fontsize=8, color='cyan',
                 ha='center', va='bottom')
        plt.text(x_data[i], parameter6_data[i], '{:.2f}'.format(parameter6_data[i]), fontsize=8, color='magenta',
                 ha='center', va='bottom')
    # 创建自定义图例标记
    custom_legend = [
        mlines.Line2D([], [], color='blue', marker='*', linestyle='None', label='DHRL-ACTF'),
        mlines.Line2D([], [], color='red', marker='+', linestyle='None', label='DRL-PPONSA'),
        mlines.Line2D([], [], color='green', marker='^', linestyle='None', label='DRL-DQN'),
        mlines.Line2D([], [], color='orange', marker='o', linestyle='None', label='Q_learning'),
        mlines.Line2D([], [], color='c', marker='d', linestyle='None', label='OSPF'),
        mlines.Line2D([], [], color='m', marker='p', linestyle='None', label='BGP')
    ]

    # 添加图例，并设置图例标记样式
    plt.legend(handles=custom_legend)
    plt.grid(True, linestyle='--', alpha=0.25)
    plt.savefig("{}/{}.jpeg".format(path, yLable), dpi=300, bbox_inches='tight', pad_inches=0.2)

    # 显示图形
    plt.show()


def zone_and_linked(ax, axins, zone_left, zone_right, x, y, linked='bottom',
                    x_ratio=0.05, y_ratio=0.05):
    """缩放内嵌图形，并且进行连线
    ax:         调用plt.subplots返回的画布。例如： fig,ax = plt.subplots(1,1)
    axins:      内嵌图的画布。 例如 axins = ax.inset_axes((0.4,0.1,0.4,0.3))
    zone_left:  要放大区域的横坐标左端点
    zone_right: 要放大区域的横坐标右端点
    x:          X轴标签
    y:          列表，所有y值
    linked:     进行连线的位置，{'bottom','top','left','right'}
    x_ratio:    X轴缩放比例
    y_ratio:    Y轴缩放比例
    """
    xlim_left = x[zone_left] - (x[zone_right] - x[zone_left]) * x_ratio
    xlim_right = x[zone_right] + (x[zone_right] - x[zone_left]) * x_ratio

    y_data = np.hstack([yi[zone_left:zone_right] for yi in y])
    ylim_bottom = np.min(y_data) - (np.max(y_data) - np.min(y_data)) * y_ratio
    ylim_top = np.max(y_data) + (np.max(y_data) - np.min(y_data)) * y_ratio

    axins.set_xlim(xlim_left, xlim_right)
    axins.set_ylim(ylim_bottom, ylim_top)

    ax.plot([xlim_left, xlim_right, xlim_right, xlim_left, xlim_left],
            [ylim_bottom, ylim_bottom, ylim_top, ylim_top, ylim_bottom], "black")

    if linked == 'bottom':
        xyA_1, xyB_1 = (xlim_left, ylim_top), (xlim_left, ylim_bottom)
        xyA_2, xyB_2 = (xlim_right, ylim_top), (xlim_right, ylim_bottom)
    elif linked == 'top':
        xyA_1, xyB_1 = (xlim_left, ylim_bottom), (xlim_left, ylim_top)
        xyA_2, xyB_2 = (xlim_right, ylim_bottom), (xlim_right, ylim_top)
    elif linked == 'left':
        xyA_1, xyB_1 = (xlim_right, ylim_top), (xlim_left, ylim_top)
        xyA_2, xyB_2 = (xlim_right, ylim_bottom), (xlim_left, ylim_bottom)
    elif linked == 'right':
        xyA_1, xyB_1 = (xlim_left, ylim_top), (xlim_right, ylim_top)
        xyA_2, xyB_2 = (xlim_left, ylim_bottom), (xlim_right, ylim_bottom)

    con = ConnectionPatch(xyA=xyA_1, xyB=xyB_1, coordsA="data",
                          coordsB="data", axesA=axins, axesB=ax)
    axins.add_artist(con)
    con = ConnectionPatch(xyA=xyA_2, xyB=xyB_2, coordsA="data",
                          coordsB="data", axesA=axins, axesB=ax)
    axins.add_artist(con)
def smooth(data_array, weight=0.9):
    # 一个类似 tensorboard smooth 功能的平滑滤波
    # https://dingguanglei.com/tensorboard-xia-smoothgong-neng-tan-jiu/
    last = data_array[0]
    smoothed = []
    for new in data_array:
        smoothed_val = last * weight + (1 - weight) * new
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed

if __name__ == "__main__":
    # x = np.array([[3., -1., 2.],
    #               [2., 0., 0.],
    #               [0., 1., -1.], ])
    #
    # agent_node = 1
    # adj_node = [3, 4, 5, 11]
    # link1 = (1, 6)
    # link2 = [(1, 3), (1, 4), (1, 5), (1, 11)]
    # x_data = ['0:00', '3:00', '6:00', '9:00', '12:00', '15:00', '18:00', '21:00']
    # x1 = [0.02, 0.01, 0.01, 0.01, 0.02, 0.01, 0.01, 0.03]
    # x2 = [0.02, 0.14, 0.25, 0.25, 0.36, 0.68, 0.14, 0.29]
    # draw_fold_line(x_data, x1, x2)

    # a = compare_link(link1, link2)
    # print(a)

    # x坐标
    x = np.arange(1, 1001)

    # 生成y轴数据，并添加随机波动
    y1 = np.log(x)
    indexs = np.random.randint(0, 1000, 800)
    for index in indexs:
        y1[index] += np.random.rand() - 0.5

    y2 = np.log(x)
    indexs = np.random.randint(0, 1000, 800)
    for index in indexs:
        y2[index] += np.random.rand() - 0.5

    y3 = np.log(x)
    indexs = np.random.randint(0, 1000, 800)
    for index in indexs:
        y3[index] += np.random.rand() - 0.5

    # 绘制主图
    fig, ax = plt.subplots(1, 1, figsize=(12, 7))
    ax.plot(x, y1, color='#f0bc94', label='trick-1', alpha=0.7)
    ax.plot(x, y2, color='#7fe2b3', label='trick-2', alpha=0.7)
    ax.plot(x, y3, color='#cba0e6', label='trick-3', alpha=0.7)
    ax.legend(loc='right')

    # plt.show()

    # 绘制缩放图
    axins = ax.inset_axes((0.4, 0.1, 0.4, 0.3))

    # 在缩放图中也绘制主图所有内容，然后根据限制横纵坐标来达成局部显示的目的
    axins.plot(x, y1, color='#f0bc94', label='trick-1', alpha=0.7)
    axins.plot(x, y2, color='#7fe2b3', label='trick-2', alpha=0.7)
    axins.plot(x, y3, color='#cba0e6', label='trick-3', alpha=0.7)

    # 局部显示并且进行连线
    zone_and_linked(ax, axins, 100, 150, x, [y1, y2, y3], 'right')

    plt.show()
