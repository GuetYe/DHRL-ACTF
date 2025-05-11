import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch
import numpy as np
from pathlib import Path
import networkx as nx
import time
import os
import torch
import matplotlib.lines as mlines

# 设置中文字体
plt.rcParams['font.sans-serif'] = [u'simHei']  # 显示中文
plt.rcParams['axes.unicode_minus'] = False  # 解决负号问题


def normalize_value(tensor_data):
    """
    对数据进行归一化处理
    """
    # Calculate min and max values
    min_vals = torch.min(tensor_data)
    max_vals = torch.max(tensor_data)

    # Perform min-max normalization
    normalized_data = (tensor_data - min_vals) / (max_vals - min_vals) + 1e-6
    return normalized_data  # 升一个维度


def denormalization_value(normalized_data, max_val, min_val):
    """
    normalized_data: 归一化后的数据
    max_val:最大值
    min_val:最小值
    restored_data:恢复后的数据
    """

    # Restore original values using min-max normalization
    restored_data = (normalized_data * (max_val - min_val)) + min_val
    return restored_data


def mean_absolute_error(predictions, targets):
    """
    平均值绝对误差
    # 示例用法
    predictions = model(inputs, inputs)
    mae = mean_absolute_error(predictions, targets)
    print(f'MAE: {mae}')
    """
    arg_mae = []
    _mae = []
    for i in range(len(predictions)):
        absolute_errors = np.abs(predictions[i] - targets[i])
        mae = np.mean(absolute_errors)
        arg_mae.append(mae)
        _mae.append(sum(arg_mae) / len(arg_mae))
    return _mae


def mean_squared_error(predictions, targets):
    """
    均方误差
    # 示例用法
    predictions = model(inputs, inputs)
    mse = mean_squared_error(predictions, targets)
    print(f'MSE: {mse}')
    """
    arg_mse = []
    _mse = []
    for i in range(len(predictions)):
        squared_errors = (predictions[i] - targets[i]) ** 2
        mse = np.mean(squared_errors)
        arg_mse.append(mse)
        _mse.append(sum(arg_mse) / len(arg_mse))
    return _mse


def read_pickle_and_modify(pkl_graph_path):
    """
        读取图graph的pickle文件
        初始化所有参数
    :param pkl_graph_path: pickle文件路径
    :return: 图 nx.graph
    """
    pkl_graph = nx.read_gpickle(pkl_graph_path)  # 将pickle中的图读取出来
    return pkl_graph


def create_graph_path():
    """
    在文件夹下创建文件
    """
    local_time = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
    WORK_DIR = Path.cwd().parent
    loss_graph_path = WORK_DIR / "TRANSFORMER/loss_graph/loss_{}".format(local_time)
    accuracy_graph_path = WORK_DIR / "TRANSFORMER/performance_graph/ace_{}".format(local_time)

    if not os.path.exists(loss_graph_path):
        os.mkdir(loss_graph_path)
    if not os.path.exists(accuracy_graph_path):
        os.mkdir(accuracy_graph_path)
    return loss_graph_path, accuracy_graph_path


def create_result_graph():
    local_time = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
    WORK_DIR = Path.cwd().parent
    result_graph_path = WORK_DIR / "TRANSFORMER/result/result_{}".format(local_time)
    if not os.path.exists(result_graph_path):
        os.mkdir(result_graph_path)
    return result_graph_path


def plot_episode_data(picture_path, x, y, x_label: str, y_label: str, file_name):
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
    # plt.legend(loc='best', fontsize=12, frameon=False)

    # Add padding and show the plot
    plt.tight_layout()
    plt.savefig("{}/{}.pdf".format(picture_path, file_name))
    plt.show()


def draw_fold_line(picture_path, file_name, x_data, param1_data, param2_data, param3_data):
    """
    # 假设有两个参数的数据，分别存储在 parameter1_data 和 parameter2_data、parameter3_data  中
    # 假设还有 x 轴上的数据，存储在 x_data 中
    """
    confidence_interval = 0.01  # 置信水平
    upper_bound = []
    lower_bound = []
    # 计算置信区间的上限和下限
    for i in range(len(param2_data)):
        upper_bound.append(param2_data[i] + confidence_interval)
        lower_bound.append(param2_data[i] - confidence_interval)
    # 绘制折线图
    line1, = plt.plot(x_data, param1_data, linestyle='-', linewidth=1, color='blue', label='Real Value')
    line2, = plt.plot(x_data, param2_data, linestyle='-', linewidth=1, color='red',
                      label='Predicted Value(Transformer Model)')
    line3, = plt.plot(x_data, param3_data, linestyle='-', linewidth=1, color='green',
                      label='Predicted Value(GCN_GRU Model)')
    plt.fill_between(x_data, lower_bound, upper_bound, color='c', alpha=0.3, label='90% Confidence Interval')

    # 添加标签和标题
    plt.xlabel('时间')
    plt.ylabel('网络链路流量(Mbps)')

    # 添加图例，并设置图例标记样式
    plt.legend()

    # 添加网格线
    plt.grid(True)

    # 保存信息
    plt.savefig("{}/{}.png".format(picture_path, file_name), dpi=300, bbox_inches='tight', pad_inches=0.2)
    # 显示图形
    plt.show()


def draw_2fold_line(picture_path, file_name, x_data, param1_data, param2_data):
    """
    # 假设有两个参数的数据，分别存储在 parameter1_data 和 parameter2_data、parameter3_data  中
    # 假设还有 x 轴上的数据，存储在 x_data 中
    """

    # 绘制折线图
    line1, = plt.plot(x_data, param1_data, linestyle='-', color='c', label='均方误差值(Transformer模型)')
    line2, = plt.plot(x_data, param2_data, linestyle='-', color='m', label='均方误差值(GCN_GRU模型)')

    # 添加标签和标题
    plt.xlabel('时间')
    plt.ylabel('均方误差值(%)')
    # plt.title('网络流量预测曲线')

    # 添加图例，并设置图例标记样式
    plt.legend()

    # 添加网格线
    plt.grid(True)

    # 保存信息
    plt.savefig("{}/{}.png".format(picture_path, file_name), dpi=300, bbox_inches='tight', pad_inches=0.2)
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


if __name__ == "__main__":
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
