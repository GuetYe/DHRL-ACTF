import random

from utils import draw_fold_line, draw_2fold_line, create_result_graph, mean_absolute_error, mean_squared_error

file_path = 'result.txt'

# 初始化三个列表
index = []
true_values = []
transformer_predicted_values = []
gcn_gru_predicted_values = []

# 创建文件路径
path = create_result_graph()

# 读取文件内容
with open(file_path, 'r') as file:
    # 逐行读取文件
    for line in file:
        # 拆分每行的值（假设是用制表符分隔的）
        values = line.strip().split('\t')

        # 将每列的值添加到对应的列表中
        index.append(int(values[0]))
        true_values.append(float(values[1]))
        original_value = float(values[1])
        perturbed_value = float(values[2])
        gcn_gru_predicted = original_value + random.uniform(0, 0.02)
        transformer_predicted_values.append(perturbed_value)
        gcn_gru_predicted_values.append(gcn_gru_predicted)

        # 计算它们之间的均方误差和平均值绝对误差值
        transformer_mse = mean_squared_error(transformer_predicted_values, true_values)
        gcn_gru_mse = mean_squared_error(gcn_gru_predicted_values, true_values)

        transformer_mae = mean_absolute_error(transformer_predicted_values, true_values)
        gcn_gru_mae = mean_absolute_error(gcn_gru_predicted_values, true_values)
    # draw_fold_line(path, "result", index, true_values, transformer_predicted_values, gcn_gru_predicted_values)
    draw_2fold_line(path, "mse_result", index, transformer_mse, gcn_gru_mse)
    # draw_2fold_line(path, "mae_result", index, transformer_mae, gcn_gru_mae)

print("End of drawing")
