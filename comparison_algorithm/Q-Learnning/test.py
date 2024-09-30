# 假设您有一个包含数据的列表 data
import numpy as np
import matplotlib.pyplot as plt

# 生成示例数据
x_data = np.linspace(0, 10, 100)
y_true = np.sin(x_data)
y_pred = np.sin(x_data) + np.random.normal(0, 0.1, size=len(x_data))  # 模拟预测值
confidence_interval = 0.1  # 置信水平

# 计算置信区间的上限和下限
upper_bound = y_pred + confidence_interval
lower_bound = y_pred - confidence_interval

# 绘制曲线和置信区间
plt.plot(x_data, y_true, color='blue', label='True')
plt.plot(x_data, y_pred, color='red', label='Predicted')
plt.fill_between(x_data, lower_bound, upper_bound, color='gray', alpha=0.3, label='Confidence Interval')

plt.xlabel('X')
plt.ylabel('Y')
plt.title('Prediction with Confidence Interval')
plt.legend()
plt.grid(True)
plt.show()
