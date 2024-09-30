import matplotlib.pyplot as plt


def draw_episode_data(x, y, x_label: str, y_label: str):
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
    plt.show()
