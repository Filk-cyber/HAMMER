import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# 创建输出文件夹
output_dir = 'output_figures'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 定义不同方法的标记样式
markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
          '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']


def process_excel_table1(file_path):
    """
    处理第一个Excel表格,生成12张折线图
    """
    # 读取所有sheet
    excel_file = pd.ExcelFile(file_path)

    for sheet_name in excel_file.sheet_names:
        print(f"Processing Table 1 - Sheet: {sheet_name}")
        df = pd.read_excel(file_path, sheet_name=sheet_name)

        # 获取第5列和第10列的列名(实验设置名称)
        setting1_name = df.columns[4]  # 第5列(索引4)
        setting2_name = df.columns[9]  # 第10列(索引9)

        # 处理第一个实验设置(第1-4列)
        data1 = df.iloc[:, 0:4].copy()
        data1.columns = ['fakenum', 'Method', 'EM', 'F1_score']
        data1 = data1.dropna()

        # 处理第二个实验设置(第6-9列)
        data2 = df.iloc[:, 5:9].copy()
        data2.columns = ['fakenum', 'Method', 'EM', 'F1_score']
        data2 = data2.dropna()

        # 为第一个实验设置生成2张图(EM和F1 score)
        # 第一个表：每行最多2个，列间距30
        plot_line_chart(data1, sheet_name, 'EM', setting1_name, output_dir,
                       max_ncol=2, col_spacing=30)
        plot_line_chart(data1, sheet_name, 'F1_score', setting1_name, output_dir,
                       max_ncol=2, col_spacing=30)

        # 为第二个实验设置生成2张图(EM和F1 score)
        plot_line_chart(data2, sheet_name, 'EM', setting2_name, output_dir,
                       max_ncol=2, col_spacing=30)
        plot_line_chart(data2, sheet_name, 'F1_score', setting2_name, output_dir,
                       max_ncol=2, col_spacing=30)


def process_excel_table2(file_path):
    """
    处理第二个Excel表格,生成6张折线图
    """
    # 读取所有sheet
    excel_file = pd.ExcelFile(file_path)

    for sheet_name in excel_file.sheet_names:
        print(f"Processing Table 2 - Sheet: {sheet_name}")
        df = pd.read_excel(file_path, sheet_name=sheet_name)

        # 获取第5列的列名(实验设置名称)
        setting_name = df.columns[4]  # 第5列(索引4)

        # 处理前4列数据
        data = df.iloc[:, 0:4].copy()
        data.columns = ['fakenum', 'Method', 'Accuracy', 'F1_score']
        data = data.dropna()

        # 生成2张图(Accuracy和F1 score)
        # 第二个表：每行最多3个，列间距可以自定义（这里设为5）
        plot_line_chart(data, sheet_name, 'Accuracy', setting_name, output_dir,
                       suffix='_add_cag', max_ncol=3, col_spacing=10)
        plot_line_chart(data, sheet_name, 'F1_score', setting_name, output_dir,
                       suffix='_add_cag', max_ncol=3, col_spacing=10)


def plot_line_chart(data, dataset_name, metric, setting_name, output_dir,
                   suffix='', max_ncol=2, col_spacing=30):
    """
    绘制折线图

    参数:
    - data: 包含fakenum, Method和指标的DataFrame
    - dataset_name: 数据集名称(sheet名称)
    - metric: 指标名称(EM, F1_score, Accuracy)
    - setting_name: 实验设置名称
    - output_dir: 输出目录
    - suffix: 文件名后缀(用于第二个表格)
    - max_ncol: 每行最多显示的图例数量（默认2）
    - col_spacing: 图例列间距（默认30）
    """
    # 获取所有唯一的方法
    methods = data['Method'].unique()
    num_methods = len(methods)
    ncol = min(max_ncol, num_methods)  # 使用传入的max_ncol参数

    # 根据方法数量动态调整图片高度，为图例留出空间
    fig_height = 6 + 0.3 * ((num_methods - 1) // ncol + 1)  # 动态增加高度

    fig, ax = plt.subplots(figsize=(10, fig_height))

    # 为每个方法绘制折线
    for idx, method in enumerate(methods):
        method_data = data[data['Method'] == method].sort_values('fakenum')

        ax.plot(method_data['fakenum'],
                method_data[metric],
                marker=markers[idx % len(markers)],
                color=colors[idx % len(colors)],
                linewidth=2,
                markersize=8,
                label=method)

    # 设置标题和标签(全英文)，横纵坐标轴标签加粗
    metric_display = metric.replace('_', ' ')  # 将下划线替换为空格用于显示
    ax.set_title(f'{metric_display} on {dataset_name}', fontsize=14, fontweight='bold')
    ax.set_xlabel('Misinformation document count', fontsize=12, fontweight='bold')  # 加粗
    ax.set_ylabel(metric_display, fontsize=12, fontweight='bold')  # 加粗

    # 获取数据的最小值和最大值
    x_min = data['fakenum'].min()
    x_max = data['fakenum'].max()
    y_min = data[metric].min()
    y_max = data[metric].max()

    # 计算数据范围
    x_range = x_max - x_min
    y_range = y_max - y_min

    # 添加小边距以确保边界点完整显示
    x_margin = x_range * 0.02 if x_range > 0 else 1  # 横向留2%边距
    y_margin = y_range * 0.03 if y_range > 0 else 0.01  # 纵向留3%边距

    # 设置坐标轴范围
    ax.set_xlim(x_min - x_margin, x_max + x_margin)
    ax.set_ylim(y_min - y_margin, y_max + y_margin)

    # 设置纵坐标刻度，使用0.05的间隔
    y_tick_start = np.floor(y_min / 0.05) * 0.05
    y_tick_end = np.ceil(y_max / 0.05) * 0.05
    y_ticks = np.arange(y_tick_start, y_tick_end + 0.001, 0.05)
    ax.set_yticks(y_ticks)

    # 设置横坐标刻度
    x_unique = sorted(data['fakenum'].unique())
    if len(x_unique) <= 20:
        ax.set_xticks(x_unique)

    # 设置刻度标签字体加粗
    ax.tick_params(axis='both', which='major', labelsize=10)  # 设置刻度标签大小

    # 将横坐标和纵坐标的刻度数字加粗
    for label in ax.get_xticklabels():
        label.set_fontweight('bold')
    for label in ax.get_yticklabels():
        label.set_fontweight('bold')

    # 设置网格
    ax.grid(True, alpha=0.3, linestyle='--')

    # 设置图例：放在横坐标轴下方，不显示边框，图例文字加粗
    # 使用传入的列间距参数
    legend = ax.legend(loc='upper center',  # 图例位置
                       bbox_to_anchor=(0.5, -0.12),  # 图例在坐标轴下方
                       ncol=ncol,  # 每行显示的列数（使用传入的参数）
                       frameon=False,  # 不显示边框
                       fontsize=10,  # 字体大小
                       columnspacing=col_spacing,  # 列间距（使用传入的参数）
                       handletextpad=0.5,  # 图例标记和文本之间的间距
                       borderaxespad=0,  # 图例与坐标轴的间距
                       prop={'weight': 'bold'})  # 图例文字加粗

    # 调整子图位置，为图例留出空间
    plt.subplots_adjust(bottom=0.15)

    # 保存图片 - 关键：使用 bbox_extra_artists 参数包含图例
    filename = f'{metric}_{dataset_name}_{setting_name}{suffix}.png'
    filename = filename.replace('/', '_').replace('\\', '_').replace(':', '_')
    filepath = os.path.join(output_dir, filename)

    # 这是关键：bbox_extra_artists 告诉 savefig 要包含图例
    plt.savefig(filepath, dpi=300, bbox_inches='tight', bbox_extra_artists=[legend])
    print(f"Saved: {filepath}")

    plt.close()


# 主程序
if __name__ == "__main__":
    # 指定你的Excel文件路径
    table1_path = '/home/jiangjp/trace-idea/data/cram_ourmethod_fakenum(Mistral-7B).xlsx'
    table2_path = '/home/jiangjp/trace-idea/data/cag_cram_ourmethod_fakenum(Mistral-7B).xlsx'

    print("=" * 50)
    print("开始处理第一个Excel表格...")
    print("=" * 50)
    try:
        process_excel_table1(table1_path)
        print("\n第一个表格处理完成！生成了12张折线图。")
    except Exception as e:
        print(f"处理第一个表格时出错: {e}")

    print("\n" + "=" * 50)
    print("开始处理第二个Excel表格...")
    print("=" * 50)
    try:
        process_excel_table2(table2_path)
        print("\n第二个表格处理完成！生成了6张折线图。")
    except Exception as e:
        print(f"处理第二个表格时出错: {e}")

    print("\n" + "=" * 50)
    print(f"所有图片已保存到 '{output_dir}' 文件夹中")
    print("=" * 50)