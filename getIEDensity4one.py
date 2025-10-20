import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from scipy.stats import gaussian_kde

# 设置英文字体和学术论文样式
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 14


def plot_ie_density_distribution(excel_file_path):
    """
    为学术论文绘制IE值的概率密度分布图
    """

    # 读取Excel文件中的TotalDatasets表
    df = pd.read_excel(excel_file_path, sheet_name='TotalDatasets')

    print(f"总数据点: {len(df)} 个IE值")
    print(f"IE均值: {df['IE'].mean():.6f}")
    print(f"IE标准差: {df['IE'].std():.6f}")
    print(f"正效应比例: {(df['IE'] > 0).sum() / len(df):.2%}")

    # 创建高质量的单一图表
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    # 生成高分辨率的平滑密度曲线
    ie_values = df['IE'].values

    # 使用核密度估计创建更平滑的曲线
    kde = gaussian_kde(ie_values)
    x_range = np.linspace(ie_values.min() - 0.001, ie_values.max() + 0.001, 1000)
    density = kde(x_range)

    # 绘制主要的密度曲线，使用优雅的样式
    ax.plot(x_range, density, linewidth=3, color='steelblue', alpha=0.9,
            label='Probability Density')

    # 在曲线下方填充，增强视觉效果
    ax.fill_between(x_range, density, alpha=0.3, color='steelblue')

    # 添加专业样式的参考线
    ax.axvline(x=0, color='red', linestyle='--', alpha=0.8,
               linewidth=2, label='No Effect Line (IE=0)')
    ax.axvline(x=df['IE'].mean(), color='darkgreen', linestyle='-',
               alpha=0.9, linewidth=2,
               label=f'Mean = {df["IE"].mean():.6f}')

    # 在左上角创建统计信息文本框
    stats_text = f'Statistics Summary:\n'
    stats_text += f'Sample Size: {len(df):,}\n'
    stats_text += f'Mean: {df["IE"].mean():.6f}\n'
    stats_text += f'Std Dev: {df["IE"].std():.6f}\n'
    stats_text += f'Median: {df["IE"].median():.6f}\n'
    stats_text += f'Skewness: {df["IE"].skew():.4f}\n'
    stats_text += f'Kurtosis: {df["IE"].kurtosis():.4f}\n'
    stats_text += f'Positive Effects: {(df["IE"] > 0).sum() / len(df):.1%}'

    # 将统计信息放在左上角
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            verticalalignment='top', horizontalalignment='left',
            bbox=dict(boxstyle='round,pad=0.6', facecolor='lightblue',
                      alpha=0.9, edgecolor='navy'), fontsize=11, fontweight='normal')

    # 设置专业的标题和标签
    ax.set_title('HSARM Framework Correction Effect Distribution Analysis',
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Improvement Effect (IE)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Probability Density', fontsize=14, fontweight='bold')

    # 设置简化的图例，只显示三个主要元素，放在右上角
    legend = ax.legend(loc='upper right', fontsize=12, frameon=True,
                       fancybox=True, shadow=True, framealpha=0.9)
    legend.get_frame().set_facecolor('white')

    # 添加专业网格
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax.set_axisbelow(True)

    # 设置坐标轴范围，留出适当边距以便更好地可视化
    x_margin = (ie_values.max() - ie_values.min()) * 0.05
    ax.set_xlim(ie_values.min() - x_margin, ie_values.max() + x_margin)

    # 改善刻度格式
    ax.tick_params(axis='both', which='major', labelsize=12)

    # 添加淡雅的背景色
    ax.set_facecolor('#fafafa')

    # 确保紧凑布局
    plt.tight_layout()

    # 保存高质量图片
    output_path = '/home/jiangjp/trace-idea/results/myidea/IE_Density_Distribution_Academic.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white',
                edgecolor='none', format='png')
    print(f"\n图片已保存至: {output_path}")

    # 同时保存PDF版本用于发表
    pdf_path = output_path.replace('.png', '.pdf')
    plt.savefig(pdf_path, bbox_inches='tight', facecolor='white',
                edgecolor='none', format='pdf')
    print(f"PDF版本已保存至: {pdf_path}")

    plt.show()

    # 进行全面的统计检验
    print(f"\n=== 统计分析结果 ===")

    # 1. 基本统计
    mean_ie = df['IE'].mean()
    print(f"IE均值: {mean_ie:.4f}")

    # 2. 单侧t检验（检验均值是否显著大于0）
    t_stat, p_value_one_sided = stats.ttest_1samp(df['IE'], 0, alternative='greater')
    print(f"单侧t检验(μ > 0): p = {p_value_one_sided:.2e}")

    # 3. 正值比例
    positive_ratio = (df['IE'] > 0).mean()
    print(f"正值比例: {positive_ratio:.2%}")

    # 4. 显著性判断
    if p_value_one_sided < 0.001:
        print("*** 正向修正效应高度显著")
    elif p_value_one_sided < 0.01:
        print("** 正向修正效应非常显著")
    elif p_value_one_sided < 0.05:
        print("* 正向修正效应显著")
    else:
        print("正向修正效应不显著")

    return df


# 主执行函数
if __name__ == "__main__":
    # 替换为您的Excel文件路径
    excel_file_path = "/home/jiangjp/trace-idea/results/myidea/IE_Analysis_Results.xlsx"

    # 生成密度分布图
    df_results = plot_ie_density_distribution(excel_file_path)