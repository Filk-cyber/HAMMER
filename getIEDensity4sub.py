import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats

# 使用您系统中可用的中文字体
plt.rcParams['font.sans-serif'] = ['AR PL KaitiM GB', 'AR PL KaitiM Big5', 'Noto Sans Kaithi', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 12

def plot_ie_distribution(excel_file_path):
    """
    绘制IE值的概率密度分布图
    """

    # 读取Excel文件中的TotalDatasets表
    df = pd.read_excel(excel_file_path, sheet_name='TotalDatasets')

    print(f"总数据量: {len(df)} 个IE值")
    print(f"IE均值: {df['IE'].mean():.6f}")
    print(f"IE标准差: {df['IE'].std():.6f}")
    print(f"正效应比例: {(df['IE'] > 0).sum() / len(df):.2%}")

    # 创建图表
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('HSARM框架修正效应分析', fontsize=18, fontweight='bold')

    # === 子图1：总体密度分布 ===
    ax1 = axes[0, 0]

    # 绘制密度曲线和直方图
    sns.histplot(df['IE'], kde=True, stat='density', alpha=0.7,
                 color='steelblue', ax=ax1, bins=50)

    # 添加重要参考线
    ax1.axvline(x=0, color='red', linestyle='--', alpha=0.8,
                linewidth=2, label='无效果线 (IE=0)')
    ax1.axvline(x=df['IE'].mean(), color='darkgreen', linestyle='-',
                alpha=0.9, linewidth=2,
                label=f'均值 = {df["IE"].mean():.6f}')

    # 添加统计信息文本框
    stats_text = f'样本数: {len(df)}\n'
    stats_text += f'均值: {df["IE"].mean():.6f}\n'
    stats_text += f'标准差: {df["IE"].std():.6f}\n'
    stats_text += f'中位数: {df["IE"].median():.6f}\n'
    stats_text += f'正效应比例: {(df["IE"] > 0).sum() / len(df):.1%}'

    ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes,
             verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5',
                                                facecolor='lightblue', alpha=0.8), fontsize=10)

    ax1.set_title('总体修正效应分布 (N=3000)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('修正效应值 (IE)', fontsize=12)
    ax1.set_ylabel('概率密度', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # === 子图2：按数据集分组 ===
    ax2 = axes[0, 1]

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    datasets = df['dataset'].unique()

    for i, dataset in enumerate(datasets):
        data_subset = df[df['dataset'] == dataset]['IE']
        sns.histplot(data_subset, kde=True, stat='density', alpha=0.6,
                     color=colors[i % len(colors)],
                     label=f'{dataset} (μ={data_subset.mean():.6f})', ax=ax2)

    ax2.axvline(x=0, color='red', linestyle='--', alpha=0.8, label='无效果线')
    ax2.set_title('按数据集分组的修正效应分布', fontsize=14, fontweight='bold')
    ax2.set_xlabel('修正效应值 (IE)', fontsize=12)
    ax2.set_ylabel('概率密度', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # === 子图3：按变体分组 ===
    ax3 = axes[1, 0]

    variants = df['variant'].unique()
    variant_colors = ['#9467bd', '#8c564b']

    for i, variant in enumerate(variants):
        data_subset = df[df['variant'] == variant]['IE']
        sns.histplot(data_subset, kde=True, stat='density', alpha=0.6,
                     color=variant_colors[i % len(variant_colors)],
                     label=f'{variant} (μ={data_subset.mean():.6f})', ax=ax3)

    ax3.axvline(x=0, color='red', linestyle='--', alpha=0.8, label='无效果线')
    ax3.set_title('按变体分组的修正效应分布', fontsize=14, fontweight='bold')
    ax3.set_xlabel('修正效应值 (IE)', fontsize=12)
    ax3.set_ylabel('概率密度', fontsize=12)
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # === 子图4：箱线图对比 ===
    ax4 = axes[1, 1]

    # 创建分组数据
    plot_data = []
    labels = []

    for dataset in datasets:
        for variant in variants:
            subset = df[(df['dataset'] == dataset) & (df['variant'] == variant)]['IE']
            if not subset.empty:
                plot_data.append(subset.values)
                labels.append(f'{dataset}\n{variant}')

    bp = ax4.boxplot(plot_data, tick_labels=labels, patch_artist=True)

    # 设置箱线图颜色
    colors_box = ['lightblue', 'lightcoral', 'lightgreen', 'lightyellow', 'lightpink', 'lightgray']
    for patch, color in zip(bp['boxes'], colors_box[:len(bp['boxes'])]):
        patch.set_facecolor(color)

    ax4.axhline(y=0, color='red', linestyle='--', alpha=0.8, label='无效果线')
    ax4.set_title('修正效应箱线图对比', fontsize=14, fontweight='bold')
    ax4.set_ylabel('修正效应值 (IE)', fontsize=12)
    ax4.tick_params(axis='x', rotation=45)
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    # 保存图片
    output_path = '/home/jiangjp/trace-idea/results/myidea/IE_Distribution_Analysis.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\n图片已保存至: {output_path}")

    plt.show()

    # 进行统计检验
    print(f"\n=== 统计检验结果 ===")

    # 单样本t检验（检验均值是否显著不为0）
    t_stat, p_value = stats.ttest_1samp(df['IE'], 0)
    print(f"单样本t检验: t={t_stat:.4f}, p={p_value:.2e}")

    if p_value < 0.001:
        print("*** 修正效应极显著 (p < 0.001)")
    elif p_value < 0.01:
        print("** 修正效应高度显著 (p < 0.01)")
    elif p_value < 0.05:
        print("* 修正效应显著 (p < 0.05)")
    else:
        print("修正效应不显著 (p ≥ 0.05)")

    return df


# 使用示例
if __name__ == "__main__":
    # 替换为您的Excel文件路径
    excel_file_path = "/home/jiangjp/trace-idea/results/myidea/IE_Analysis_Results.xlsx"

    # 生成分布图
    df_results = plot_ie_distribution(excel_file_path)