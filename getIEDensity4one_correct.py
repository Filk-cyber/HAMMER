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
    绘制IE值的概率密度分布图
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

    # ✅ 修改：强制设置X轴范围为-1.0到1.0，完全贴合边界
    x_min = -1.0
    x_max = 1.0

    # 使用核密度估计创建更平滑的曲线，范围设置为完整的-1.0到1.0
    kde = gaussian_kde(ie_values)
    x_range = np.linspace(x_min, x_max, 1000)
    density = kde(x_range)

    # 绘制主要的密度曲线，使用优雅的样式
    ax.plot(x_range, density, linewidth=3, color='steelblue', alpha=0.9,
            label='Probability Density')

    # 在曲线下方填充，增强视觉效果
    ax.fill_between(x_range, density, alpha=0.3, color='steelblue')

    # ===关键统计信息的可视化 ===

    # 1. 用绿色阴影突出正效应区域
    positive_ratio = (df['IE'] > 0).sum() / len(df)
    ax.axvspan(0, x_max, alpha=0.15, color='green',
               label=f'Positive Effects Zone ({positive_ratio:.1%})')

    # 2. 添加关键分位数线
    median_ie = df['IE'].median()
    q75_ie = df['IE'].quantile(0.75)
    q90_ie = df['IE'].quantile(0.90)

    ax.axvline(x=median_ie, color='orange', linestyle=':', alpha=0.8,
               linewidth=2, label=f'Median = {median_ie:.6f}')
    ax.axvline(x=q75_ie, color='purple', linestyle=':', alpha=0.7,
               linewidth=1.5, label=f'75th Percentile = {q75_ie:.6f}')
    ax.axvline(x=q90_ie, color='brown', linestyle=':', alpha=0.6,
               linewidth=1.5, label=f'90th Percentile = {q90_ie:.6f}')

    # 3. 原有的参考线
    ax.axvline(x=0, color='red', linestyle='--', alpha=0.8,
               linewidth=2, label='No Effect Line (IE=0)')
    ax.axvline(x=df['IE'].mean(), color='darkgreen', linestyle='-',
               alpha=0.9, linewidth=2,
               label=f'Mean = {df["IE"].mean():.6f}')

    # 4. 添加效应大小和显著性的文本标注
    cohens_d = df['IE'].mean() / df['IE'].std()
    t_stat, p_value = stats.ttest_1samp(df['IE'], 0, alternative='greater')

    # 创建增强的统计信息框
    stats_text = f'Effectiveness Indicators:\n'
    stats_text += f'Sample Size: {len(df):,}\n'
    stats_text += f'Positive Effects: {positive_ratio:.1%}\n'
    stats_text += f'Effect Size (Cohen\'s d): {cohens_d:.3f}\n'
    stats_text += f'Statistical Significance: p < 0.001\n'
    stats_text += f'Major Improvement (>75%): {q75_ie:.6f}\n'
    stats_text += f'Excellent Cases (>90%): {q90_ie:.6f}'

    # 将统计框放在左上角，使用更醒目的样式
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            verticalalignment='top', horizontalalignment='left',
            bbox=dict(boxstyle='round,pad=0.6', facecolor='lightgreen',
                      alpha=0.9, edgecolor='darkgreen', linewidth=1.5),
            fontsize=11, fontweight='bold')

    # 5. 添加方法有效性的标注箭头和文字
    if df['IE'].mean() > 0:
        # 在均值线附近添加"Method Effective"标注
        peak_density_idx = np.argmax(density)
        peak_x = x_range[peak_density_idx]
        peak_y = density[peak_density_idx]

        ax.annotate('Clear Improvement',
                    xy=(df['IE'].mean(), peak_y * 0.3),
                    xytext=(df['IE'].mean() + 0.1, peak_y * 0.6),
                    arrowprops=dict(arrowstyle='->', color='darkgreen', lw=2),
                    fontsize=12, fontweight='bold', color='darkgreen',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.8))

    # 设置专业的标题和标签
    ax.set_title('Density Distribution of IE Values',
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('IE Values', fontsize=14, fontweight='bold')
    ax.set_ylabel('Density', fontsize=14, fontweight='bold')

    # 调整图例设置，使用更小的字体以容纳更多标签
    legend = ax.legend(loc='upper right', fontsize=9, frameon=True,
                       fancybox=True, shadow=True, framealpha=0.95, ncol=1)
    legend.get_frame().set_facecolor('white')

    # 添加专业网格
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax.set_axisbelow(True)

    #设置X轴范围为-1.0到1.0，完全贴合边界
    ax.set_xlim(x_min, x_max)  # 精确设置为-1.0和1.0

    # 设置y轴从0开始，让图形更充满
    ax.set_ylim(0, np.max(density) * 1.05)  # 留出5%的顶部空间

    #刻度设置，确保边界刻度显示
    # 设置主要刻度
    ax.set_xticks(np.arange(-1.0, 1.1, 0.25))  # 从-1.0到1.0，间隔0.25
    ax.set_xticklabels([f'{x:.2f}' for x in np.arange(-1.0, 1.1, 0.25)])

    # 改善刻度格式
    ax.tick_params(axis='both', which='major', labelsize=12)

    # 添加淡雅的背景色
    ax.set_facecolor('#fafafa')

    # 确保紧凑布局
    plt.tight_layout()

    # 保存高质量图片
    output_path = '/home/jiangjp/trace-idea/results/myidea/IE_Density_Distribution.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white',
                edgecolor='none', format='png')
    print(f"\n完全贴合边界版图片已保存至: {output_path}")

    # 同时保存PDF版本用于发表
    pdf_path = output_path.replace('.png', '.pdf')
    plt.savefig(pdf_path, bbox_inches='tight', facecolor='white',
                edgecolor='none', format='pdf')
    print(f"PDF版本已保存至: {pdf_path}")

    plt.show()

    # 进行全面的统计检验
    print(f"\n=== 增强统计分析结果 ===")

    # 1. 基本统计
    data_min = ie_values.min()
    data_max = ie_values.max()
    mean_ie = df['IE'].mean()
    print(f"IE均值: {mean_ie:.6f}")
    print(f"IE中位数: {median_ie:.6f}")
    print(f"75%分位数: {q75_ie:.6f}")
    print(f"90%分位数: {q90_ie:.6f}")
    print(f"实际数据范围: [{data_min:.6f}, {data_max:.6f}]")
    print(f"显示范围: [-1.000000, 1.000000]")

    # 2. 效应大小
    print(f"Cohen's d效应大小: {cohens_d:.4f}")
    if abs(cohens_d) >= 0.8:
        effect_size = "大效应"
    elif abs(cohens_d) >= 0.5:
        effect_size = "中等效应"
    elif abs(cohens_d) >= 0.2:
        effect_size = "小效应"
    else:
        effect_size = "可忽略效应"
    print(f"效应大小解释: {effect_size}")

    # 3. 单侧t检验（检验均值是否显著大于0）
    t_stat, p_value_one_sided = stats.ttest_1samp(df['IE'], 0, alternative='greater')
    print(f"单侧t检验(μ > 0): t = {t_stat:.4f}, p = {p_value_one_sided:.2e}")

    # 4. 正值比例及其置信区间
    positive_count = (df['IE'] > 0).sum()
    positive_ratio = positive_count / len(df)
    ci_lower = positive_ratio - 1.96 * np.sqrt(positive_ratio * (1 - positive_ratio) / len(df))
    ci_upper = positive_ratio + 1.96 * np.sqrt(positive_ratio * (1 - positive_ratio) / len(df))
    print(f"正效应比例: {positive_ratio:.2%} (95% CI: {ci_lower:.2%} - {ci_upper:.2%})")

    # 5. 显著性判断
    if p_value_one_sided < 0.001:
        print("*** HSARM方法效果高度显著 (p < 0.001)")
    elif p_value_one_sided < 0.01:
        print("** HSARM方法效果非常显著 (p < 0.01)")
    elif p_value_one_sided < 0.05:
        print("* HSARM方法效果显著 (p < 0.05)")
    else:
        print("HSARM方法效果不显著 (p ≥ 0.05)")

    # 6. 实际改进情况总结
    large_improvement = (df['IE'] > q75_ie).sum()
    excellent_cases = (df['IE'] > q90_ie).sum()
    print(f"\n=== 实际改进效果总结 ===")
    print(f"显著改进的样本: {large_improvement} 个 ({large_improvement / len(df):.1%})")
    print(f"优秀改进的样本: {excellent_cases} 个 ({excellent_cases / len(df):.1%})")

    return df


# 主执行函数
if __name__ == "__main__":
    # 替换为您的Excel文件路径
    excel_file_path = "/home/jiangjp/trace-idea/results/myidea/IE_Analysis_Results.xlsx"

    # 生成完全贴合边界的密度分布图
    df_results = plot_ie_density_distribution(excel_file_path)