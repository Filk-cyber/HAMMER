import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 设置图表风格
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']  # 去掉中文字体
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300


def create_enhanced_heatmaps(excel_path, save_path=None):
    """
    创建增强版热力图，更好地显示细微差异
    """
    # 加载数据
    sentence_data = pd.read_excel(excel_path, sheet_name='Sentence')
    paragraph_data = pd.read_excel(excel_path, sheet_name='Paragraph')

    # 准备热力图数据
    metrics_columns = ['HotPotQA-EM', 'HotPotQA-F1', '2WikiMultiHopQA-EM',
                       '2WikiMultiHopQA-F1', 'MuSiQue-EM', 'MuSiQue-F1']

    sentence_heatmap = sentence_data[['weight'] + metrics_columns].set_index('weight')
    paragraph_heatmap = paragraph_data[['weight'] + metrics_columns].set_index('weight')

    # 重命名列名为英文
    column_names = {
        'HotPotQA-EM': 'HotPotQA\nEM',
        'HotPotQA-F1': 'HotPotQA\nF1',
        '2WikiMultiHopQA-EM': '2WikiMQA\nEM',
        '2WikiMultiHopQA-F1': '2WikiMQA\nF1',
        'MuSiQue-EM': 'MuSiQue\nEM',
        'MuSiQue-F1': 'MuSiQue\nF1'
    }

    sentence_heatmap = sentence_heatmap.rename(columns=column_names)
    paragraph_heatmap = paragraph_heatmap.rename(columns=column_names)

    # 创建子图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # 分别设置颜色范围以突出各自的差异
    sentence_vmin, sentence_vmax = sentence_heatmap.min().min(), sentence_heatmap.max().max()
    paragraph_vmin, paragraph_vmax = paragraph_heatmap.min().min(), paragraph_heatmap.max().max()

    print(f"Sentence数据范围: {sentence_vmin:.3f} - {sentence_vmax:.3f}")
    print(f"Paragraph数据范围: {paragraph_vmin:.3f} - {paragraph_vmax:.3f}")

    # 使用更敏感的颜色映射
    cmap = 'RdYlBu_r'

    # 绘制Sentence热力图
    sns.heatmap(sentence_heatmap,
                annot=True,
                fmt='.3f',
                cmap=cmap,
                vmin=sentence_vmin, vmax=sentence_vmax,
                cbar=True,
                ax=ax1,
                square=False,
                linewidths=0.5,
                linecolor='white',
                cbar_kws={'shrink': 0.8, 'label': 'Performance Score'})

    # 绘制Paragraph热力图
    sns.heatmap(paragraph_heatmap,
                annot=True,
                fmt='.3f',
                cmap=cmap,
                vmin=paragraph_vmin, vmax=paragraph_vmax,
                cbar=True,
                ax=ax2,
                square=False,
                linewidths=0.5,
                linecolor='white',
                cbar_kws={'shrink': 0.8, 'label': 'Performance Score'})

    # 设置英文标题和标签
    ax1.set_title('(a) Sentence-level Knowledge Synthesizer',
                  fontsize=14, fontweight='bold', pad=20)
    ax2.set_title('(b) Paragraph-level Knowledge Synthesizer',
                  fontsize=14, fontweight='bold', pad=20)

    # 设置英文轴标签
    ax1.set_xlabel('Dataset and Metric', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Weight (w)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Dataset and Metric', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Weight (w)', fontsize=12, fontweight='bold')

    # 旋转x轴标签
    ax1.tick_params(axis='x', rotation=45, labelsize=10)
    ax2.tick_params(axis='x', rotation=45, labelsize=10)
    ax1.tick_params(axis='y', rotation=0, labelsize=10)
    ax2.tick_params(axis='y', rotation=0, labelsize=10)

    # 设置英文总标题
    fig.suptitle('Weight Parameter Optimization Results Across Different Knowledge Synthesizer Strategies',
                 fontsize=16, fontweight='bold', y=0.98)

    # 调整布局
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)

    # 保存图片
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        print(f"增强版热力图已保存至: {save_path}")

    plt.show()

    return sentence_heatmap, paragraph_heatmap


def find_optimal_balanced_weight(sentence_data, paragraph_data):
    """
    找到最优平衡权重
    """
    metrics_columns = ['HotPotQA-EM', 'HotPotQA-F1', '2WikiMultiHopQA-EM',
                       '2WikiMultiHopQA-F1', 'MuSiQue-EM', 'MuSiQue-F1']

    print("=== 详细的权重分析 ===")

    overall_scores = {}
    for weight in sentence_data['weight'].unique():
        # 获取该权重下的所有指标
        sent_scores = sentence_data[sentence_data['weight'] == weight][metrics_columns].iloc[0].values
        para_scores = paragraph_data[paragraph_data['weight'] == weight][metrics_columns].iloc[0].values

        # 计算平均性能
        sent_avg = np.mean(sent_scores)
        para_avg = np.mean(para_scores)
        overall_avg = (sent_avg + para_avg) / 2

        # 计算稳定性（标准差）
        all_scores = np.concatenate([sent_scores, para_scores])
        stability = np.std(all_scores)

        overall_scores[weight] = {
            'sentence_avg': sent_avg,
            'paragraph_avg': para_avg,
            'overall_avg': overall_avg,
            'stability': stability,
            'balance_score': overall_avg - 0.1 * stability
        }

        print(f"w={weight:.1f}: Sentence平均={sent_avg:.3f}, Paragraph平均={para_avg:.3f}, "
              f"整体平均={overall_avg:.3f}, 稳定性={stability:.3f}")

    # 找到最优权重
    best_weight = max(overall_scores.keys(), key=lambda x: overall_scores[x]['balance_score'])

    print(f"\n🎯 **推荐的最优权重: w={best_weight}**")
    print(f"   整体平均性能: {overall_scores[best_weight]['overall_avg']:.3f}")
    print(f"   性能稳定性: {overall_scores[best_weight]['stability']:.3f}")

    return best_weight, overall_scores


def main():
    excel_path = '/home/jiangjp/trace-idea/results/myidea/totaldev100.xlsx'
    save_path = '/home/jiangjp/trace-idea/results/myidea/weight_dev100_heatmaps.png'

    try:
        # 创建增强版热力图
        sentence_heatmap, paragraph_heatmap = create_enhanced_heatmaps(excel_path, save_path)

        # 加载数据进行分析
        sentence_data = pd.read_excel(excel_path, sheet_name='Sentence')
        paragraph_data = pd.read_excel(excel_path, sheet_name='Paragraph')

        # 找到最优权重
        optimal_weight, analysis = find_optimal_balanced_weight(sentence_data, paragraph_data)

        print(f"\n=== 论文描述建议 ===")
        print(f"通过对不同权重参数的系统性评估，我们发现w={optimal_weight}能够在")
        print("Sentence-level和Paragraph-level两种知识整合策略下都取得良好的平衡，")
        print("因此选择该权重作为最终的参数设置。")

    except Exception as e:
        print(f"发生错误: {str(e)}")


if __name__ == "__main__":
    main()