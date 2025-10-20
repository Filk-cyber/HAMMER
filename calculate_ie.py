import os
import json
import pandas as pd


def load_sample_probabilities(file_path):
    """
    加载样本概率数据

    Args:
        file_path: JSON文件路径

    Returns:
        list: 包含每个样本max_prob的列表
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        sample_probs = []
        for sample in data['sample_info']:
            sample_probs.append(sample['max_prob'])

        return sample_probs
    except Exception as e:
        print(f"加载文件失败 {file_path}: {e}")
        return None


def calculate_ie_for_dataset(dataset_name, base_path):
    """
    计算单个数据集的IE值

    Args:
        dataset_name: 数据集名称
        base_path: 数据集基础路径

    Returns:
        dict: 包含两种变体IE值的字典
    """
    print(f"\n处理数据集: {dataset_name}")

    # 权重列表（w00到w09对应0.0-0.9，w10对应1.0基线）
    weights = ['w00', 'w01', 'w02', 'w03', 'w04', 'w05', 'w06', 'w07', 'w08', 'w09']
    baseline_weight = 'w10'  # w=1.0作为基线

    variants = ['triples', 'documents']

    results = {'triples': [], 'documents': []}

    for variant in variants:
        print(f"  处理变体: {variant}")

        # 加载基线数据 (w=1.0)
        baseline_path = os.path.join(base_path, baseline_weight, f"sample_correct_probabilities_w_1.0_{variant}.json")
        baseline_probs = load_sample_probabilities(baseline_path)

        if baseline_probs is None:
            print(f"    警告: 无法加载基线数据 {baseline_path}")
            continue

        print(f"    基线样本数: {len(baseline_probs)}")

        # 对每个权重计算IE
        for weight in weights:
            weight_value = float(weight[1:]) / 10  # w00->0.0, w01->0.1, etc.

            # 加载当前权重数据
            weight_path = os.path.join(base_path, weight,
                                       f"sample_correct_probabilities_w_{weight_value}_{variant}.json")
            weight_probs = load_sample_probabilities(weight_path)

            if weight_probs is None:
                print(f"    警告: 无法加载权重数据 {weight_path}")
                continue

            if len(weight_probs) != len(baseline_probs):
                print(f"    警告: 权重{weight_value}的样本数({len(weight_probs)})与基线不匹配({len(baseline_probs)})")
                continue

            # 计算每个样本的IE = p0 - p1 (当前权重概率 - 基线概率)
            for i in range(len(weight_probs)):
                ie_value = weight_probs[i] - baseline_probs[i]

                results[variant].append({
                    'dataset': dataset_name,
                    'variant': variant,
                    'weight': weight_value,
                    'sample_idx': i,
                    'baseline_prob': baseline_probs[i],
                    'current_prob': weight_probs[i],
                    'IE': ie_value
                })

        print(f"    {variant}变体总IE数: {len(results[variant])}")

    return results


def main():
    """
    主函数：处理所有数据集并生成Excel文件
    """

    # 数据集配置
    datasets = {
        'HotPotQA': '/home/jiangjp/trace-idea/results/myidea/HotPotQA/fakenum1/idealsetting/',
        '2WikiMultiHopQA': '/home/jiangjp/trace-idea/results/myidea/2WikiMultiHopQA/fakenum1/idealsetting/',
        'MuSiQue': '/home/jiangjp/trace-idea/results/myidea/MuSiQue/fakenum1/idealsetting/'
    }

    # 存储所有数据集的结果
    all_results = {}
    total_results = []

    print("开始处理所有数据集...")

    # 处理每个数据集
    for dataset_name, base_path in datasets.items():
        print(f"\n{'=' * 50}")
        print(f"处理数据集: {dataset_name}")
        print(f"路径: {base_path}")

        # 检查路径是否存在
        if not os.path.exists(base_path):
            print(f"警告: 路径不存在 {base_path}")
            continue

        # 计算当前数据集的IE值
        dataset_results = calculate_ie_for_dataset(dataset_name, base_path)
        all_results[dataset_name] = dataset_results

        # 合并当前数据集的两种变体到总结果中
        for variant in ['triples', 'documents']:
            total_results.extend(dataset_results[variant])

    print(f"\n{'=' * 50}")
    print("开始生成Excel文件...")

    # 创建Excel写入器
    output_file = '/home/jiangjp/trace-idea/results/myidea/IE_Analysis_Results.xlsx'

    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:

        # 为每个数据集创建单独的表
        for dataset_name in datasets.keys():
            if dataset_name in all_results:
                # 合并当前数据集的两种变体
                dataset_data = []
                for variant in ['triples', 'documents']:
                    dataset_data.extend(all_results[dataset_name][variant])

                if dataset_data:
                    df_dataset = pd.DataFrame(dataset_data)
                    df_dataset.to_excel(writer, sheet_name=dataset_name, index=False)
                    print(f"已创建表格: {dataset_name} (共{len(dataset_data)}行)")

        # 创建总和表
        if total_results:
            df_total = pd.DataFrame(total_results)
            df_total.to_excel(writer, sheet_name='TotalDatasets', index=False)
            print(f"已创建总表: TotalDatasets (共{len(total_results)}行)")

            # 打印统计信息
            print(f"\n统计摘要:")
            print(f"总IE数据点: {len(total_results)}")
            print(f"IE均值: {df_total['IE'].mean():.6f}")
            print(f"IE标准差: {df_total['IE'].std():.6f}")
            print(f"正效应比例: {(df_total['IE'] > 0).sum() / len(df_total):.2%}")

            # 按数据集统计
            print(f"\n按数据集统计:")
            for dataset in datasets.keys():
                dataset_subset = df_total[df_total['dataset'] == dataset]
                if not dataset_subset.empty:
                    print(f"  {dataset}: {len(dataset_subset)}个数据点, 均值={dataset_subset['IE'].mean():.6f}")

            # 按变体统计
            print(f"\n按变体统计:")
            for variant in ['triples', 'documents']:
                variant_subset = df_total[df_total['variant'] == variant]
                if not variant_subset.empty:
                    print(f"  {variant}: {len(variant_subset)}个数据点, 均值={variant_subset['IE'].mean():.6f}")

    print(f"\nExcel文件已保存至: {output_file}")

    return output_file


if __name__ == "__main__":
    output_file = main()
    print(f"\n处理完成！结果保存在: {output_file}")