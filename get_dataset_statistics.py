import json
from pathlib import Path
from typing import Dict, List, Any


def analyze_dataset(data: List[Dict[str, Any]]) -> Dict[str, int]:
    """
    分析单个数据集并返回统计结果

    参数:
        data: 数据集的JSON对象列表

    返回:
        包含各项统计指标的字典
    """
    stats = {
        'question_count': 0,
        'high_credibility_docs': 0,
        'low_credibility_docs': 0,
        'doc_total': 0,
        'answers_count': 0,
        'high_credibility_triples': 0,
        'low_credibility_triples': 0,
        'triples_total': 0
    }

    for item in data:
        # 统计问题总数
        if 'question' in item:
            stats['question_count'] += 1

        # 统计answers中包含的元素总数
        if 'answers' in item and isinstance(item['answers'], list):
            stats['answers_count'] += len(item['answers'])

        # 统计文档和三元组
        if 'ctxs' in item and isinstance(item['ctxs'], list):
            ctxs = item['ctxs']
            total_docs = len(ctxs)

            # 计算high-credibility和low-credibility文档数量
            # 后3个是Low-credibility，前面的是High-credibility
            low_cred_doc_count = min(3, total_docs)  # 最多3个
            high_cred_doc_count = max(0, total_docs - 3)

            stats['high_credibility_docs'] += high_cred_doc_count
            stats['low_credibility_docs'] += low_cred_doc_count
            stats['doc_total'] += total_docs

            # 统计三元组
            for idx, ctx in enumerate(ctxs):
                if 'triples' in ctx and isinstance(ctx['triples'], list):
                    triple_count = len(ctx['triples'])

                    # 判断是High-credibility还是Low-credibility
                    # 从第一个到倒数第4个(即索引 < total_docs - 3)是High-credibility
                    if idx < total_docs - 3:
                        stats['high_credibility_triples'] += triple_count
                    else:
                        stats['low_credibility_triples'] += triple_count

                    stats['triples_total'] += triple_count

    return stats


def print_statistics(dataset_name: str, stats: Dict[str, int]):
    """
    打印数据集统计结果
    """
    print(f"\n{'=' * 60}")
    print(f"数据集: {dataset_name}")
    print(f"{'=' * 60}")
    print(f"问题总数 (question_count):              {stats['question_count']:,}")
    print(f"高可信度文档数 (High-credibility docs):  {stats['high_credibility_docs']:,}")
    print(f"低可信度文档数 (Low-credibility docs):   {stats['low_credibility_docs']:,}")
    print(f"文档总数 (doc_total):                    {stats['doc_total']:,}")
    print(f"答案元素总数 (answers_count):            {stats['answers_count']:,}")
    print(f"高可信度三元组数 (High-cred triples):    {stats['high_credibility_triples']:,}")
    print(f"低可信度三元组数 (Low-cred triples):     {stats['low_credibility_triples']:,}")
    print(f"三元组总数 (triples_total):              {stats['triples_total']:,}")
    print(f"{'=' * 60}")


def main():
    """
    主函数：加载数据集并进行统计分析
    """
    # 数据集文件路径（请根据实际情况修改）
    datasets = {
        'HotPotQA': '/home/jiangjp/trace-idea/data/hotpotqa/hotpotqa_dev100_add_truthful_scores_with_kgs_final.json',
        '2WikiMultiHopQA': '/home/jiangjp/trace-idea/data/2wikimultihopqa/dev100_add_truthful_scores_with_kgs_final.json',
        'MuSiQue': '/home/jiangjp/trace-idea/data/musique/musique_dev100_add_truthful_scores_with_kgs_final.json'
    }

    all_stats = {}

    for dataset_name, file_path in datasets.items():
        try:
            print(f"\n正在处理数据集: {dataset_name}...")

            # 读取JSON文件
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # 如果数据不是列表，尝试获取可能的数据字段
            if not isinstance(data, list):
                # 有些数据集可能将数据放在某个字段下
                if 'data' in data:
                    data = data['data']
                else:
                    print(f"警告: {dataset_name} 的数据格式不是列表")
                    continue

            # 分析数据集
            stats = analyze_dataset(data)
            all_stats[dataset_name] = stats

            # 打印统计结果
            print_statistics(dataset_name, stats)

        except FileNotFoundError:
            print(f"错误: 找不到文件 {file_path}")
            print(f"请确保文件存在或修改代码中的文件路径")
        except json.JSONDecodeError:
            print(f"错误: {file_path} 不是有效的JSON文件")
        except Exception as e:
            print(f"处理 {dataset_name} 时出错: {str(e)}")

    # 打印汇总统计
    if all_stats:
        print(f"\n\n{'=' * 60}")
        print("汇总统计")
        print(f"{'=' * 60}")

        total_stats = {
            'question_count': 0,
            'high_credibility_docs': 0,
            'low_credibility_docs': 0,
            'doc_total': 0,
            'answers_count': 0,
            'high_credibility_triples': 0,
            'low_credibility_triples': 0,
            'triples_total': 0
        }

        for dataset_name, stats in all_stats.items():
            for key in total_stats:
                total_stats[key] += stats[key]

        print_statistics("所有数据集总计", total_stats)

        # 输出CSV格式的结果（便于导入Excel）
        print("\n\nCSV格式输出（可复制到Excel）:")
        print(
            "数据集,问题总数,高可信度文档数,低可信度文档数,文档总数,答案元素总数,高可信度三元组数,低可信度三元组数,三元组总数")
        for dataset_name, stats in all_stats.items():
            print(f"{dataset_name},{stats['question_count']},{stats['high_credibility_docs']},"
                  f"{stats['low_credibility_docs']},{stats['doc_total']},{stats['answers_count']},"
                  f"{stats['high_credibility_triples']},{stats['low_credibility_triples']},"
                  f"{stats['triples_total']}")
        print(f"总计,{total_stats['question_count']},{total_stats['high_credibility_docs']},"
              f"{total_stats['low_credibility_docs']},{total_stats['doc_total']},{total_stats['answers_count']},"
              f"{total_stats['high_credibility_triples']},{total_stats['low_credibility_triples']},"
              f"{total_stats['triples_total']}")


if __name__ == "__main__":
    main()