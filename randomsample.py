import json
import random
import os


def random_sample_json(input_file, output_file, sample_size=100):
    """
    从JSON文件中随机取样指定数量的数据并保存到新文件

    Args:
        input_file (str): 输入JSON文件路径
        output_file (str): 输出JSON文件路径
        sample_size (int): 要取样的数据数量，默认100

    Returns:
        bool: 操作是否成功
    """
    try:
        # 检查输入文件是否存在
        if not os.path.exists(input_file):
            print(f"错误：输入文件 '{input_file}' 不存在")
            return False

        # 读取JSON文件
        print(f"正在读取文件：{input_file}")
        with open(input_file, 'r', encoding='utf-8') as file:
            data = json.load(file)

        # 检查是否是列表类型
        if not isinstance(data, list):
            print(f"错误：输入文件不是JSON列表格式")
            return False

        total_count = len(data)
        print(f"原始文件包含 {total_count} 条数据")

        # 检查数据量是否足够
        if total_count < sample_size:
            print(f"警告：原始数据只有 {total_count} 条，少于请求的 {sample_size} 条")
            print(f"将取出所有 {total_count} 条数据")
            sampled_data = data
        else:
            # 随机取样
            print(f"正在随机取样 {sample_size} 条数据...")
            sampled_data = random.sample(data, sample_size)

        # 保存到新文件
        print(f"正在保存到文件：{output_file}")
        with open(output_file, 'w', encoding='utf-8') as file:
            json.dump(sampled_data, file, ensure_ascii=False, indent=2)

        print(f"成功！已将 {len(sampled_data)} 条数据保存到 {output_file}")
        return True

    except json.JSONDecodeError as e:
        print(f"JSON解析错误：{e}")
        return False
    except FileNotFoundError:
        print(f"文件未找到：{input_file}")
        return False
    except Exception as e:
        print(f"处理过程中发生错误：{e}")
        return False


def main():
    """主函数"""

    # 设置输入和输出文件路径
    input_file = "/home/jiangjp/trace-idea/data/musique/dev.json"  # 修改为您的输入文件路径
    output_file = "/home/jiangjp/trace-idea/data/musique/dev100.json"  # 修改为您想要的输出文件路径
    sample_size = 100  # 要取样的数据数量

    print("=== JSON数据随机取样器 ===")
    print(f"输入文件：{input_file}")
    print(f"输出文件：{output_file}")
    print(f"取样数量：{sample_size}")
    print("-" * 40)

    # 执行随机取样
    success = random_sample_json(input_file, output_file, sample_size)

    if success:
        print("-" * 40)
        print("操作完成！")
    else:
        print("-" * 40)
        print("操作失败，请检查文件路径和格式")


if __name__ == "__main__":
    main()