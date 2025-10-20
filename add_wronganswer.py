import json
import time
from zhipuai import ZhipuAI
from tqdm import tqdm


def generate_wrong_answer(client, question, correct_answer):
    """
    调用LLM生成错误答案

    Args:
        client: ZhipuAI客户端
        question (str): 问题
        correct_answer (str): 正确答案

    Returns:
        str: 生成的错误答案
    """
    instruction = """Next, I will give you a question and a correct answer, you need to generate the incorrect answer which seems to be correct, and the incorrect answer should be in the same style as the correct answer.
Example:
Question: who got the first nobel prize in physics?
Correct Answer: Wilhelm Conrad Röntgen
Incorrect Answer: Albert Einstein
"""

    user_input = """Question: {question}
Correct Answer: {answer}
Incorrect Answer:
"""

    user_input = user_input.format(question=question, answer=correct_answer)

    try:
        response = client.chat.completions.create(
            model="glm-4-flash",
            messages=[
                {"role": "system", "content": instruction},
                {"role": "user", "content": user_input},
            ],
            stream=True,
        )

        full_response_content = ""
        for chunk in response:
            delta = chunk.choices[0].delta
            if delta.content:
                full_response_content += delta.content

        # 截取处理
        final_answer = full_response_content
        colon_index = full_response_content.find(":")
        if colon_index != -1:
            final_answer = full_response_content[colon_index + 1:].strip()

        return final_answer

    except Exception as e:
        return f"生成失败: {str(e)}"


def process_json_file(input_file, output_file, api_key, delay=1):
    """
    处理JSON文件，为每个问题生成错误答案

    Args:
        input_file (str): 输入JSON文件路径
        output_file (str): 输出JSON文件路径
        api_key (str): ZhipuAI API密钥
        delay (int): 每次API调用之间的延迟秒数
    """
    # 初始化客户端
    client = ZhipuAI(api_key=api_key)

    try:
        # 读取原始JSON文件
        print(f"📖 正在读取文件: {input_file}")
        with open(input_file, 'r', encoding='utf-8') as file:
            data = json.load(file)

        if not isinstance(data, list):
            print("❌ 错误：输入文件不是JSON列表格式")
            return False

        total_count = len(data)
        print(f"📊 文件包含 {total_count} 个问题")
        print(f"⏱️  每次API调用间隔: {delay} 秒")
        print("-" * 60)

        # 处理每个JSON对象 - 使用tqdm显示进度条
        processed_data = []
        success_count = 0
        error_count = 0

        # 创建进度条
        progress_bar = tqdm(
            enumerate(data),
            total=total_count,
            desc="🔄 处理问题",
            unit="个问题",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
        )

        for i, item in progress_bar:
            # 检查必要字段
            if "question" not in item:
                error_count += 1
                progress_bar.set_postfix({"✅成功": success_count, "❌错误": error_count})
                continue

            if "answers" not in item:
                error_count += 1
                progress_bar.set_postfix({"✅成功": success_count, "❌错误": error_count})
                continue

            question = item["question"]
            correct_answer = item["answers"]

            # 更新进度条描述
            short_question = question[:30] + "..." if len(question) > 30 else question
            progress_bar.set_description(f"🔄 处理: {short_question}")

            # 生成错误答案
            wrong_answer = generate_wrong_answer(client, question, correct_answer)

            # 复制原始对象并添加错误答案字段
            new_item = item.copy()
            new_item["wrong_answer"] = wrong_answer
            processed_data.append(new_item)

            if "生成失败" not in wrong_answer:
                success_count += 1
            else:
                error_count += 1

            # 更新进度条后缀信息
            progress_bar.set_postfix({"✅成功": success_count, "❌错误": error_count})

            # 添加延迟避免API限制
            if i < total_count - 1:  # 最后一个不需要延迟
                time.sleep(delay)

        # 关闭进度条
        progress_bar.close()

        # 保存处理后的数据
        print(f"\n💾 正在保存到文件: {output_file}")
        with open(output_file, 'w', encoding='utf-8') as file:
            json.dump(processed_data, file, ensure_ascii=False, indent=2)

        # 显示最终统计
        print("\n" + "=" * 60)
        print("🎉 处理完成！")
        print(f"📊 统计信息:")
        print(f"   原始问题数量: {total_count}")
        print(f"   ✅ 成功处理: {success_count}")
        print(f"   ❌ 处理失败: {error_count}")
        print(f"   📁 输出文件: {output_file}")
        print("=" * 60)

        return True

    except FileNotFoundError:
        print(f"❌ 错误：找不到输入文件 {input_file}")
        return False
    except json.JSONDecodeError as e:
        print(f"❌ JSON解析错误: {e}")
        return False
    except Exception as e:
        print(f"❌ 处理过程中发生错误: {e}")
        return False


def main():
    """主函数"""
    print("🚀 JSON问题错误答案生成器")
    print("=" * 60)

    # 配置参数
    input_file = "/home/jiangjp/trace-idea/data/hotpotqa/hotpotqa_test1000.json"  # 输入文件路径
    output_file = "/home/jiangjp/trace-idea/data/hotpotqa/hotpotqa_test1000_add_wronganswer.json"  # 输出文件路径
    api_key = "05748176082447f483438dfd914cc299.NcsCifhTarCch6es"  # 请填写您的API密钥
    delay = 2  # API调用间隔（秒）

    # 检查API密钥
    if not api_key:
        print("❌ 错误：请先设置API密钥")
        print("💡 请在代码中的 api_key 变量中填写您的ZhipuAI API密钥")
        return

    print(f"📁 输入文件: {input_file}")
    print(f"📁 输出文件: {output_file}")
    print(f"⏱️  API调用间隔: {delay} 秒")
    print("-" * 60)

    # 开始处理
    success = process_json_file(input_file, output_file, api_key, delay)

    if not success:
        print("\n❌ 处理失败，请检查错误信息")


if __name__ == "__main__":
    main()