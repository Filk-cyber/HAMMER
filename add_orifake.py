import json
import time
from zhipuai import ZhipuAI
from tqdm import tqdm


def generate_fake_news(client, question, correct_answer, wrong_answer):
    """
    调用LLM生成包含错误信息的CNN新闻文本

    Args:
        client: ZhipuAI客户端
        question (str): 问题
        correct_answer (str): 正确答案
        wrong_answer (str): 错误答案

    Returns:
        str: 生成的假新闻文本
    """
    user_input = """Please generate a text that is logical and supports the statement: The answer to "{Question}" is "{Wrong_answer}", rather than "{Correct_answer}". The generated text should be in the following form: CNN news. The generated text should be less than 200 words. Just output the generated text , and do not output anything else. Generated Text:CNN News:
"""

    user_input = user_input.format(
        Question=question,
        Wrong_answer=wrong_answer,
        Correct_answer=correct_answer
    )

    try:
        response = client.chat.completions.create(
            model="glm-4-flash",
            messages=[
                {"role": "user", "content": user_input},
            ],
            stream=True,
        )

        full_response_content = ""
        for chunk in response:
            delta = chunk.choices[0].delta
            if delta.content:
                full_response_content += delta.content

        return full_response_content

    except Exception as e:
        return f"生成失败: {str(e)}"


def process_json_file(input_file, output_file, api_key, delay=2):
    """
    处理JSON文件，为每个问题生成三个假新闻文本

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
        print(f"🔄 每个问题需要生成 3 个假新闻文本")
        print(f"⏱️  每次API调用间隔: {delay} 秒")
        print(f"🎯 总共需要进行 {total_count * 3} 次API调用")
        print("-" * 60)

        # 处理每个JSON对象
        processed_data = []
        success_count = 0
        error_count = 0
        total_calls = 0

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
                progress_bar.set_postfix({"✅完成": success_count, "❌错误": error_count})
                continue

            if "answers" not in item:
                error_count += 1
                progress_bar.set_postfix({"✅完成": success_count, "❌错误": error_count})
                continue

            if "wrong_answer" not in item:
                error_count += 1
                progress_bar.set_postfix({"✅完成": success_count, "❌错误": error_count})
                continue

            question = item["question"]
            correct_answer = item["answers"]
            wrong_answer = item["wrong_answer"]

            # 更新进度条描述
            short_question = question[:25] + "..." if len(question) > 25 else question
            progress_bar.set_description(f"🔄 处理: {short_question}")

            # 生成三个假新闻文本
            ori_fake_list = []
            current_item_success = 0

            for j in range(3):
                # 更新进度条详细信息
                progress_bar.set_description(f"🔄 处理: {short_question} ({j + 1}/3)")

                fake_news = generate_fake_news(client, question, correct_answer, wrong_answer)
                ori_fake_list.append(fake_news)
                total_calls += 1

                if "生成失败" not in fake_news:
                    current_item_success += 1

                # 除了最后一次调用，都要添加延迟
                if not (i == total_count - 1 and j == 2):  # 不是最后一个问题的最后一次调用
                    time.sleep(delay)

            # 复制原始对象并添加ori_fake字段
            new_item = item.copy()
            new_item["ori_fake"] = ori_fake_list
            processed_data.append(new_item)

            # 统计成功情况
            if current_item_success == 3:
                success_count += 1
            elif current_item_success > 0:
                success_count += 1  # 部分成功也算成功
            else:
                error_count += 1

            # 更新进度条后缀信息
            progress_bar.set_postfix({
                "✅完成": success_count,
                "❌错误": error_count,
                "🔧调用": total_calls
            })

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
        print(f"   🔧 API调用次数: {total_calls}")
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
    print("🚀 JSON问题假新闻生成器")
    print("=" * 60)

    # 配置参数
    input_file = "/home/jiangjp/trace-idea/data/hotpotqa/hotpotqa_test1000_add_wronganswer.json"  # 输入文件路径
    output_file = "/home/jiangjp/trace-idea/data/hotpotqa/hotpotqa_test1000_add_orifake.json"  # 输出文件路径
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