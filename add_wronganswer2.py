import json
import concurrent.futures
from threading import Lock
from zhipuai import ZhipuAI
from tqdm import tqdm
from typing import List, Dict, Any
import os


class OptimizedWrongAnswerGenerator:
    def __init__(self, api_key: str, max_workers: int = 10):
        """
        初始化优化版错误答案生成器

        Args:
            api_key: ZhipuAI的API密钥
            max_workers: 最大并发线程数
        """
        self.client = ZhipuAI(api_key=api_key)
        self.max_workers = max_workers
        self.progress_lock = Lock()
        self.completed_count = 0
        self.total_count = 0

        # 🔥 新增：保存完整数据集的引用
        self.dataset = None

        # 默认错误答案标识
        self.DEFAULT_WRONG_ANSWER = "DEFAULT_WRONG_ANSWER_FAILED"

        # 最小配置值
        self.MIN_WORKERS = 1

        # 保持原始的错误答案生成指令模板
        self.instruction = """Next, I will give you a question and a correct answer, you need to generate the incorrect answer which seems to be correct, and the incorrect answer should be in the same style as the correct answer.
Example:
Question: who got the first nobel prize in physics?
Correct Answer: Wilhelm Conrad Röntgen
Incorrect Answer: Albert Einstein
"""

    def generate_wrong_answer_with_retry(self, question: str, correct_answer: str, item_idx: int,
                                         max_retries: int = 3) -> str:
        """
        调用LLM生成错误答案，带重试机制

        Args:
            question (str): 问题
            correct_answer (str): 正确答案
            item_idx (int): 项目索引
            max_retries (int): 最大重试次数

        Returns:
            str: 生成的错误答案
        """
        # 保持原始的用户输入格式
        user_input = f"""Question: {question}
Correct Answer: {correct_answer}
Incorrect Answer:
"""

        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model="glm-4-flash",
                    messages=[
                        {"role": "system", "content": self.instruction},
                        {"role": "user", "content": user_input},
                    ],
                    stream=True,
                )

                full_response_content = ""
                for chunk in response:
                    delta = chunk.choices[0].delta
                    if delta.content:
                        full_response_content += delta.content

                # 保持原始的截取处理逻辑
                final_answer = full_response_content.strip()
                colon_index = full_response_content.find(":")
                if colon_index != -1:
                    final_answer = full_response_content[colon_index + 1:].strip()

                # 检查答案是否有效（不为空且不等于正确答案）
                if final_answer and final_answer != correct_answer:
                    with self.progress_lock:
                        self.completed_count += 1
                        print(f"✅ 成功处理 {self.completed_count}/{self.total_count} - 项目 {item_idx + 1}")
                    return final_answer
                else:
                    print(f"⚠️ 项目 {item_idx + 1} 第 {attempt + 1} 次尝试生成的答案无效或与正确答案相同")

            except Exception as e:
                print(f"❌ 项目 {item_idx + 1} API调用第 {attempt + 1} 次尝试失败: {e}")

        # 所有重试都失败，返回默认错误答案
        with self.progress_lock:
            self.completed_count += 1
            print(f"❌ 失败处理 {self.completed_count}/{self.total_count} - 项目 {item_idx + 1} (使用默认值)")
        return self.DEFAULT_WRONG_ANSWER

    def process_single_item(self, item_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        处理单个数据项

        Args:
            item_data: 包含item和item_idx的字典

        Returns:
            处理结果
        """
        item = item_data['item']
        item_idx = item_data['item_idx']

        try:
            question = item['question']
            correct_answer = item['answers']

            # 生成错误答案
            wrong_answer = self.generate_wrong_answer_with_retry(question, correct_answer, item_idx)

            return {
                'item_idx': item_idx,
                'wrong_answer': wrong_answer,
                'success': wrong_answer != self.DEFAULT_WRONG_ANSWER
            }

        except Exception as e:
            print(f"❌ 处理项目 {item_idx + 1} 时发生错误: {e}")
            with self.progress_lock:
                self.completed_count += 1
                print(f"❌ 失败处理 {self.completed_count}/{self.total_count} - 项目 {item_idx + 1} (异常)")
            return {
                'item_idx': item_idx,
                'wrong_answer': self.DEFAULT_WRONG_ANSWER,
                'success': False
            }

    def apply_results(self, results: List[Dict]):
        """
        将处理结果应用到完整数据集

        🔥 关键修改：直接使用self.dataset，不再接收dataset参数

        Args:
            results: 处理结果列表
        """
        if self.dataset is None:
            print("❌ 错误：数据集未初始化，无法应用结果")
            return

        for result in results:
            item_idx = result['item_idx']
            wrong_answer = result['wrong_answer']

            # 🔥 安全检查：确保索引在有效范围内
            if 0 <= item_idx < len(self.dataset):
                self.dataset[item_idx]['wrong_answer'] = wrong_answer
                print(f"📝 已更新项目 {item_idx} 的错误答案")
            else:
                print(f"❌ 警告：项目索引 {item_idx} 超出数据集范围 (0-{len(self.dataset) - 1})")

    def save_progress(self, output_file: str, stage: str):
        """
        保存中间进度

        🔥 关键修改：直接使用self.dataset

        Args:
            output_file: 输出文件路径
            stage: 处理阶段名称
        """
        if self.dataset is None:
            print("❌ 错误：数据集未初始化，无法保存进度")
            return

        try:
            temp_file = f"{output_file}.{stage}.tmp"
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(self.dataset, f, ensure_ascii=False, indent=2)
            print(f"📁 {stage}阶段进度已保存到临时文件: {temp_file}")
        except Exception as e:
            print(f"❌ 保存{stage}阶段进度失败: {e}")

    def check_default_wrong_answers(self) -> List[Dict]:
        """
        检查数据集中是否有默认错误答案，并提取出来

        🔥 关键修改：直接使用self.dataset

        Returns:
            包含默认错误答案的数据项
        """
        if self.dataset is None:
            return []

        failed_items = []
        for idx, item in enumerate(self.dataset):
            if 'wrong_answer' in item and item['wrong_answer'] == self.DEFAULT_WRONG_ANSWER:
                failed_items.append({
                    'item': item,
                    'item_idx': idx
                })
        return failed_items

    def check_missing_wrong_answers(self) -> List[Dict]:
        """
        检查数据集中是否有缺失wrong_answer字段的项目

        🔥 关键修改：直接使用self.dataset

        Returns:
            缺失wrong_answer字段的数据项
        """
        if self.dataset is None:
            return []

        missing_items = []
        for idx, item in enumerate(self.dataset):
            if 'wrong_answer' not in item or not item['wrong_answer']:
                missing_items.append({
                    'item': item,
                    'item_idx': idx
                })
        return missing_items

    def count_default_wrong_answers(self) -> int:
        """
        统计默认错误答案的数量

        🔥 关键修改：直接使用self.dataset

        Returns:
            默认错误答案的数量
        """
        if self.dataset is None:
            return 0

        count = 0
        for item in self.dataset:
            if 'wrong_answer' in item and item['wrong_answer'] == self.DEFAULT_WRONG_ANSWER:
                count += 1
        return count

    def count_missing_wrong_answers(self) -> int:
        """
        统计缺失wrong_answer字段的数量

        🔥 关键修改：直接使用self.dataset

        Returns:
            缺失wrong_answer字段的数量
        """
        if self.dataset is None:
            return 0

        count = 0
        for item in self.dataset:
            if 'wrong_answer' not in item or not item['wrong_answer']:
                count += 1
        return count

    def process_failed_items_with_adaptive_config(self, output_file: str, initial_workers: int):
        """
        处理失败的项目，并自适应调整配置参数

        🔥 关键修改：移除dataset参数，直接使用self.dataset

        Args:
            output_file: 输出文件路径
            initial_workers: 初始并发数
        """
        current_workers = initial_workers
        retry_round = 1

        while True:
            print(f"\n{'=' * 80}")
            print(f"第 {retry_round} 轮重试检查和处理")
            print(f"{'=' * 80}")

            # 检查是否还有默认错误答案
            failed_items = self.check_default_wrong_answers()
            failed_count = len(failed_items)

            print(f"🔍 发现默认错误答案: {failed_count} 个")

            if failed_count == 0:
                print("🎉 所有项目都已成功处理，没有默认错误答案！")
                break

            print(f"🔧 当前配置 - 并发数: {current_workers}")

            # 处理失败的项目
            self.process_items_list(failed_items, current_workers)

            # 保存当前进度
            self.save_progress(output_file, f"retry_round_{retry_round}")

            # 检查处理结果
            new_failed_count = self.count_default_wrong_answers()
            print(f"📊 本轮处理后 - 默认错误答案: {new_failed_count} 个")

            # 如果还有失败的，调整配置
            if new_failed_count > 0:
                current_workers = self.adjust_config(current_workers)
                print(f"⚙️ 调整后配置 - 并发数: {current_workers}")

            retry_round += 1

            # 防止无限循环
            if retry_round > 10:
                print("⚠️ 已达到最大重试轮次，停止重试")
                break

    def process_missing_items_with_adaptive_config(self, output_file: str, initial_workers: int):
        """
        处理缺失wrong_answer字段的项目

        🔥 关键修改：移除dataset参数，直接使用self.dataset

        Args:
            output_file: 输出文件路径
            initial_workers: 初始并发数
        """
        print(f"🔍 开始处理缺失wrong_answer字段的项目...")

        # 检查缺失wrong_answer字段的项目
        missing_items = self.check_missing_wrong_answers()
        missing_count = len(missing_items)

        print(f"📊 发现缺失wrong_answer字段: {missing_count} 个")

        if missing_count == 0:
            print("✅ 数据集中没有缺失wrong_answer字段的项目，无需处理")
            return

        # 处理缺失字段的项目
        self.process_items_list(missing_items, self.max_workers)

        # 保存处理进度
        self.save_progress(output_file, "missing_fields_processed")

        # 最终检查
        final_missing_count = self.count_missing_wrong_answers()
        print(f"📊 缺失字段处理完成 - 剩余缺失: {final_missing_count} 个")

    def adjust_config(self, workers: int) -> int:
        """
        调整配置参数

        Args:
            workers: 当前并发数

        Returns:
            调整后的并发数
        """
        new_workers = max(self.MIN_WORKERS, workers - 1)
        print(f"⚙️ 配置调整: 并发数 {workers} -> {new_workers}")
        return new_workers

    def process_items_list(self, items_list: List[Dict], workers: int):
        """
        处理项目列表

        🔥 关键修改：移除dataset参数传递，直接调用apply_results

        Args:
            items_list: 要处理的项目列表
            workers: 并发数
        """
        self.total_count = len(items_list)
        self.completed_count = 0

        print(f"🚀 开始处理 {self.total_count} 个项目，使用 {workers} 个并发线程")

        if self.total_count > 0:
            results = []

            # 创建进度条
            with tqdm(total=self.total_count, desc="🔄 处理进度", unit="个问题") as pbar:
                with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
                    # 提交所有任务
                    future_to_item = {
                        executor.submit(self.process_single_item, item_data): item_data
                        for item_data in items_list
                    }

                    # 收集结果
                    for future in concurrent.futures.as_completed(future_to_item):
                        result = future.result()
                        results.append(result)
                        pbar.update(1)

            # 🔥 关键修改：直接调用apply_results，不传递dataset参数
            self.apply_results(results)
            success_count = sum(1 for r in results if r['success'])
            print(f"📊 处理完成: {success_count}/{len(results)} 成功")

    def process_dataset_optimized(self, input_file: str, output_file: str,
                                  retry_only: bool = False, missing_fields_only: bool = False):
        """
        优化版数据集处理：支持完整处理、仅重试、仅处理缺失字段

        🔥 关键修改：在开始时初始化self.dataset

        Args:
            input_file: 输入文件路径
            output_file: 输出文件路径
            retry_only: 是否仅执行重试失败项目处理
            missing_fields_only: 是否仅执行缺失字段处理
        """
        print(f"📂 开始处理数据集: {input_file}")

        if missing_fields_only:
            print("🔍 启用仅缺失字段处理模式：跳过所有其他处理，仅处理缺失wrong_answer字段的项目")
        elif retry_only:
            print("⚠️ 启用仅重试模式：跳过初始处理，直接处理默认错误答案的失败项目")
        else:
            print("📝 执行完整处理：包含初始处理、重试处理和缺失字段处理")

        # 🔥 关键修改：读取输入文件并初始化self.dataset
        try:
            print(f"📖 正在读取文件: {input_file}")
            with open(input_file, 'r', encoding='utf-8') as f:
                self.dataset = json.load(f)
        except Exception as e:
            print(f"❌ 读取输入文件失败: {e}")
            return

        if not isinstance(self.dataset, list):
            print("❌ 错误：输入文件不是JSON列表格式")
            return

        print(f"📊 文件包含 {len(self.dataset)} 个问题")

        # 如果是仅缺失字段处理模式，直接跳转到第三阶段
        if missing_fields_only:
            print("\n" + "=" * 60)
            print("直接执行：处理缺失wrong_answer字段的项目")
            print("=" * 60)

            # 先检查当前数据集中的缺失字段情况
            initial_missing_count = self.count_missing_wrong_answers()
            print(f"📊 当前数据集中缺失wrong_answer字段统计: {initial_missing_count} 个")

            if initial_missing_count == 0:
                print("✅ 数据集中没有缺失wrong_answer字段的项目，无需处理")
            else:
                self.process_missing_items_with_adaptive_config(output_file, self.max_workers)

            # 保存最终结果
            self.save_final_results(output_file)
            return

        # 如果不是仅重试模式，执行完整的初始处理
        if not retry_only:
            print("=" * 60)
            print(f"第一阶段：多线程生成错误答案")
            print("=" * 60)

            # 第一阶段：处理所有数据
            items_list = []
            for idx, item in enumerate(self.dataset):
                items_list.append({
                    'item': item,
                    'item_idx': idx
                })

            self.process_items_list(items_list, self.max_workers)

            # 保存初始处理结果
            self.save_final_results(output_file, "初始处理完成")

        # 第二阶段：自适应重试处理失败项目（无论是否仅重试模式，都会执行）
        print("\n" + "=" * 60)
        if retry_only:
            print("直接执行：重试处理默认错误答案的失败项目")
        else:
            print("第二阶段：自适应重试处理失败项目")
        print("=" * 60)

        # 先检查当前数据集中的默认错误答案情况
        initial_failed_count = self.count_default_wrong_answers()
        print(f"📊 当前数据集中默认错误答案统计: {initial_failed_count} 个")

        if initial_failed_count == 0:
            print("✅ 数据集中没有默认错误答案，无需重试处理")
        else:
            self.process_failed_items_with_adaptive_config(output_file, self.max_workers)

        # 第三阶段：处理缺失wrong_answer字段的项目
        print("\n" + "=" * 60)
        print("第三阶段：处理缺失wrong_answer字段的项目")
        print("=" * 60)

        # 先检查当前数据集中的缺失字段情况
        missing_count = self.count_missing_wrong_answers()
        print(f"📊 当前数据集中缺失wrong_answer字段统计: {missing_count} 个")

        if missing_count == 0:
            print("✅ 数据集中没有缺失wrong_answer字段的项目，无需处理")
        else:
            self.process_missing_items_with_adaptive_config(output_file, self.max_workers)

        # 保存最终结果
        self.save_final_results(output_file, "最终处理完成")

        # 清理临时文件
        self.cleanup_temp_files(output_file)

    def save_final_results(self, output_file: str, stage: str = "处理完成"):
        """
        保存最终结果到输出文件

        🔥 新增方法：专门用于保存最终结果

        Args:
            output_file: 输出文件路径
            stage: 处理阶段描述
        """
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(self.dataset, f, ensure_ascii=False, indent=2)
            print(f"✅ {stage}！结果已保存到: {output_file}")

            # 最终统计
            default_count = self.count_default_wrong_answers()
            missing_count = self.count_missing_wrong_answers()
            print(f"🏁 最终统计:")
            print(f"   - 剩余默认错误答案: {default_count} 个")
            print(f"   - 剩余缺失wrong_answer字段: {missing_count} 个")

        except Exception as e:
            print(f"❌ 保存最终输出文件失败: {e}")

    def cleanup_temp_files(self, output_file: str):
        """
        清理临时文件

        🔥 新增方法：专门用于清理临时文件

        Args:
            output_file: 输出文件路径
        """
        try:
            # 删除重试阶段的临时文件
            for i in range(1, 11):
                temp_file = f"{output_file}.retry_round_{i}.tmp"
                if os.path.exists(temp_file):
                    os.remove(temp_file)
                    print(f"🗑️ 已删除临时文件: {temp_file}")

            # 删除缺失字段处理的临时文件
            temp_file = f"{output_file}.missing_fields_processed.tmp"
            if os.path.exists(temp_file):
                os.remove(temp_file)
                print(f"🗑️ 已删除临时文件: {temp_file}")

        except Exception as e:
            print(f"⚠️ 清理临时文件时出现错误: {e}")


def main():
    """主函数"""
    print("🚀 优化版JSON问题错误答案生成器")
    print("=" * 60)

    # 配置参数
    API_KEY = "05748176082447f483438dfd914cc299.NcsCifhTarCch6es"  # 请填写您的API密钥
    INPUT_FILE = "/home/jiangjp/trace-idea/data/2wikimultihopqa/wiki_test1000_optimized_final.json"  # 输入文件路径
    OUTPUT_FILE = "/home/jiangjp/trace-idea/data/2wikimultihopqa/wiki_test1000_add_wronganswer.json"  # 输出文件路径

    # 并行处理参数
    MAX_WORKERS = 3000  # 并发线程数，根据API限制调整

    # ⭐ 控制参数：选择执行模式
    RETRY_ONLY = False  # 设置为True表示仅处理默认错误答案的失败项目
    MISSING_FIELDS_ONLY = False  # 设置为True表示仅处理缺失wrong_answer字段的项目

    # 注意：如果MISSING_FIELDS_ONLY=True，则RETRY_ONLY的值会被忽略
    # 三种模式：
    # 1. MISSING_FIELDS_ONLY=True: 仅处理缺失wrong_answer字段
    # 2. RETRY_ONLY=True, MISSING_FIELDS_ONLY=False: 仅处理默认错误答案的项目
    # 3. 两者都为False: 执行完整流程

    # 检查API密钥
    if not API_KEY:
        print("❌ 错误：请先设置API密钥")
        print("💡 请在代码中的 API_KEY 变量中填写您的ZhipuAI API密钥")
        return

    print(f"📁 输入文件: {INPUT_FILE}")
    print(f"📁 输出文件: {OUTPUT_FILE}")
    print(f"🔧 并发数: {MAX_WORKERS}")

    if MISSING_FIELDS_ONLY:
        print("🔍 启用仅缺失字段处理模式")
        print(f"📂 将从文件 {INPUT_FILE} 中读取数据，仅处理缺失wrong_answer字段的项目")
    elif RETRY_ONLY:
        print("🔄 启用仅重试模式")
        print(f"📂 将从文件 {INPUT_FILE} 中读取数据，仅处理默认错误答案的项目")
    else:
        print("🚀 启用完整处理模式")
        print(f"📂 将完整处理文件 {INPUT_FILE} 中的所有数据")

    print("-" * 60)

    # 创建生成器实例
    generator = OptimizedWrongAnswerGenerator(API_KEY, max_workers=MAX_WORKERS)

    # 开始处理
    generator.process_dataset_optimized(
        INPUT_FILE,
        OUTPUT_FILE,
        retry_only=RETRY_ONLY,
        missing_fields_only=MISSING_FIELDS_ONLY
    )


if __name__ == "__main__":
    main()