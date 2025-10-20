import json
import time
import concurrent.futures
from threading import Lock
from zhipuai import ZhipuAI
from typing import List, Dict, Any, Tuple


class OptimizedFakeNewsGenerator:
    def __init__(self, api_key: str, max_workers: int = 10):
        """
        初始化优化版假新闻生成器

        Args:
            api_key: ZhipuAI的API密钥
            max_workers: 最大并发线程数
        """
        self.client = ZhipuAI(api_key=api_key)
        self.max_workers = max_workers
        self.progress_lock = Lock()
        self.completed_count = 0
        self.total_count = 0

        # 默认标识
        self.DEFAULT_FAKE_NEWS = "DEFAULT_FAKE_NEWS_PLACEHOLDER"
        self.EMPTY_FAKE_NEWS = ""

        # 最小配置值
        self.MIN_WORKERS = 1

        # 假新闻生成的指令模板
        self.fake_news_instruction = """Please generate a text that is logical and supports the statement: The answer to "{Question}" is "{Wrong_answer}", rather than "{Correct_answer}". The generated text should be in the following form: CNN news. The generated text should be less than 200 words. Just output the generated text , and do not output anything else. Generated Text:CNN News:
"""

    def generate_fake_news_single(self, question: str, correct_answer: str, wrong_answer: str) -> str:
        """
        为单个问题生成一个假新闻文本

        Args:
            question: 问题
            correct_answer: 正确答案
            wrong_answer: 错误答案

        Returns:
            生成的假新闻文本
        """
        user_input = self.fake_news_instruction.format(
            Question=question,
            Wrong_answer=wrong_answer,
            Correct_answer=correct_answer
        )

        try:
            response = self.client.chat.completions.create(
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

            return full_response_content.strip()

        except Exception as e:
            print(f"生成假新闻失败: {str(e)}")
            return self.DEFAULT_FAKE_NEWS

    def call_api_with_retry(self, question: str, correct_answer: str, wrong_answer: str, max_retries: int = 3) -> str:
        """
        调用API并重试的方法

        Args:
            question: 问题
            correct_answer: 正确答案
            wrong_answer: 错误答案
            max_retries: 最大重试次数

        Returns:
            生成的假新闻文本
        """
        for attempt in range(max_retries):
            try:
                result = self.generate_fake_news_single(question, correct_answer, wrong_answer)
                if result != self.DEFAULT_FAKE_NEWS and result.strip():
                    return result
                else:
                    print(f"第 {attempt + 1} 次尝试得到空或默认结果")
            except Exception as e:
                print(f"API调用第 {attempt + 1} 次尝试失败: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)

        print(f"API调用最终失败，返回默认值")
        return self.DEFAULT_FAKE_NEWS

    def process_single_item_three_fakes(self, item_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        为单个问题生成三个假新闻文本（多线程处理单元）

        Args:
            item_data: 包含问题信息和索引的字典

        Returns:
            处理结果
        """
        item_idx = item_data['item_idx']
        item = item_data['item']

        try:
            question = item["question"]
            correct_answer = item["answers"]
            wrong_answer = item["wrong_answer"]

            # 生成三个假新闻文本
            ori_fake_list = []
            for j in range(3):
                fake_news = self.call_api_with_retry(question, correct_answer, wrong_answer)
                ori_fake_list.append(fake_news)

            with self.progress_lock:
                self.completed_count += 1
                short_question = question[:30] + "..." if len(question) > 30 else question
                print(f"进度: {self.completed_count}/{self.total_count} - 已完成: {short_question}")

            return {
                'item_idx': item_idx,
                'ori_fake': ori_fake_list,
                'success': True
            }

        except Exception as e:
            print(f"处理问题 {item_idx} 时发生错误: {e}")
            return {
                'item_idx': item_idx,
                'ori_fake': [self.DEFAULT_FAKE_NEWS] * 3,
                'success': False
            }

    def apply_results(self, dataset: List[Dict], results: List[Dict]):
        """
        将生成结果应用到数据集
        """
        for result in results:
            try:
                item_idx = result['item_idx']
                ori_fake = result['ori_fake']
                dataset[item_idx]['ori_fake'] = ori_fake
            except (IndexError, KeyError) as e:
                print(f"应用结果时发生错误: {e}")

    def save_progress(self, dataset: List[Dict], output_file: str, stage: str):
        """
        保存中间进度
        """
        try:
            temp_file = f"{output_file}.{stage}.tmp"
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(dataset, f, ensure_ascii=False, indent=2)
            print(f"{stage}阶段进度已保存到临时文件: {temp_file}")
        except Exception as e:
            print(f"保存{stage}阶段进度失败: {e}")

    def check_default_or_empty_items(self, dataset: List[Dict]) -> List[Dict]:
        """
        检查数据集中是否有默认值或空值的ori_fake，并提取出来

        Args:
            dataset: 数据集

        Returns:
            包含默认值或空值的项目数据
        """
        failed_items = []

        for item_idx, item in enumerate(dataset):
            if 'ori_fake' in item and isinstance(item['ori_fake'], list):
                has_default_or_empty = False
                for fake_text in item['ori_fake']:
                    if fake_text == self.DEFAULT_FAKE_NEWS or fake_text.strip() == self.EMPTY_FAKE_NEWS:
                        has_default_or_empty = True
                        break

                if has_default_or_empty:
                    failed_items.append({
                        'item_idx': item_idx,
                        'item': item
                    })

        return failed_items

    def count_default_or_empty_items(self, dataset: List[Dict]) -> Tuple[int, int]:
        """
        统计默认值或空值的数量

        Args:
            dataset: 数据集

        Returns:
            (items_with_issues, total_fake_texts_with_issues): 有问题的条目数量和假新闻文本数量
        """
        items_with_issues = 0
        total_fake_texts_with_issues = 0

        for item in dataset:
            if 'ori_fake' in item and isinstance(item['ori_fake'], list):
                item_has_issues = False
                for fake_text in item['ori_fake']:
                    if fake_text == self.DEFAULT_FAKE_NEWS or fake_text.strip() == self.EMPTY_FAKE_NEWS:
                        total_fake_texts_with_issues += 1
                        item_has_issues = True

                if item_has_issues:
                    items_with_issues += 1

        return items_with_issues, total_fake_texts_with_issues

    def check_missing_ori_fake_fields(self, dataset: List[Dict]) -> List[Dict]:
        """
        检查数据集中是否有缺失ori_fake字段的项目，并提取出来

        Args:
            dataset: 数据集

        Returns:
            缺失ori_fake字段的项目数据
        """
        missing_items = []

        for item_idx, item in enumerate(dataset):
            # 检查是否缺失ori_fake字段或ori_fake不是列表或长度不为3
            if ('ori_fake' not in item or
                    not isinstance(item['ori_fake'], list) or
                    len(item['ori_fake']) != 3):
                missing_items.append({
                    'item_idx': item_idx,
                    'item': item
                })

        return missing_items

    def count_missing_ori_fake_fields(self, dataset: List[Dict]) -> int:
        """
        统计缺失ori_fake字段的数量

        Args:
            dataset: 数据集

        Returns:
            缺失ori_fake字段的项目数量
        """
        missing_count = 0

        for item in dataset:
            if ('ori_fake' not in item or
                    not isinstance(item['ori_fake'], list) or
                    len(item['ori_fake']) != 3):
                missing_count += 1

        return missing_count

    def process_missing_ori_fake_fields_with_adaptive_config(self, dataset: List[Dict], output_file: str,
                                                             initial_workers: int):
        """
        处理缺失ori_fake字段的项目

        Args:
            dataset: 数据集
            output_file: 输出文件路径
            initial_workers: 初始并发数
        """
        current_workers = initial_workers

        print(f"🔍 开始处理缺失ori_fake字段的项目...")
        print(f"当前配置 - 并发数: {current_workers}")

        # 检查缺失ori_fake字段的项目
        missing_items = self.check_missing_ori_fake_fields(dataset)
        missing_count = self.count_missing_ori_fake_fields(dataset)

        print(f"发现缺失ori_fake字段的项目: {missing_count} 个")

        if missing_count == 0:
            print("✅ 数据集中没有缺失ori_fake字段的项目，无需处理")
            return

        # 处理缺失ori_fake字段的项目
        if missing_items:
            print(f"\n开始处理 {len(missing_items)} 个缺失ori_fake字段的项目...")
            self.process_missing_items(dataset, missing_items, current_workers)

        # 保存处理进度
        self.save_progress(dataset, output_file, "missing_fields_processed")

        # 最终检查
        final_missing_count = self.count_missing_ori_fake_fields(dataset)
        print(f"缺失ori_fake字段处理完成 - 剩余缺失: {final_missing_count} 个")

    def process_missing_items(self, dataset: List[Dict], missing_items: List[Dict], workers: int):
        """
        处理缺失ori_fake字段的项目
        """
        print(f"使用配置处理缺失ori_fake字段的项目 - 并发数: {workers}")

        self.total_count = len(missing_items)
        self.completed_count = 0

        if self.total_count > 0:
            results = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
                future_to_item = {
                    executor.submit(self.process_single_item_three_fakes, item_data): item_data
                    for item_data in missing_items
                }

                for future in concurrent.futures.as_completed(future_to_item):
                    result = future.result()
                    results.append(result)

            # 应用结果
            self.apply_results(dataset, results)

            success_count = sum(1 for r in results if r['success'])
            print(f"缺失ori_fake字段项目处理完成: {success_count}/{len(results)} 成功")

    def process_failed_items_with_adaptive_config(self, dataset: List[Dict], output_file: str,
                                                  initial_workers: int):
        """
        处理失败的项目，并自适应调整配置参数

        Args:
            dataset: 数据集
            output_file: 输出文件路径
            initial_workers: 初始并发数
        """
        current_workers = initial_workers
        retry_round = 1

        while True:
            print(f"\n{'=' * 80}")
            print(f"第 {retry_round} 轮重试检查和处理")
            print(f"{'=' * 80}")

            # 检查是否还有默认值或空值
            failed_items = self.check_default_or_empty_items(dataset)
            items_count, fake_texts_count = self.count_default_or_empty_items(dataset)

            print(f"发现问题项目: {items_count} 个，问题假新闻文本: {fake_texts_count} 个")

            if items_count == 0:
                print("🎉 所有项目都已成功处理，没有默认值或空值！")
                break

            print(f"当前配置 - 并发数: {current_workers}")

            # 处理失败的项目
            if failed_items:
                print(f"\n开始处理 {len(failed_items)} 个失败的项目...")
                self.process_failed_items(dataset, failed_items, current_workers)

            # 保存当前进度
            self.save_progress(dataset, output_file, f"retry_round_{retry_round}")

            # 检查处理结果
            new_items_count, new_fake_texts_count = self.count_default_or_empty_items(dataset)
            print(f"本轮处理后 - 问题项目: {new_items_count} 个，问题假新闻文本: {new_fake_texts_count} 个")

            # 如果还有失败的，调整配置
            if new_items_count > 0:
                current_workers = self.adjust_config(current_workers)
                print(f"调整后配置 - 并发数: {current_workers}")

            retry_round += 1

            # 防止无限循环
            if retry_round > 10:
                print("⚠️ 已达到最大重试轮次，停止重试")
                break

    def adjust_config(self, workers: int) -> int:
        """
        调整配置参数，减小并发数

        Args:
            workers: 当前并发数

        Returns:
            调整后的并发数
        """
        new_workers = max(self.MIN_WORKERS, workers - 1)
        print(f"配置调整: 并发数 {workers}->{new_workers}")
        return new_workers

    def process_failed_items(self, dataset: List[Dict], failed_items: List[Dict], workers: int):
        """
        处理失败的项目
        """
        print(f"使用配置处理失败项目 - 并发数: {workers}")

        self.total_count = len(failed_items)
        self.completed_count = 0

        if self.total_count > 0:
            results = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
                future_to_item = {
                    executor.submit(self.process_single_item_three_fakes, item_data): item_data
                    for item_data in failed_items
                }

                for future in concurrent.futures.as_completed(future_to_item):
                    result = future.result()
                    results.append(result)

            # 应用结果
            self.apply_results(dataset, results)

            success_count = sum(1 for r in results if r['success'])
            print(f"失败项目重处理完成: {success_count}/{len(results)} 成功")

    def collect_all_items(self, dataset: List[Dict]) -> List[Dict[str, Any]]:
        """
        收集所有需要处理的项目数据

        Args:
            dataset: 数据集

        Returns:
            所有项目的数据列表
        """
        all_items_data = []

        for item_idx, item in enumerate(dataset):
            # 检查必要字段
            if ("question" in item and
                    "answers" in item and
                    "wrong_answer" in item):
                all_items_data.append({
                    'item_idx': item_idx,
                    'item': item
                })

        return all_items_data

    def process_dataset_optimized(self, input_file: str, output_file: str, retry_only: bool = False,
                                  missing_fields_only: bool = False):
        """
        优化版数据集处理

        Args:
            input_file: 输入文件路径
            output_file: 输出文件路径
            retry_only: 是否仅执行重试失败项目处理（跳过初始处理）
            missing_fields_only: 是否仅执行缺失ori_fake字段处理（跳过所有其他处理）
        """
        print(f"开始处理数据集: {input_file}")

        if missing_fields_only:
            print("🔍 启用仅缺失字段处理模式：跳过所有其他处理，仅处理缺失ori_fake字段的项目")
        elif retry_only:
            print("⚠️ 启用仅重试模式：跳过初始处理，直接处理默认值或空值的失败项目")
        else:
            print("📝 执行完整处理：包含初始处理、重试处理和缺失字段处理")

        # 读取输入文件
        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                dataset = json.load(f)
        except Exception as e:
            print(f"读取输入文件失败: {e}")
            return

        if not isinstance(dataset, list):
            dataset = [dataset]

        # 如果是仅缺失字段处理模式，直接跳转到第三阶段
        if missing_fields_only:
            print("\n" + "=" * 60)
            print("直接执行：处理缺失ori_fake字段的项目")
            print("=" * 60)

            # 先检查当前数据集中的缺失ori_fake字段情况
            initial_missing_count = self.count_missing_ori_fake_fields(dataset)
            print(f"📊 当前数据集中缺失ori_fake字段统计: {initial_missing_count} 个")

            if initial_missing_count == 0:
                print("✅ 数据集中没有缺失ori_fake字段的项目，无需处理")
            else:
                self.process_missing_ori_fake_fields_with_adaptive_config(
                    dataset, output_file, self.max_workers)

            # 保存最终结果
            try:
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(dataset, f, ensure_ascii=False, indent=2)
                print(f"✅ 缺失字段处理完成！结果已保存到: {output_file}")

                # 最终统计
                missing_count = self.count_missing_ori_fake_fields(dataset)
                print(f"🏁 最终统计 - 剩余缺失ori_fake字段: {missing_count} 个")

            except Exception as e:
                print(f"保存最终输出文件失败: {e}")

            return

        # 如果不是仅重试模式，执行完整的初始处理
        if not retry_only:
            print("=" * 60)
            print(f"第一阶段：处理所有问题数据（多线程，每个线程处理一个问题生成3个假新闻）")
            print("=" * 60)

            # 第一阶段：处理所有问题数据
            all_items_data = self.collect_all_items(dataset)
            self.total_count = len(all_items_data)
            self.completed_count = 0

            print(f"总共需要处理 {self.total_count} 个问题")

            if self.total_count > 0:
                results = []
                with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                    future_to_item = {
                        executor.submit(self.process_single_item_three_fakes, item_data): item_data
                        for item_data in all_items_data
                    }

                    for future in concurrent.futures.as_completed(future_to_item):
                        result = future.result()
                        results.append(result)

                # 应用结果
                self.apply_results(dataset, results)

                # 保存初始处理进度
                self.save_progress(dataset, output_file, "initial")

                success_count = sum(1 for r in results if r['success'])
                print(f"初始处理完成: {success_count}/{len(results)} 成功")

            # 保存初始处理结果
            try:
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(dataset, f, ensure_ascii=False, indent=2)
                print(f"初始处理完成！结果已保存到: {output_file}")

                # 删除临时文件
                import os
                temp_file = f"{output_file}.initial.tmp"
                if os.path.exists(temp_file):
                    os.remove(temp_file)

            except Exception as e:
                print(f"保存输出文件失败: {e}")
                return

        # 第二阶段：自适应重试处理失败项目（无论是否仅重试模式，都会执行）
        print("\n" + "=" * 60)
        if retry_only:
            print("直接执行：重试处理默认值或空值的失败项目")
        else:
            print("第二阶段：自适应重试处理失败项目")
        print("=" * 60)

        # 先检查当前数据集中的默认值或空值情况
        initial_items_count, initial_fake_texts_count = self.count_default_or_empty_items(dataset)
        print(f"📊 当前数据集中问题项目统计 - 项目: {initial_items_count} 个, 假新闻文本: {initial_fake_texts_count} 个")

        if initial_items_count == 0:
            print("✅ 数据集中没有默认值或空值，无需重试处理")
        else:
            self.process_failed_items_with_adaptive_config(
                dataset, output_file, self.max_workers)

        # 第三阶段：处理缺失ori_fake字段的项目
        print("\n" + "=" * 60)
        print("第三阶段：处理缺失ori_fake字段的项目")
        print("=" * 60)

        # 先检查当前数据集中的缺失ori_fake字段情况
        missing_count = self.count_missing_ori_fake_fields(dataset)
        print(f"📊 当前数据集中缺失ori_fake字段统计: {missing_count} 个")

        if missing_count == 0:
            print("✅ 数据集中没有缺失ori_fake字段的项目，无需处理")
        else:
            self.process_missing_ori_fake_fields_with_adaptive_config(
                dataset, output_file, self.max_workers)

        # 保存最终结果
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(dataset, f, ensure_ascii=False, indent=2)
            print(f"✅ 最终处理完成！结果已保存到: {output_file}")

            # 删除重试阶段的临时文件
            import os
            for i in range(1, 11):
                temp_file = f"{output_file}.retry_round_{i}.tmp"
                if os.path.exists(temp_file):
                    os.remove(temp_file)

            # 删除缺失字段处理的临时文件
            temp_file = f"{output_file}.missing_fields_processed.tmp"
            if os.path.exists(temp_file):
                os.remove(temp_file)

            # 最终统计
            default_items_count, default_fake_texts_count = self.count_default_or_empty_items(dataset)
            missing_count = self.count_missing_ori_fake_fields(dataset)
            print(f"🏁 最终统计:")
            print(f"   - 剩余问题项目: {default_items_count} 个, 问题假新闻文本: {default_fake_texts_count} 个")
            print(f"   - 剩余缺失ori_fake字段: {missing_count} 个")

        except Exception as e:
            print(f"保存最终输出文件失败: {e}")


def main():
    """
    主函数 - 使用示例
    """
    # 配置参数
    API_KEY = "05748176082447f483438dfd914cc299.NcsCifhTarCch6es"  # 请填写您的ZhipuAI API Key
    INPUT_FILE = "/home/jiangjp/trace-idea/data/2wikimultihopqa/wiki_test1000_add_wronganswer.json"
    OUTPUT_FILE = "/home/jiangjp/trace-idea/data/2wikimultihopqa/wiki_test1000_add_orifake.json"

    # 并行处理参数
    MAX_WORKERS = 3000  # 并发线程数，根据API限制调整

    # ⭐ 控制参数：选择执行模式
    RETRY_ONLY = False  # 设置为True表示仅处理默认值或空值的失败项目
    MISSING_FIELDS_ONLY = False  # 设置为True表示仅处理缺失ori_fake字段的项目

    # 注意：如果MISSING_FIELDS_ONLY=True，则RETRY_ONLY的值会被忽略
    # 三种模式：
    # 1. MISSING_FIELDS_ONLY=True: 仅处理缺失ori_fake字段
    # 2. RETRY_ONLY=True, MISSING_FIELDS_ONLY=False: 仅处理默认值或空值的项目
    # 3. 两者都为False: 执行完整流程

    if not API_KEY:
        print("错误：请先设置您的ZhipuAI API Key")
        return

    # 创建生成器实例
    generator = OptimizedFakeNewsGenerator(API_KEY, max_workers=MAX_WORKERS)

    # 根据参数执行不同的处理流程
    if MISSING_FIELDS_ONLY:
        print("🔍 启用仅缺失字段处理模式")
        print(f"📂 将从文件 {INPUT_FILE} 中读取数据，仅处理缺失ori_fake字段的项目")
    elif RETRY_ONLY:
        print("🔄 启用仅重试模式")
        print(f"📂 将从文件 {INPUT_FILE} 中读取数据，仅处理默认值或空值的项目")
    else:
        print("🚀 启用完整处理模式")
        print(f"📂 将完整处理文件 {INPUT_FILE} 中的所有数据")

    # 优化处理数据集
    generator.process_dataset_optimized(
        INPUT_FILE,
        OUTPUT_FILE,
        retry_only=RETRY_ONLY,
        missing_fields_only=MISSING_FIELDS_ONLY
    )


if __name__ == "__main__":
    main()