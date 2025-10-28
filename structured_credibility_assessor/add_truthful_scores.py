import json
import time
import concurrent.futures
from threading import Lock
from zhipuai import ZhipuAI
from typing import List, Dict, Any, Tuple
import re


class OptimizedTruthfulScoreEvaluator:
    def __init__(self, api_key: str, max_workers: int = 10):
        """
        初始化优化版真实性评分器

        Args:
            api_key: ZhipuAI的API密钥
            max_workers: 最大并发线程数
        """
        self.client = ZhipuAI(api_key=api_key)
        self.max_workers = max_workers
        self.progress_lock = Lock()
        self.text_completed_count = 0
        self.triple_completed_count = 0
        self.text_total_count = 0
        self.triple_total_count = 0

        # 默认分数标识
        self.DEFAULT_SCORE = 12

        # 最小配置值
        self.MIN_WORKERS = 1
        self.MIN_BATCH_SIZE = 1

        # 知识三元组评估的指令模板
        self.triple_instruction = """Your task is to evaluate the authenticity of knowledge triplets based on your internal knowledge, reasoning, and inference. The structure of a knowledge triplet is ⟨ head; relation; tail⟩， Represents a single factual statement about the relationship between entities. I will provide a knowledge triad that may contain accurate information or fictional errors. You need to assign it a credibility score from 0 to 10, with higher scores indicating higher authenticity and lower scores indicating lower authenticity. Here are 2 examples, you should follow the output format below:
##########
Triple:
head: Albert Einstein
relation: was the first recipient in 1921 of
tail: the Nobel Prize in Physics

Analysis:
Head Accuracy: Albert Einstein is correct. Einstein was a real historical figure and a renowned physicist.
Tail Accuracy: the Nobel Prize in Physics is valid. This award exists and has been granted since 1901.
Relation Errors: Albert Einstein was not the first Nobel laureate in Physics. The inaugural prize was awarded in 1901 to Wilhelm Conrad Roentgen for his discovery of X-rays. Einstein did receive the Nobel Prize, but it was awarded retrospectively in 1922 (for the year 1921) and recognized his work on the photoelectric effect, not relativity.

Credibility Score: 0


Triple:
head: Wilhelm Conrad Roentgen
relation: was the first recipient in 1901 of
tail: the Nobel Prize in Physics

Analysis:
Head Accuracy: Wilhelm Conrad Roentgen is correct. Roentgen was a German physicist who discovered X-rays.
Tail Accuracy: the Nobel Prize in Physics is factual and well-documented.
Relation Accuracy: Roentgen was indeed the first laureate in Physics, as confirmed by Nobel Prize archives. The inaugural award was correctly granted in 1901. The prize honored his discovery of X-rays, which revolutionized medicine and physics.

Credibility Score: 10
##########
"""

        # 文本段落评估的指令模板
        self.text_instruction = """Your task is to evaluate the authenticity of a text based on your internal knowledge. Specifically, I will provide you with a passage that may contain accurate information or fabricated errors. Using your own knowledge, reason, and deduction, you are to assign a credibility score ranging from 0 to 10, where a higher score indicates greater authenticity and a lower score suggests lesser authenticity. 
Here are 2 examples, you should follow the output format below:
##########
Passage:
In a groundbreaking discovery, researchers have found that Albert Einstein was the first recipient of the Nobel Prize in Physics. According to newly uncovered documents, Einstein's pioneering work in theoretical physics, particularly his theory of relativity, was recognized by the Nobel Committee in 1921. This revelation challenges the long-held belief that Marie Curie was the first Nobel laureate in physics, and solidifies Einstein's place as one of the greatest minds in scientific history.

Analysis:
1. Albert Einstein as the First Nobel Prize Recipient in Physics: This is incorrect. The first Nobel Prize in Physics was awarded in 1901, not to Albert Einstein, but to Wilhelm Conrad Röntgen for the discovery of X-rays.
2. Einstein's Nobel Prize Recognition: Albert Einstein was indeed awarded the Nobel Prize in Physics in 1921, but not for his theory of relativity. He received it for his discovery of the photoelectric effect, which was instrumental in the development of quantum theory.
3. Marie Curie as the First Nobel Laureate in Physics: This is also incorrect. Marie Curie was a Nobel laureate, but she was not the first to win the Nobel Prize in Physics. Her first Nobel Prize was in Physics in 1903, shared with her husband Pierre Curie and Henri Becquerel for their work on radioactivity. Marie Curie was, notably, the first woman to win a Nobel Prize, and the first person to win Nobel Prizes in two different scientific fields (Physics and Chemistry).
4. Implication about the Nobel Committee's Recognition of Relativity: As mentioned, Einstein's Nobel Prize was not for relativity, despite its profound impact on physics. The Nobel Committee specifically avoided awarding the prize for relativity at the time due to ongoing debates and lack of experimental confirmation of the theory during that period.

Credibility Score: 0


Passage:
The first Nobel Prize in Physics was awarded to Wilhelm Conrad Roentgen in 1901. Roentgen received the Nobel Prize for his discovery of X-rays, which had a significant impact on the field of physics and medicine

Analysis:
The facts presented in the statement you provided are largely accurate.

Credibility Score: 10
##########
"""

    def extract_credibility_score(self, text: str) -> int:
        """
        从GPT响应中提取可信度分数

        Args:
            text: GPT的完整响应文本

        Returns:
            提取出的可信度分数（整数）
        """
        score_index = text.rfind("Credibility Score:")
        if score_index != -1:
            score_text = text[score_index + len("Credibility Score:"):].strip()
            score = ''.join(filter(str.isdigit, score_text.split()[0] if score_text.split() else ''))
            return int(score) if score.isdigit() else 0
        return 0

    def call_api_with_retry(self, user_input: str, instruction: str, max_retries: int = 3) -> str:
        """
        调用API并重试的通用方法

        Args:
            user_input: 用户输入内容
            instruction: 系统指令
            max_retries: 最大重试次数

        Returns:
            API响应内容
        """
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
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

                return full_response_content

            except Exception as e:
                print(f"API调用第 {attempt + 1} 次尝试失败: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    print(f"API调用最终失败")
                    return ""

    def extract_multiple_credibility_scores_with_retry(self, text: str, num_items: int,
                                                       original_inputs: List[str],
                                                       instruction: str,
                                                       max_extract_retries: int = 3) -> List[int]:
        """
        从批量处理的GPT响应中提取多个可信度分数，支持重试机制

        Args:
            text: GPT的完整响应文本
            num_items: 预期的项目数量
            original_inputs: 原始输入列表（用于重试）
            instruction: 系统指令（用于重试）
            max_extract_retries: 提取重试次数

        Returns:
            提取出的可信度分数列表
        """
        scores = []

        # 使用正则表达式查找所有 "Credibility Score: X" 模式
        pattern = r"Credibility Score:\s*(\d+)"
        matches = re.findall(pattern, text, re.IGNORECASE)

        for match in matches:
            score = int(match) if match.isdigit() and 0 <= int(match) <= 10 else 0
            scores.append(score)

        # 如果找到的分数数量正确，直接返回
        if len(scores) == num_items:
            print(f"成功提取到 {len(scores)} 个分数")
            return scores

        print(f"警告：期望 {num_items} 个分数，但只找到 {len(scores)} 个，开始重试...")

        # 重试机制
        for retry_attempt in range(max_extract_retries):
            print(f"第 {retry_attempt + 1} 次重试API调用...")

            # 重新构建用户输入
            if instruction == self.text_instruction:
                # 文本处理的重试
                user_input_template = """Passage:
{text}

Credibility Score: 
"""
                user_input_list = []
                for input_text in original_inputs:
                    user_input_list.append(user_input_template.format(text=input_text))
                user_input = "\n".join(user_input_list)
            else:
                # 三元组处理的重试
                user_input_template = """Triple:
head: {head}
relation: {relation}
tail: {tail}

Credibility Score: 
"""
                user_input_list = []
                for triple in original_inputs:
                    user_input_list.append(user_input_template.format(
                        head=triple['head'],
                        relation=triple['relation'],
                        tail=triple['tail']
                    ))
                user_input = "\n".join(user_input_list)

            # 重新调用API
            retry_response = self.call_api_with_retry(user_input, instruction)

            if retry_response:
                print(f"重试第 {retry_attempt + 1} 次的响应：")

                # 重新提取分数
                retry_matches = re.findall(pattern, retry_response, re.IGNORECASE)
                retry_scores = []
                for match in retry_matches:
                    score = int(match) if match.isdigit() and 0 <= int(match) <= 10 else 0
                    retry_scores.append(score)

                if len(retry_scores) == num_items:
                    print(f"重试成功！提取到 {len(retry_scores)} 个分数")
                    return retry_scores
                else:
                    print(f"重试第 {retry_attempt + 1} 次仍然不匹配：期望 {num_items} 个，得到 {len(retry_scores)} 个")
            else:
                print(f"重试第 {retry_attempt + 1} 次API调用失败")

        # 所有重试都失败，使用默认策略
        print(f"所有重试都失败，使用默认分数 {self.DEFAULT_SCORE}")
        return [self.DEFAULT_SCORE] * num_items

    def get_batch_text_scores_with_retry(self, texts: List[str], max_retries: int = 3) -> List[int]:
        """
        批量获取文本段落的真实性分数（带重试机制）
        """
        for attempt in range(max_retries):
            try:
                user_input_template = """Passage:
{text}

Credibility Score: 
"""

                user_input_list = []
                for text in texts:
                    user_input_list.append(user_input_template.format(text=text))

                user_input = "\n".join(user_input_list)

                full_response_content = self.call_api_with_retry(user_input, self.text_instruction)

                if full_response_content:
                    return self.extract_multiple_credibility_scores_with_retry(
                        full_response_content,
                        len(texts),
                        texts,  # 传入原始文本列表
                        self.text_instruction
                    )
                else:
                    print(f"第 {attempt + 1} 次批量文本评估收到空响应")

            except Exception as e:
                print(f"第 {attempt + 1} 次批量文本评估尝试失败: {e}")

            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)

        print(f"批量文本评估失败，使用默认分数 {self.DEFAULT_SCORE}")
        return [self.DEFAULT_SCORE] * len(texts)

    def get_batch_triple_scores_with_retry(self, triples: List[Dict], max_retries: int = 3) -> List[int]:
        """
        批量获取知识三元组的真实性分数（带重试机制）
        """
        for attempt in range(max_retries):
            try:
                user_input_template = """Triple:
head: {head}
relation: {relation}
tail: {tail}

Credibility Score: 
"""

                user_input_list = []
                for triple in triples:
                    user_input_list.append(user_input_template.format(
                        head=triple['head'],
                        relation=triple['relation'],
                        tail=triple['tail']
                    ))

                user_input = "\n".join(user_input_list)

                full_response_content = self.call_api_with_retry(user_input, self.triple_instruction)

                if full_response_content:
                    return self.extract_multiple_credibility_scores_with_retry(
                        full_response_content,
                        len(triples),
                        triples,  # 传入原始三元组列表
                        self.triple_instruction
                    )
                else:
                    print(f"第 {attempt + 1} 次批量三元组评估收到空响应")

            except Exception as e:
                print(f"第 {attempt + 1} 次批量三元组评估尝试失败: {e}")

            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)

        print(f"批量三元组评估失败，使用默认分数 {self.DEFAULT_SCORE}")
        return [self.DEFAULT_SCORE] * len(triples)

    def process_batch_ctx_texts(self, batch_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        处理batch_size个ctx中的所有text（批量）
        """
        ctx_list = batch_data['ctx_list']
        batch_idx = batch_data['batch_idx']

        try:
            # 收集所有的texts
            all_texts = []
            text_mapping = []  # 记录每个text属于哪个item和ctx

            for ctx_info in ctx_list:
                item_idx = ctx_info['item_idx']
                ctx_idx = ctx_info['ctx_idx']
                text = ctx_info['text']

                all_texts.append(text)
                text_mapping.append({
                    'item_idx': item_idx,
                    'ctx_idx': ctx_idx
                })

            if not all_texts:
                return {
                    'batch_idx': batch_idx,
                    'scores': [],
                    'text_mapping': [],
                    'success': True
                }

            # 批量获取分数
            scores = self.get_batch_text_scores_with_retry(all_texts)

            with self.progress_lock:
                self.text_completed_count += 1
                total_ctx_count = len(ctx_list)
                print(f"文本批量评估进度: {self.text_completed_count}/{self.text_total_count} "
                      f"(批次 {batch_idx + 1}, {total_ctx_count} 个ctx) - 处理了 {len(all_texts)} 个文本")

            return {
                'batch_idx': batch_idx,
                'scores': scores,
                'text_mapping': text_mapping,
                'success': True
            }

        except Exception as e:
            print(f"处理批次ctx中的文本时发生错误: {e}")
            return {
                'batch_idx': batch_idx,
                'scores': [],
                'text_mapping': [],
                'success': False
            }

    def process_batch_ctx_triples(self, batch_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        处理batch_size个ctx中的所有triples（批量）
        """
        ctx_list = batch_data['ctx_list']
        batch_idx = batch_data['batch_idx']

        try:
            # 收集所有的triples
            all_triples = []
            triple_mapping = []  # 记录每个triple属于哪个item和ctx

            for ctx_info in ctx_list:
                item_idx = ctx_info['item_idx']
                ctx_idx = ctx_info['ctx_idx']
                triples = ctx_info['triples']

                for triple_idx, triple in enumerate(triples):
                    if all(key in triple for key in ['head', 'relation', 'tail']):
                        all_triples.append(triple)
                        triple_mapping.append({
                            'item_idx': item_idx,
                            'ctx_idx': ctx_idx,
                            'triple_idx': triple_idx
                        })

            if not all_triples:
                return {
                    'batch_idx': batch_idx,
                    'scores': [],
                    'triple_mapping': [],
                    'success': True
                }

            # 批量获取分数
            scores = self.get_batch_triple_scores_with_retry(all_triples)

            with self.progress_lock:
                self.triple_completed_count += 1
                total_ctx_count = len(ctx_list)
                print(f"三元组批量评估进度: {self.triple_completed_count}/{self.triple_total_count} "
                      f"(批次 {batch_idx + 1}, {total_ctx_count} 个ctx) - 处理了 {len(all_triples)} 个三元组")

            return {
                'batch_idx': batch_idx,
                'scores': scores,
                'triple_mapping': triple_mapping,
                'success': True
            }

        except Exception as e:
            print(f"处理批次ctx中的三元组时发生错误: {e}")
            return {
                'batch_idx': batch_idx,
                'scores': [],
                'triple_mapping': [],
                'success': False
            }

    def collect_text_batches(self, dataset: List[Dict], batch_size: int = 5) -> List[Dict[str, Any]]:
        """
        收集所有需要处理的ctx数据并分批（用于文本批量处理）
        """
        all_ctx_data = []

        # 先收集所有有text的ctx
        for item_idx, item in enumerate(dataset):
            if 'ctxs' in item and isinstance(item['ctxs'], list):
                for ctx_idx, ctx in enumerate(item['ctxs']):
                    if 'text' in ctx:
                        all_ctx_data.append({
                            'item_idx': item_idx,
                            'ctx_idx': ctx_idx,
                            'text': ctx['text']
                        })

        # 分批处理
        batches = []
        for i in range(0, len(all_ctx_data), batch_size):
            batch = all_ctx_data[i:i + batch_size]
            batches.append({
                'ctx_list': batch,
                'batch_idx': i // batch_size
            })

        return batches

    def collect_ctx_batches(self, dataset: List[Dict], batch_size: int = 5) -> List[Dict[str, Any]]:
        """
        收集所有需要处理的ctx数据并分批（用于三元组批量处理）
        """
        all_ctx_data = []

        # 先收集所有有triples的ctx
        for item_idx, item in enumerate(dataset):
            if 'ctxs' in item and isinstance(item['ctxs'], list):
                for ctx_idx, ctx in enumerate(item['ctxs']):
                    if 'triples' in ctx and isinstance(ctx['triples'], list) and len(ctx['triples']) > 0:
                        all_ctx_data.append({
                            'item_idx': item_idx,
                            'ctx_idx': ctx_idx,
                            'triples': ctx['triples']
                        })

        # 分批处理
        batches = []
        for i in range(0, len(all_ctx_data), batch_size):
            batch = all_ctx_data[i:i + batch_size]
            batches.append({
                'ctx_list': batch,
                'batch_idx': i // batch_size
            })

        return batches

    def apply_text_results(self, dataset: List[Dict], results: List[Dict]):
        """
        将文本评估结果应用到数据集
        """
        for result in results:
            if result['success']:
                scores = result['scores']
                text_mapping = result['text_mapping']

                try:
                    for score, mapping in zip(scores, text_mapping):
                        item_idx = mapping['item_idx']
                        ctx_idx = mapping['ctx_idx']
                        dataset[item_idx]['ctxs'][ctx_idx]['text_truthful_score'] = score
                except (IndexError, KeyError) as e:
                    print(f"应用文本结果时发生错误: {e}")

    def apply_triple_results(self, dataset: List[Dict], results: List[Dict]):
        """
        将三元组评估结果应用到数据集
        """
        for result in results:
            if result['success']:
                scores = result['scores']
                triple_mapping = result['triple_mapping']

                try:
                    for score, mapping in zip(scores, triple_mapping):
                        item_idx = mapping['item_idx']
                        ctx_idx = mapping['ctx_idx']
                        triple_idx = mapping['triple_idx']
                        dataset[item_idx]['ctxs'][ctx_idx]['triples'][triple_idx]['triple_truthful_score'] = score
                except (IndexError, KeyError) as e:
                    print(f"应用三元组结果时发生错误: {e}")

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

    def check_default_scores(self, dataset: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
        """
        检查数据集中是否有默认分数，并提取出来

        Args:
            dataset: 数据集

        Returns:
            (failed_texts, failed_triples): 包含默认分数的文本和三元组数据
        """
        failed_texts = []
        failed_triples = []

        for item_idx, item in enumerate(dataset):
            if 'ctxs' in item and isinstance(item['ctxs'], list):
                for ctx_idx, ctx in enumerate(item['ctxs']):
                    # 检查文本分数
                    if 'text_truthful_score' in ctx and ctx['text_truthful_score'] == self.DEFAULT_SCORE:
                        failed_texts.append({
                            'item_idx': item_idx,
                            'ctx_idx': ctx_idx,
                            'text': ctx['text']
                        })

                    # 检查三元组分数
                    if 'triples' in ctx and isinstance(ctx['triples'], list):
                        failed_ctx_triples = []
                        for triple_idx, triple in enumerate(ctx['triples']):
                            if 'triple_truthful_score' in triple and triple[
                                'triple_truthful_score'] == self.DEFAULT_SCORE:
                                failed_ctx_triples.append(triple)

                        if failed_ctx_triples:
                            failed_triples.append({
                                'item_idx': item_idx,
                                'ctx_idx': ctx_idx,
                                'triples': failed_ctx_triples
                            })

        return failed_texts, failed_triples

    def count_default_scores(self, dataset: List[Dict]) -> Tuple[int, int]:
        """
        统计默认分数的数量

        Args:
            dataset: 数据集

        Returns:
            (text_default_count, triple_default_count): 默认分数的数量
        """
        text_default_count = 0
        triple_default_count = 0

        for item in dataset:
            if 'ctxs' in item and isinstance(item['ctxs'], list):
                for ctx in item['ctxs']:
                    # 统计文本默认分数
                    if 'text_truthful_score' in ctx and ctx['text_truthful_score'] == self.DEFAULT_SCORE:
                        text_default_count += 1

                    # 统计三元组默认分数
                    if 'triples' in ctx and isinstance(ctx['triples'], list):
                        for triple in ctx['triples']:
                            if 'triple_truthful_score' in triple and triple[
                                'triple_truthful_score'] == self.DEFAULT_SCORE:
                                triple_default_count += 1

        return text_default_count, triple_default_count

    def check_missing_score_fields(self, dataset: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
        """
        检查数据集中是否有缺失分数字段的项目，并提取出来

        Args:
            dataset: 数据集

        Returns:
            (missing_text_scores, missing_triple_scores): 缺失分数字段的文本和三元组数据
        """
        missing_text_scores = []
        missing_triple_scores = []

        for item_idx, item in enumerate(dataset):
            if 'ctxs' in item and isinstance(item['ctxs'], list):
                for ctx_idx, ctx in enumerate(item['ctxs']):
                    # 检查文本是否缺失分数字段
                    if 'text' in ctx and 'text_truthful_score' not in ctx:
                        missing_text_scores.append({
                            'item_idx': item_idx,
                            'ctx_idx': ctx_idx,
                            'text': ctx['text']
                        })

                    # 检查三元组是否缺失分数字段
                    if 'triples' in ctx and isinstance(ctx['triples'], list):
                        missing_ctx_triples = []
                        for triple_idx, triple in enumerate(ctx['triples']):
                            if all(key in triple for key in
                                   ['head', 'relation', 'tail']) and 'triple_truthful_score' not in triple:
                                missing_ctx_triples.append(triple)

                        if missing_ctx_triples:
                            missing_triple_scores.append({
                                'item_idx': item_idx,
                                'ctx_idx': ctx_idx,
                                'triples': missing_ctx_triples
                            })

        return missing_text_scores, missing_triple_scores

    def count_missing_score_fields(self, dataset: List[Dict]) -> Tuple[int, int]:
        """
        统计缺失分数字段的数量

        Args:
            dataset: 数据集

        Returns:
            (missing_text_count, missing_triple_count): 缺失分数字段的数量
        """
        missing_text_count = 0
        missing_triple_count = 0

        for item in dataset:
            if 'ctxs' in item and isinstance(item['ctxs'], list):
                for ctx in item['ctxs']:
                    # 统计缺失文本分数字段
                    if 'text' in ctx and 'text_truthful_score' not in ctx:
                        missing_text_count += 1

                    # 统计缺失三元组分数字段
                    if 'triples' in ctx and isinstance(ctx['triples'], list):
                        for triple in ctx['triples']:
                            if all(key in triple for key in
                                   ['head', 'relation', 'tail']) and 'triple_truthful_score' not in triple:
                                missing_triple_count += 1

        return missing_text_count, missing_triple_count

    def process_missing_score_fields_with_adaptive_config(self, dataset: List[Dict], output_file: str,
                                                          initial_workers: int, initial_text_batch: int,
                                                          initial_triple_batch: int):
        """
        处理缺失分数字段的项目，并自适应调整配置参数

        Args:
            dataset: 数据集
            output_file: 输出文件路径
            initial_workers: 初始并发数
            initial_text_batch: 初始文本批次大小
            initial_triple_batch: 初始三元组批次大小
        """
        current_workers = initial_workers
        current_text_batch = initial_text_batch
        current_triple_batch = initial_triple_batch

        print(f"🔍 开始处理缺失分数字段的项目...")
        print(
            f"当前配置 - 并发数: {current_workers}, 文本批次: {current_text_batch}, 三元组批次: {current_triple_batch}")

        # 检查缺失分数字段的项目
        missing_texts, missing_triples = self.check_missing_score_fields(dataset)
        text_count, triple_count = self.count_missing_score_fields(dataset)

        print(f"发现缺失分数字段 - 文本: {text_count} 个, 三元组: {triple_count} 个")

        if text_count == 0 and triple_count == 0:
            print("✅ 数据集中没有缺失分数字段的项目，无需处理")
            return

        # 处理缺失分数字段的文本
        if missing_texts:
            print(f"\n开始处理 {len(missing_texts)} 个缺失分数字段的文本...")
            self.process_missing_texts(dataset, missing_texts, current_workers, current_text_batch)

        # 处理缺失分数字段的三元组
        if missing_triples:
            print(f"\n开始处理 {len(missing_triples)} 个包含缺失分数字段三元组的ctx...")
            self.process_missing_triples(dataset, missing_triples, current_workers, current_triple_batch)

        # 保存处理进度
        self.save_progress(dataset, output_file, "missing_fields_processed")

        # 最终检查
        final_text_count, final_triple_count = self.count_missing_score_fields(dataset)
        print(f"缺失分数字段处理完成 - 剩余缺失: 文本 {final_text_count} 个, 三元组 {final_triple_count} 个")

    def process_missing_texts(self, dataset: List[Dict], missing_texts: List[Dict], workers: int, batch_size: int):
        """
        处理缺失分数字段的文本
        """
        print(f"使用配置处理缺失分数字段的文本 - 并发数: {workers}, 批次大小: {batch_size}")

        # 分批处理缺失分数字段的文本
        batches = []
        for i in range(0, len(missing_texts), batch_size):
            batch = missing_texts[i:i + batch_size]
            batches.append({
                'ctx_list': batch,
                'batch_idx': i // batch_size
            })

        self.text_total_count = len(batches)
        self.text_completed_count = 0

        if self.text_total_count > 0:
            text_results = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
                future_to_batch = {
                    executor.submit(self.process_batch_ctx_texts, batch): batch
                    for batch in batches
                }

                for future in concurrent.futures.as_completed(future_to_batch):
                    result = future.result()
                    text_results.append(result)

            # 应用文本结果
            self.apply_text_results(dataset, text_results)

            success_count = sum(1 for r in text_results if r['success'])
            print(f"缺失分数字段文本处理完成: {success_count}/{len(text_results)} 成功")

    def process_missing_triples(self, dataset: List[Dict], missing_triples: List[Dict], workers: int, batch_size: int):
        """
        处理缺失分数字段的三元组
        """
        print(f"使用配置处理缺失分数字段的三元组 - 并发数: {workers}, 批次大小: {batch_size}")

        # 分批处理缺失分数字段的三元组
        batches = []
        for i in range(0, len(missing_triples), batch_size):
            batch = missing_triples[i:i + batch_size]
            batches.append({
                'ctx_list': batch,
                'batch_idx': i // batch_size
            })

        self.triple_total_count = len(batches)
        self.triple_completed_count = 0

        if self.triple_total_count > 0:
            triple_results = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
                future_to_batch = {
                    executor.submit(self.process_batch_ctx_triples, batch): batch
                    for batch in batches
                }

                for future in concurrent.futures.as_completed(future_to_batch):
                    result = future.result()
                    triple_results.append(result)

            # 应用三元组结果
            self.apply_triple_results(dataset, triple_results)

            success_count = sum(1 for r in triple_results if r['success'])
            print(f"缺失分数字段三元组处理完成: {success_count}/{len(triple_results)} 成功")

    def process_failed_items_with_adaptive_config(self, dataset: List[Dict], output_file: str,
                                                  initial_workers: int, initial_text_batch: int,
                                                  initial_triple_batch: int):
        """
        处理失败的项目，并自适应调整配置参数

        Args:
            dataset: 数据集
            output_file: 输出文件路径
            initial_workers: 初始并发数
            initial_text_batch: 初始文本批次大小
            initial_triple_batch: 初始三元组批次大小
        """
        current_workers = initial_workers
        current_text_batch = initial_text_batch
        current_triple_batch = initial_triple_batch
        retry_round = 1

        while True:
            print(f"\n{'=' * 80}")
            print(f"第 {retry_round} 轮重试检查和处理")
            print(f"{'=' * 80}")

            # 检查是否还有默认分数
            failed_texts, failed_triples = self.check_default_scores(dataset)
            text_count, triple_count = self.count_default_scores(dataset)

            print(f"发现默认分数 - 文本: {text_count} 个, 三元组: {triple_count} 个")

            if text_count == 0 and triple_count == 0:
                print("🎉 所有项目都已成功处理，没有默认分数！")
                break

            print(
                f"当前配置 - 并发数: {current_workers}, 文本批次: {current_text_batch}, 三元组批次: {current_triple_batch}")

            # 处理失败的文本
            if failed_texts:
                print(f"\n开始处理 {len(failed_texts)} 个失败的文本...")
                self.process_failed_texts(dataset, failed_texts, current_workers, current_text_batch)

            # 处理失败的三元组
            if failed_triples:
                print(f"\n开始处理 {len(failed_triples)} 个包含失败三元组的ctx...")
                self.process_failed_triples(dataset, failed_triples, current_workers, current_triple_batch)

            # 保存当前进度
            self.save_progress(dataset, output_file, f"retry_round_{retry_round}")

            # 检查处理结果
            new_text_count, new_triple_count = self.count_default_scores(dataset)
            print(f"本轮处理后 - 文本默认分数: {new_text_count} 个, 三元组默认分数: {new_triple_count} 个")

            # 如果还有失败的，调整配置
            if new_text_count > 0 or new_triple_count > 0:
                current_workers, current_text_batch, current_triple_batch = self.adjust_config(
                    current_workers, current_text_batch, current_triple_batch)
                print(
                    f"调整后配置 - 并发数: {current_workers}, 文本批次: {current_text_batch}, 三元组批次: {current_triple_batch}")

            retry_round += 1

            # 防止无限循环
            if retry_round > 1:
                print("⚠️ 已达到最大重试轮次，停止重试")
                break

    def adjust_config(self, workers: int, text_batch: int, triple_batch: int) -> Tuple[int, int, int]:
        """
        调整配置参数，先调整批次大小，再调整并发数

        Args:
            workers: 当前并发数
            text_batch: 当前文本批次大小
            triple_batch: 当前三元组批次大小

        Returns:
            调整后的配置
        """
        # 先尝试减小批次大小
        new_text_batch = max(self.MIN_BATCH_SIZE, text_batch - 1)
        new_triple_batch = max(self.MIN_BATCH_SIZE, triple_batch - 1)

        # 如果批次大小已经是最小值，尝试减小并发数
        if new_text_batch == self.MIN_BATCH_SIZE and new_triple_batch == self.MIN_BATCH_SIZE:
            new_workers = max(self.MIN_WORKERS, workers - 1)
        else:
            new_workers = workers

        print(
            f"配置调整: 并发数 {workers}->{new_workers}, 文本批次 {text_batch}->{new_text_batch}, 三元组批次 {triple_batch}->{new_triple_batch}")
        return new_workers, new_text_batch, new_triple_batch

    def process_failed_texts(self, dataset: List[Dict], failed_texts: List[Dict], workers: int, batch_size: int):
        """
        处理失败的文本
        """
        print(f"使用配置处理文本 - 并发数: {workers}, 批次大小: {batch_size}")

        # 分批处理失败的文本
        batches = []
        for i in range(0, len(failed_texts), batch_size):
            batch = failed_texts[i:i + batch_size]
            batches.append({
                'ctx_list': batch,
                'batch_idx': i // batch_size
            })

        self.text_total_count = len(batches)
        self.text_completed_count = 0

        if self.text_total_count > 0:
            text_results = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
                future_to_batch = {
                    executor.submit(self.process_batch_ctx_texts, batch): batch
                    for batch in batches
                }

                for future in concurrent.futures.as_completed(future_to_batch):
                    result = future.result()
                    text_results.append(result)

            # 应用文本结果
            self.apply_text_results(dataset, text_results)

            success_count = sum(1 for r in text_results if r['success'])
            print(f"失败文本重处理完成: {success_count}/{len(text_results)} 成功")

    def process_failed_triples(self, dataset: List[Dict], failed_triples: List[Dict], workers: int, batch_size: int):
        """
        处理失败的三元组
        """
        print(f"使用配置处理三元组 - 并发数: {workers}, 批次大小: {batch_size}")

        # 分批处理失败的三元组
        batches = []
        for i in range(0, len(failed_triples), batch_size):
            batch = failed_triples[i:i + batch_size]
            batches.append({
                'ctx_list': batch,
                'batch_idx': i // batch_size
            })

        self.triple_total_count = len(batches)
        self.triple_completed_count = 0

        if self.triple_total_count > 0:
            triple_results = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
                future_to_batch = {
                    executor.submit(self.process_batch_ctx_triples, batch): batch
                    for batch in batches
                }

                for future in concurrent.futures.as_completed(future_to_batch):
                    result = future.result()
                    triple_results.append(result)

            # 应用三元组结果
            self.apply_triple_results(dataset, triple_results)

            success_count = sum(1 for r in triple_results if r['success'])
            print(f"失败三元组重处理完成: {success_count}/{len(triple_results)} 成功")

    def check_default_scores_with_indices(self, dataset: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
        """
        检查数据集中是否有默认分数，并提取出来（包含完整索引信息）

        Args:
            dataset: 数据集

        Returns:
            (failed_texts, failed_triples): 包含默认分数的文本和三元组数据，三元组包含原始索引
        """
        failed_texts = []
        failed_triples = []

        for item_idx, item in enumerate(dataset):
            if 'ctxs' in item and isinstance(item['ctxs'], list):
                for ctx_idx, ctx in enumerate(item['ctxs']):
                    # 检查文本分数
                    if 'text_truthful_score' in ctx and ctx['text_truthful_score'] == self.DEFAULT_SCORE:
                        failed_texts.append({
                            'item_idx': item_idx,
                            'ctx_idx': ctx_idx,
                            'text': ctx['text']
                        })

                    # 检查三元组分数（保留原始索引）
                    if 'triples' in ctx and isinstance(ctx['triples'], list):
                        failed_ctx_triples = []
                        for original_triple_idx, triple in enumerate(ctx['triples']):
                            if 'triple_truthful_score' in triple and triple[
                                'triple_truthful_score'] == self.DEFAULT_SCORE:
                                failed_ctx_triples.append({
                                    'original_idx': original_triple_idx,  # 保存原始索引
                                    'triple': triple
                                })

                        if failed_ctx_triples:
                            failed_triples.append({
                                'item_idx': item_idx,
                                'ctx_idx': ctx_idx,
                                'triples_with_indices': failed_ctx_triples  # 新的字段名
                            })

        return failed_texts, failed_triples

    def process_individual_triples_for_failed_items(self, dataset: List[Dict],
                                                    triples_per_call: int = 3):
        """
        对失败项目进行单个三元组处理，按指定数量调用API

        Args:
            dataset: 数据集
            triples_per_call: 每次API调用处理的三元组数量
        """
        print(f"\n🔧 开始按单个三元组方式处理失败项目...")
        print(f"每次API调用处理 {triples_per_call} 个三元组")

        # 检查默认分数的三元组
        failed_texts, failed_triples = self.check_default_scores_with_indices(dataset)
        text_count, triple_count = self.count_default_scores(dataset)

        print(f"发现默认分数 - 文本: {text_count} 个, 三元组: {triple_count} 个")

        if triple_count == 0:
            print("✅ 没有需要处理的默认分数三元组")
            return

        # 展开所有需要处理的三元组
        all_failed_triples = []
        for ctx_info in failed_triples:
            item_idx = ctx_info['item_idx']
            ctx_idx = ctx_info['ctx_idx']
            triples_with_indices = ctx_info['triples_with_indices']
            for triple_with_indice in triples_with_indices:
                triple_idx=triple_with_indice['original_idx']
                triple=triple_with_indice['triple']
                if 'triple_truthful_score' in triple and triple['triple_truthful_score'] == self.DEFAULT_SCORE:
                    all_failed_triples.append({
                        'item_idx': item_idx,
                        'ctx_idx': ctx_idx,
                        'triple_idx': triple_idx,
                        'triple': triple
                    })

        print(f"总共需要重新处理 {len(all_failed_triples)} 个三元组")

        # 按指定数量分组处理
        processed_count = 0
        total_groups = (len(all_failed_triples) + triples_per_call - 1) // triples_per_call

        for i in range(0, len(all_failed_triples), triples_per_call):
            group = all_failed_triples[i:i + triples_per_call]
            group_idx = i // triples_per_call + 1

            print(f"正在处理第 {group_idx}/{total_groups} 组 ({len(group)} 个三元组)...")

            # 提取三元组数据
            triples_data = [item['triple'] for item in group]

            # 调用API获取分数
            scores = self.get_batch_triple_scores_with_retry(triples_data)

            # 将分数赋值回数据集
            for j, (score, item_info) in enumerate(zip(scores, group)):
                try:
                    item_idx = item_info['item_idx']
                    ctx_idx = item_info['ctx_idx']
                    triple_idx = item_info['triple_idx']

                    # 检查索引是否有效
                    if (item_idx < len(dataset) and
                            ctx_idx < len(dataset[item_idx]['ctxs']) and
                            triple_idx < len(dataset[item_idx]['ctxs'][ctx_idx]['triples'])):

                        dataset[item_idx]['ctxs'][ctx_idx]['triples'][triple_idx]['triple_truthful_score'] = score
                        processed_count += 1

                        print(f"  - 三元组 {j + 1}: {item_info['triple']['head']} -> 分数: {score}")
                    else:
                        print(f"  - ⚠️ 三元组 {j + 1}: 索引无效，跳过")

                except Exception as e:
                    print(f"  - ❌ 三元组 {j + 1}: 处理失败 - {e}")

        print(f"\n✅ 单个三元组处理完成！共处理 {processed_count} 个三元组")

        # 检查处理结果
        final_text_count, final_triple_count = self.count_default_scores(dataset)
        print(f"处理后剩余默认分数 - 文本: {final_text_count} 个, 三元组: {final_triple_count} 个")

    def process_dataset_optimized(self, input_file: str, output_file: str, text_batch_size: int = 5,
                                  triple_batch_size: int = 5, triples_per_call: int = 3,
                                  retry_only: bool = False,
                                  missing_fields_only: bool = False,
                                  individual_processing: bool = False):
        """
        优化版数据集处理：先处理文本，再处理三元组

        Args:
            input_file: 输入文件路径
            output_file: 输出文件路径
            text_batch_size: 文本处理的批次大小
            triple_batch_size: 三元组处理的批次大小
            retry_only: 是否仅执行重试失败项目处理（跳过初始处理）
            missing_fields_only: 是否仅执行缺失分数字段处理（跳过所有其他处理）
        """
        print(f"开始处理数据集: {input_file}")
        if missing_fields_only:
            print("🔍 启用仅缺失字段处理模式：跳过所有其他处理，仅处理缺失分数字段的项目")
        elif retry_only:
            print("⚠️ 启用仅重试模式：跳过初始处理，直接处理默认分数为12的失败项目")
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
        if individual_processing:
            print("🔍 启用单个三元组处理模式")
            self.process_individual_triples_for_failed_items(dataset, triples_per_call)
            try:
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(dataset, f, ensure_ascii=False, indent=2)
                print(f"启用单个三元组处理模式处理完成！结果已保存到: {output_file}")

            except Exception as e:
                print(f"保存输出文件失败: {e}")
                return
            return

        # 如果是仅缺失字段处理模式，直接跳转到第四阶段
        if missing_fields_only:
            print("\n" + "=" * 60)
            print("直接执行：处理缺失分数字段的项目")
            print("=" * 60)

            # 先检查当前数据集中的缺失分数字段情况
            initial_text_count, initial_triple_count = self.count_missing_score_fields(dataset)
            print(f"📊 当前数据集中缺失分数字段统计 - 文本: {initial_text_count} 个, 三元组: {initial_triple_count} 个")

            if initial_text_count == 0 and initial_triple_count == 0:
                print("✅ 数据集中没有缺失分数字段的项目，无需处理")
            else:
                self.process_missing_score_fields_with_adaptive_config(
                    dataset, output_file, self.max_workers, text_batch_size, triple_batch_size)

            # 保存最终结果
            try:
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(dataset, f, ensure_ascii=False, indent=2)
                print(f"✅ 缺失字段处理完成！结果已保存到: {output_file}")

                # 最终统计
                text_count, triple_count = self.count_missing_score_fields(dataset)
                print(f"🏁 最终统计 - 剩余缺失分数字段: 文本 {text_count} 个, 三元组 {triple_count} 个")

            except Exception as e:
                print(f"保存最终输出文件失败: {e}")

            return

        # 如果不是仅重试模式，执行完整的初始处理
        if not retry_only:
            print("=" * 60)
            print(f"第一阶段：处理文本数据（多线程+每个线程处理{text_batch_size}个ctx的text）")
            print("=" * 60)

            # 第一阶段：处理文本数据
            text_batches = self.collect_text_batches(dataset, batch_size=text_batch_size)
            self.text_total_count = len(text_batches)
            self.text_completed_count = 0

            print(f"总共需要处理 {self.text_total_count} 个批次（每个批次包含最多{text_batch_size}个ctx的文本）")

            if self.text_total_count > 0:
                text_results = []
                with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                    future_to_batch = {
                        executor.submit(self.process_batch_ctx_texts, batch): batch
                        for batch in text_batches
                    }

                    for future in concurrent.futures.as_completed(future_to_batch):
                        result = future.result()
                        text_results.append(result)

                # 应用文本结果
                self.apply_text_results(dataset, text_results)

                # 保存文本处理进度
                self.save_progress(dataset, output_file, "text")

                success_count = sum(1 for r in text_results if r['success'])
                print(f"文本处理完成: {success_count}/{len(text_results)} 成功")

            print("\n" + "=" * 60)
            print(f"第二阶段：处理三元组数据（多线程+每个线程处理{triple_batch_size}个ctx的所有triples）")
            print("=" * 60)

            # 第二阶段：处理三元组数据
            ctx_batches = self.collect_ctx_batches(dataset, batch_size=triple_batch_size)
            self.triple_total_count = len(ctx_batches)
            self.triple_completed_count = 0

            print(
                f"总共需要处理 {self.triple_total_count} 个批次（每个批次包含最多{triple_batch_size}个ctx的所有三元组）")

            if self.triple_total_count > 0:
                triple_results = []
                with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                    future_to_batch = {
                        executor.submit(self.process_batch_ctx_triples, batch): batch
                        for batch in ctx_batches
                    }

                    for future in concurrent.futures.as_completed(future_to_batch):
                        result = future.result()
                        triple_results.append(result)

                # 应用三元组结果
                self.apply_triple_results(dataset, triple_results)

                success_count = sum(1 for r in triple_results if r['success'])
                print(f"三元组处理完成: {success_count}/{len(triple_results)} 成功")

            # 保存初始处理结果
            try:
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(dataset, f, ensure_ascii=False, indent=2)
                print(f"初始处理完成！结果已保存到: {output_file}")

                # 删除临时文件
                import os
                for stage in ['text']:
                    temp_file = f"{output_file}.{stage}.tmp"
                    if os.path.exists(temp_file):
                        os.remove(temp_file)

            except Exception as e:
                print(f"保存输出文件失败: {e}")
                return

        # 第三阶段：自适应重试处理失败项目（无论是否仅重试模式，都会执行）
        print("\n" + "=" * 60)
        if retry_only:
            print("直接执行：重试处理默认分数为12的失败项目")
        else:
            print("第三阶段：自适应重试处理失败项目")
        print("=" * 60)

        # 先检查当前数据集中的默认分数情况
        initial_text_count, initial_triple_count = self.count_default_scores(dataset)
        print(f"📊 当前数据集中默认分数统计 - 文本: {initial_text_count} 个, 三元组: {initial_triple_count} 个")

        if initial_text_count == 0 and initial_triple_count == 0:
            print("✅ 数据集中没有默认分数，无需重试处理")
        else:
            self.process_failed_items_with_adaptive_config(
                dataset, output_file, self.max_workers, text_batch_size, triple_batch_size)

        # 第四阶段：处理缺失分数字段的项目
        print("\n" + "=" * 60)
        print("第四阶段：处理缺失分数字段的项目")
        print("=" * 60)

        # 先检查当前数据集中的缺失分数字段情况
        missing_text_count, missing_triple_count = self.count_missing_score_fields(dataset)
        print(f"📊 当前数据集中缺失分数字段统计 - 文本: {missing_text_count} 个, 三元组: {missing_triple_count} 个")

        if missing_text_count == 0 and missing_triple_count == 0:
            print("✅ 数据集中没有缺失分数字段的项目，无需处理")
        else:
            self.process_missing_score_fields_with_adaptive_config(
                dataset, output_file, self.max_workers, text_batch_size, triple_batch_size)

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
            default_text_count, default_triple_count = self.count_default_scores(dataset)
            missing_text_count, missing_triple_count = self.count_missing_score_fields(dataset)
            print(f"🏁 最终统计:")
            print(f"   - 剩余默认分数: 文本 {default_text_count} 个, 三元组 {default_triple_count} 个")
            print(f"   - 剩余缺失分数字段: 文本 {missing_text_count} 个, 三元组 {missing_triple_count} 个")

        except Exception as e:
            print(f"保存最终输出文件失败: {e}")


def main():
    """
    主函数 - 使用示例
    """
    # 配置参数
    API_KEY = ""  # 请填写您的ZhipuAI API Key
    INPUT_FILE = "wiki_test1000_add_ctxs.json"
    OUTPUT_FILE = "wiki_test1000_add_truthful_scores_with_kgs.json"

    # 并行处理参数
    MAX_WORKERS = 3000  # 并发线程数，根据API限制调整
    TEXT_BATCH_SIZE = 2  # 文本处理批次大小
    TRIPLE_BATCH_SIZE = 2  # 三元组处理批次大小

    # ⭐ 控制参数：选择执行模式
    RETRY_ONLY = True  # 设置为True表示仅处理默认分数为12的失败项目
    MISSING_FIELDS_ONLY = False  # 设置为True表示仅处理缺失分数字段的项目

    # 🆕 新增控制参数：单个处理模式
    INDIVIDUAL_PROCESSING = True  # 设置为True表示使用单个三元组/文本处理模式
    TRIPLES_PER_CALL = 1  # 每次API调用处理的三元组数量
    # 注意：如果MISSING_FIELDS_ONLY=True，则RETRY_ONLY的值会被忽略
    # 三种模式：
    # 1. MISSING_FIELDS_ONLY=True: 仅处理缺失分数字段
    # 2. RETRY_ONLY=True, MISSING_FIELDS_ONLY=False: 仅处理默认分数12的项目
    # 3. 两者都为False: 执行完整流程

    if not API_KEY:
        print("错误：请先设置您的ZhipuAI API Key")
        return

    # 创建评估器实例
    evaluator = OptimizedTruthfulScoreEvaluator(API_KEY, max_workers=MAX_WORKERS)

    # 根据参数执行不同的处理流程
    if MISSING_FIELDS_ONLY:
        print("🔍 启用仅缺失字段处理模式")
        print(f"📂 将从文件 {INPUT_FILE} 中读取数据，仅处理缺失分数字段的项目")
    elif RETRY_ONLY:
        if INDIVIDUAL_PROCESSING:
            print("🔄 启用单个三元组处理模式")
        print("🔄 启用仅重试模式")
        print(f"📂 将从文件 {INPUT_FILE} 中读取数据，仅处理默认分数为12的项目")
    else:
        print("🚀 启用完整处理模式")
        print(f"📂 将完整处理文件 {INPUT_FILE} 中的所有数据")

    # 优化处理数据集
    evaluator.process_dataset_optimized(
        INPUT_FILE,
        OUTPUT_FILE,
        text_batch_size=TEXT_BATCH_SIZE,
        triple_batch_size=TRIPLE_BATCH_SIZE,
        retry_only=RETRY_ONLY,
        missing_fields_only=MISSING_FIELDS_ONLY,  # 传入新参数
        individual_processing=INDIVIDUAL_PROCESSING,  # 传入新参数
        triples_per_call=TRIPLES_PER_CALL,  # 传入新参数
    )


if __name__ == "__main__":
    main()
