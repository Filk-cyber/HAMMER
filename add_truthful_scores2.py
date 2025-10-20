import json
import time
import concurrent.futures
from threading import Lock
from zhipuai import ZhipuAI
from typing import List, Dict, Any
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

        # 知识三元组评估的指令模板
        self.triple_instruction = """Your task is to evaluate the authenticity of knowledge triplets based on your internal knowledge, reasoning, and inference. The structure of a knowledge triplet is ⟨ head; relation; tail⟩， Represents a single factual statement about the relationship between entities. I will provide a knowledge triad that may contain accurate information or fictional errors. You need to assign it a credibility score from 0 to 10, with higher scores indicating higher authenticity and lower scores indicating lower authenticity. Here are 2 examples (you should follow the following output format):
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
Here are 2 examples (you should follow the output format below):
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

    def extract_multiple_credibility_scores(self, text: str, num_triples: int) -> List[int]:
        """
        从批量处理的GPT响应中提取多个可信度分数

        Args:
            text: GPT的完整响应文本
            num_triples: 预期的三元组数量

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

        # 如果找到的分数数量不对，尝试其他方法或填充默认值
        if len(scores) != num_triples:
            print(f"警告：期望 {num_triples} 个分数，但只找到 {len(scores)} 个")
            # 填充或截断到正确数量
            while len(scores) < num_triples:
                scores.append(0)
            scores = scores[:num_triples]

        return scores

    def get_text_score_with_retry(self, text: str, max_retries: int = 3) -> int:
        """
        获取文本段落的真实性分数（带重试机制）
        """
        for attempt in range(max_retries):
            try:
                user_input = f"""Passage:
{text}

Credibility Score: 
"""

                response = self.client.chat.completions.create(
                    model="glm-4.5-flash",
                    messages=[
                        {"role": "system", "content": self.text_instruction},
                        {"role": "user", "content": user_input},
                    ],
                    stream=True,
                )

                full_response_content = ""
                for chunk in response:
                    delta = chunk.choices[0].delta
                    if delta.content:
                        full_response_content += delta.content

                return self.extract_credibility_score(full_response_content)

            except Exception as e:
                print(f"第 {attempt + 1} 次文本评估尝试失败: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    print(f"文本评估失败，使用默认分数0")
                    return 0

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

                response = self.client.chat.completions.create(
                    model="glm-4.5-flash",
                    messages=[
                        {"role": "system", "content": self.triple_instruction},
                        {"role": "user", "content": user_input},
                    ],
                    stream=True,
                )

                full_response_content = ""
                for chunk in response:
                    delta = chunk.choices[0].delta
                    if delta.content:
                        full_response_content += delta.content

                return self.extract_multiple_credibility_scores(full_response_content, len(triples))

            except Exception as e:
                print(f"第 {attempt + 1} 次批量三元组评估尝试失败: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    print(f"批量三元组评估失败，使用默认分数")
                    return [0] * len(triples)

    def process_single_text(self, text_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        处理单个文本
        """
        text = text_data['text']
        item_idx = text_data['item_idx']
        ctx_idx = text_data['ctx_idx']

        try:
            score = self.get_text_score_with_retry(text)

            with self.progress_lock:
                self.text_completed_count += 1
                print(f"文本评估进度: {self.text_completed_count}/{self.text_total_count} "
                      f"(Item {item_idx + 1}, Ctx {ctx_idx + 1}) - Score: {score}")

            return {
                'item_idx': item_idx,
                'ctx_idx': ctx_idx,
                'score': score,
                'success': True
            }

        except Exception as e:
            print(f"处理文本时发生错误: {e}")
            return {
                'item_idx': item_idx,
                'ctx_idx': ctx_idx,
                'score': 0,
                'success': False
            }

    def process_single_ctx_triples(self, ctx_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        处理单个ctx中的所有triples（批量）
        """
        triples = ctx_data['triples']
        item_idx = ctx_data['item_idx']
        ctx_idx = ctx_data['ctx_idx']

        try:
            # 过滤完整的三元组
            valid_triples = []
            valid_indices = []

            for triple_idx, triple in enumerate(triples):
                if all(key in triple for key in ['head', 'relation', 'tail']):
                    valid_triples.append(triple)
                    valid_indices.append(triple_idx)

            if not valid_triples:
                return {
                    'item_idx': item_idx,
                    'ctx_idx': ctx_idx,
                    'scores': [],
                    'valid_indices': [],
                    'success': True
                }

            # 批量获取分数
            scores = self.get_batch_triple_scores_with_retry(valid_triples)

            with self.progress_lock:
                self.triple_completed_count += 1
                print(f"三元组批量评估进度: {self.triple_completed_count}/{self.triple_total_count} "
                      f"(Item {item_idx + 1}, Ctx {ctx_idx + 1}) - 处理了 {len(valid_triples)} 个三元组")

            return {
                'item_idx': item_idx,
                'ctx_idx': ctx_idx,
                'scores': scores,
                'valid_indices': valid_indices,
                'success': True
            }

        except Exception as e:
            print(f"处理ctx中的三元组时发生错误: {e}")
            return {
                'item_idx': item_idx,
                'ctx_idx': ctx_idx,
                'scores': [],
                'valid_indices': [],
                'success': False
            }

    def collect_text_data(self, dataset: List[Dict]) -> List[Dict[str, Any]]:
        """
        收集所有需要处理的文本数据
        """
        text_data = []

        for item_idx, item in enumerate(dataset):
            if 'ctxs' in item and isinstance(item['ctxs'], list):
                for ctx_idx, ctx in enumerate(item['ctxs']):
                    if 'text' in ctx:
                        text_data.append({
                            'text': ctx['text'],
                            'item_idx': item_idx,
                            'ctx_idx': ctx_idx
                        })

        return text_data

    def collect_triples_data(self, dataset: List[Dict]) -> List[Dict[str, Any]]:
        """
        收集所有需要处理的ctx数据（包含triples）
        """
        triples_data = []

        for item_idx, item in enumerate(dataset):
            if 'ctxs' in item and isinstance(item['ctxs'], list):
                for ctx_idx, ctx in enumerate(item['ctxs']):
                    if 'triples' in ctx and isinstance(ctx['triples'], list) and len(ctx['triples']) > 0:
                        triples_data.append({
                            'triples': ctx['triples'],
                            'item_idx': item_idx,
                            'ctx_idx': ctx_idx
                        })

        return triples_data

    def apply_text_results(self, dataset: List[Dict], results: List[Dict]):
        """
        将文本评估结果应用到数据集
        """
        for result in results:
            if result['success']:
                item_idx = result['item_idx']
                ctx_idx = result['ctx_idx']
                score = result['score']

                try:
                    dataset[item_idx]['ctxs'][ctx_idx]['text_truthful_score'] = score
                except (IndexError, KeyError) as e:
                    print(f"应用文本结果时发生错误: {e}")

    def apply_triple_results(self, dataset: List[Dict], results: List[Dict]):
        """
        将三元组评估结果应用到数据集
        """
        for result in results:
            if result['success']:
                item_idx = result['item_idx']
                ctx_idx = result['ctx_idx']
                scores = result['scores']
                valid_indices = result['valid_indices']

                try:
                    for score, triple_idx in zip(scores, valid_indices):
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

    def process_dataset_optimized(self, input_file: str, output_file: str):
        """
        优化版数据集处理：先处理文本，再处理三元组
        """
        print(f"开始优化处理数据集: {input_file}")

        # 读取输入文件
        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                dataset = json.load(f)
        except Exception as e:
            print(f"读取输入文件失败: {e}")
            return

        if not isinstance(dataset, list):
            dataset = [dataset]

        print("=" * 60)
        print("第一阶段：处理文本数据（多线程）")
        print("=" * 60)

        # 第一阶段：处理文本数据
        text_data = self.collect_text_data(dataset)
        self.text_total_count = len(text_data)
        self.text_completed_count = 0

        print(f"总共需要处理 {self.text_total_count} 个文本")

        if self.text_total_count > 0:
            text_results = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_text = {
                    executor.submit(self.process_single_text, text_item): text_item
                    for text_item in text_data
                }

                for future in concurrent.futures.as_completed(future_to_text):
                    result = future.result()
                    text_results.append(result)

            # 应用文本结果
            self.apply_text_results(dataset, text_results)

            # 保存文本处理进度
            self.save_progress(dataset, output_file, "text")

            success_count = sum(1 for r in text_results if r['success'])
            print(f"文本处理完成: {success_count}/{len(text_results)} 成功")

        print("\n" + "=" * 60)
        print("第二阶段：处理三元组数据（多线程+批量优化）")
        print("=" * 60)

        # 第二阶段：处理三元组数据
        triples_data = self.collect_triples_data(dataset)
        self.triple_total_count = len(triples_data)
        self.triple_completed_count = 0

        print(f"总共需要处理 {self.triple_total_count} 个ctx（每个ctx批量处理其中的三元组）")

        if self.triple_total_count > 0:
            triple_results = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_ctx = {
                    executor.submit(self.process_single_ctx_triples, triple_item): triple_item
                    for triple_item in triples_data
                }

                for future in concurrent.futures.as_completed(future_to_ctx):
                    result = future.result()
                    triple_results.append(result)

            # 应用三元组结果
            self.apply_triple_results(dataset, triple_results)

            success_count = sum(1 for r in triple_results if r['success'])
            print(f"三元组处理完成: {success_count}/{len(triple_results)} 成功")

        # 保存最终结果
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(dataset, f, ensure_ascii=False, indent=2)
            print(f"处理完成！结果已保存到: {output_file}")

            # 删除临时文件
            import os
            for stage in ['text']:
                temp_file = f"{output_file}.{stage}.tmp"
                if os.path.exists(temp_file):
                    os.remove(temp_file)

        except Exception as e:
            print(f"保存输出文件失败: {e}")


def main():
    """
    主函数 - 使用示例
    """
    # 配置参数
    API_KEY = "05748176082447f483438dfd914cc299.NcsCifhTarCch6es"  # 请填写您的ZhipuAI API Key
    INPUT_FILE = "/home/jiangjp/trace-idea/data/hotpotqa/hotpotqa_dev100_add_truthful_scores_with_kgs.json"
    OUTPUT_FILE = "/home/jiangjp/trace-idea/data/hotpotqa/hotpotqa_dev100_add_truthful_scores_with_kgs_optimized.json"

    # 并行处理参数
    MAX_WORKERS = 5  # 并发线程数，根据API限制调整

    if not API_KEY:
        print("错误：请先设置您的ZhipuAI API Key")
        return

    # 创建评估器实例
    evaluator = OptimizedTruthfulScoreEvaluator(API_KEY, max_workers=MAX_WORKERS)

    # 优化处理数据集
    evaluator.process_dataset_optimized(INPUT_FILE, OUTPUT_FILE)


if __name__ == "__main__":
    main()
