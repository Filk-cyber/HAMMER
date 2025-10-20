import json
import time
from zhipuai import ZhipuAI

class TruthfulScoreEvaluator:
    def __init__(self, api_key: str):
        """
        初始化真实性评分器

        Args:
            api_key: ZhipuAI的API密钥
        """
        self.client = ZhipuAI(api_key=api_key)

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
            # 提取连续数字（支持0-10分）
            score = ''.join(filter(str.isdigit, score_text.split()[0] if score_text.split() else ''))
            return int(score) if score.isdigit() else 0
        return 0

    def get_triple_score(self, head: str, relation: str, tail: str) -> int:
        """
        获取知识三元组的真实性分数

        Args:
            head: 三元组的头实体
            relation: 三元组的关系
            tail: 三元组的尾实体

        Returns:
            真实性分数（0-10的整数）
        """
        user_input = f"""Triple:
head: {head}
relation: {relation}
tail: {tail}

Credibility Score: 
"""

        try:
            response = self.client.chat.completions.create(
                model="glm-4.5",
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

            return self.extract_credibility_score(full_response_content)

        except Exception as e:
            print(f"评估三元组时发生错误: {e}")
            return 0

    def get_text_score(self, text: str) -> int:
        """
        获取文本段落的真实性分数

        Args:
            text: 要评估的文本段落

        Returns:
            真实性分数（0-10的整数）
        """
        user_input = f"""Passage:
{text}

Credibility Score: 
"""

        try:
            response = self.client.chat.completions.create(
                model="glm-4.5",
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
            print(f"评估文本时发生错误: {e}")
            return 0

    def process_dataset(self, input_file: str, output_file: str, delay: float = 1.0):
        """
        处理整个数据集，添加真实性分数字段

        Args:
            input_file: 输入JSON文件路径
            output_file: 输出JSON文件路径
            delay: API调用之间的延迟时间（秒）
        """
        print(f"开始处理数据集: {input_file}")

        # 读取输入文件
        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                dataset = json.load(f)
        except Exception as e:
            print(f"读取输入文件失败: {e}")
            return

        # 确保dataset是列表
        if not isinstance(dataset, list):
            dataset = [dataset]

        total_items = len(dataset)

        for idx, item in enumerate(dataset):
            print(f"处理第 {idx + 1}/{total_items} 个数据项...")

            # 处理ctxs字段
            if 'ctxs' in item and isinstance(item['ctxs'], list):
                for ctx_idx, ctx in enumerate(item['ctxs']):
                    print(f"  处理第 {ctx_idx + 1} 个ctx...")

                    # # 为text字段添加真实性分数
                    # if 'text' in ctx:
                    #     print("    评估text真实性分数...")
                    #     ctx['text_truthful_score'] = self.get_text_score(ctx['text'])
                    #     time.sleep(delay)  # API调用延迟

                    # 处理triples字段
                    if 'triples' in ctx and isinstance(ctx['triples'], list):
                        for triple_idx, triple in enumerate(ctx['triples']):
                            print(f"      评估第 {triple_idx + 1} 个triple...")

                            if all(key in triple for key in ['head', 'relation', 'tail']):
                                triple['triple_truthful_score'] = self.get_triple_score(
                                    triple['head'],
                                    triple['relation'],
                                    triple['tail']
                                )
                                # time.sleep(delay)  # API调用延迟
                            else:
                                print(f"        跳过不完整的triple: {triple}")

            # 定期保存进度
            if (idx + 1) % 5 == 0:
                print(f"保存中间进度...")
                with open(f"{output_file}.tmp", 'w', encoding='utf-8') as f:
                    json.dump(dataset, f, ensure_ascii=False, indent=2)

        # 保存最终结果
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(dataset, f, ensure_ascii=False, indent=2)
            print(f"处理完成！结果已保存到: {output_file}")

            # 删除临时文件
            import os
            if os.path.exists(f"{output_file}.tmp"):
                os.remove(f"{output_file}.tmp")

        except Exception as e:
            print(f"保存输出文件失败: {e}")


def main():
    """
    主函数 - 使用示例
    """
    # 配置参数
    API_KEY = "05748176082447f483438dfd914cc299.NcsCifhTarCch6es"  # 请填写您的ZhipuAI API Key
    INPUT_FILE = "/home/jiangjp/trace-idea/data/hotpotqa/hotpotqa_dev100_add_truthful_scores_with_kgs.json"  # 输入文件路径
    OUTPUT_FILE = "/home/jiangjp/trace-idea/data/hotpotqa/hotpotqa_dev100_add_truthful_scores_with_kgs2.json"  # 输出文件路径
    API_DELAY = 1.0  # API调用间隔（秒）

    if not API_KEY:
        print("错误：请先设置您的ZhipuAI API Key")
        return

    # 创建评估器实例
    evaluator = TruthfulScoreEvaluator(API_KEY)

    # 处理数据集
    evaluator.process_dataset(INPUT_FILE, OUTPUT_FILE, API_DELAY)


if __name__ == "__main__":
    main()