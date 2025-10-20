import torch
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple
from zhipuai import ZhipuAI
import re

class CollatorWithChainsChatFormat:

    def __init__(self, tokenizer, text_maxlength, answer_maxlength=25, context_type="triples", **kwargs):
        """
        Initialize the collator for chat-based models in the TRACE framework

        Args:
            tokenizer: The tokenizer corresponding to the model being used
            text_maxlength: Maximum length for the input text
            answer_maxlength: Maximum length for the generated answer
            context_type: Type of context to use ("triples", "documents", or "all_documents")
            **kwargs: Additional arguments
        """
        self.tokenizer = tokenizer
        self.tokenizer.padding_side = "left"
        self.text_maxlength = text_maxlength
        self.answer_maxlength = answer_maxlength
        assert context_type in ["triples", "documents", "all_documents"]
        self.context_type = context_type
        self.kwargs = kwargs

    def get_contexts(self, example):
        """
        (3) ANSWER GENERATION STEP: Extract context for answer generation based on reasoning chains

        This function implements different context retrieval strategies:
        - "triples": Uses the reasoning chains directly as context (TRACE-Triple)
        - "documents": Uses original documents referenced by the triples in chains (TRACE-Doc)
        - "all_documents": Uses all available documents as context

        Args:
            example: The example containing chains and documents

        Returns:
            str: Formatted context text for answer generation
        """
        chains = example["chains"]
        contexts = example["contexts"]

        if self.context_type == "triples":
            # TRACE-Triple: Use the reasoning chains directly as context
            chains_list = []
            for i, chain in enumerate(chains):
                for triple_item in chain["triples"]:
                    triple = triple_item['triple']
                    triple_sentence = triple.replace("<", "").replace(">", "").replace(";", "", 2)
                    if triple_sentence not in chains_list:
                        chains_list.append(triple_sentence)

        if self.context_type == "documents":
            # TRACE-Doc: Use original documents referenced by the triples in chains
            chains_documents_indices_count_dict = {}
            for i, chain in enumerate(chains):
                for triple_item in chain["triples"]:
                    doc_idx, sent_idx = triple_item["triple_position"]
                    if doc_idx >= 0:
                        chains_documents_indices_count_dict[doc_idx] = chains_documents_indices_count_dict.get(doc_idx,
                                                                                                               0) + 1

            chains_with_documents_list = []
            ranked_chains_documents_indices = sorted(chains_documents_indices_count_dict.items(), key=lambda x: x[1],
                                                     reverse=True)
            for idx, count in ranked_chains_documents_indices:
                chains_with_documents_list.append(
                    "title: {}, text: {}".format(contexts[idx]["title"], " ".join(contexts[idx]["sentences"])))

        if self.context_type == "all_documents":
            # Use all available documents as context
            all_documents_list = [
                "title: {}, text: {}".format(context_item["title"], " ".join(context_item["sentences"])) for
                context_item in contexts]

        # Select the appropriate context list based on context_type
        if self.context_type == "triples":
            context_text_list = chains_list
        elif self.context_type == "documents":
            context_text_list = chains_with_documents_list
        elif self.context_type == "all_documents":
            context_text_list = all_documents_list

        # context_text = "\n".join(["{}. {}".format(i + 1, text) for i, text in enumerate(context_text_list)])

        return context_text_list

    def get_prompts_chat_format(self, batch):
        """
        Generate prompts in chat format for answer generation

        Args:
            batch: Batch of examples to process

        Returns:
            list: List of formatted prompts ready for the model
        """

        def convert_several_examplars_to_text(examplars):
            return "\n\n".join(examplars)

        def extract_scores_simple(text: str) -> List[Tuple[int, int]]:
            """
            简化版本：直接按顺序提取所有Credibility和Usefulness评分
            """
            credibility_scores = re.findall(r'Credibility Score:\s*(\d+)', text)
            usefulness_scores = re.findall(r'Usefulness Score:\s*(\d+)', text)

            scores = []
            for i in range(min(len(credibility_scores), len(usefulness_scores))):
                credibility = int(credibility_scores[i])
                usefulness = int(usefulness_scores[i])
                scores.append((credibility, usefulness))

            return scores

        def call_api_for_scoring(context_text_list, question, client):
            """
            调用API对context_text_list进行评分
            """
            instruction = """Your task is to evaluate the authenticity and usefulness of the text based on your internal knowledge. Specifically, I will provide you with a question and multiple paragraphs that may contain accurate information or fabricated errors. Using your own knowledge, reasoning, and inference, you are to assign two scores: a credibility score and a usefulness score, both on a scale from 0 to 10.A higher credibility score indicates higher authenticity, while a lower score indicates lower authenticity.A higher usefulness score means the text provides greater help in the logical steps required to answer the question. A lower usefulness score means it provides less help. Here is an example, you should follow the following output format:
##########
Question: 
Who was the U.S. President when the scientist who discovered penicillin was knighted?
Paragraph:
Sir Alexander Fleming, a Scottish physician and microbiologist, was formally knighted for his services to science and medicine in 1944 by King George VI at Buckingham Palace.

Analysis: This passage is historically accurate. It correctly identifies the scientist and the year he was knighted, which is 1944. This year is the most critical piece of information needed to solve the multi-hop question. Therefore, the passage is both highly credible and highly useful. 

Credibility Score: 10 
Usefulness Score: 10


Paragraph:
In 1951, during the presidency of Harry S. Truman, the renowned scientist Alexander Fleming was knighted for his discovery of antibiotics. The ceremony was widely celebrated as a symbol of post-war scientific achievement.

Analysis: This passage contains critical factual errors. Alexander Fleming was knighted in 1944, not 1951. Consequently, the U.S. President at the time of his knighthood was Franklin D. Roosevelt, not Harry S. Truman (who took office in 1945). Although the passage appears relevant by mentioning a scientist, a knighthood, and a U.S. President, its core facts are incorrect, making it highly misleading and not useful.

Credibility Score: 1 
Usefulness Score: 0
##########"""

            question_template = """Question:
    {question}"""
            context_input_template = """Paragraph:
    {paragraph}

    Credibility Score:
    Usefulness Score:
    """

            context_input_list = []
            question_input = question_template.format(question=question)

            for ctx in context_text_list:
                context_input_list.append(context_input_template.format(paragraph=ctx))

            context_input = "\n".join(context_input_list)
            user_input = f"{question_input}\n{context_input}"

            max_retries = 3
            for attempt in range(max_retries):
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

                    scores = extract_scores_simple(full_response_content)

                    # 检查分数总数是否等于context_text_list的长度
                    if len(scores) == len(context_text_list):
                        return scores
                    else:
                        print(
                            f"尝试 {attempt + 1}: 分数数量 ({len(scores)}) 不等于上下文数量 ({len(context_text_list)})")
                        if attempt == max_retries - 1:
                            warnings.warn(f"API调用失败超过最大重试次数，返回默认分数")
                            return [(10, 10)] * len(context_text_list)

                except Exception as e:
                    print(f"尝试 {attempt + 1} 时发生错误: {e}")
                    if attempt == max_retries - 1:
                        warnings.warn(f"API调用出现异常，返回默认分数")
                        return [(10, 10)] * len(context_text_list)

            return [(10, 10)] * len(context_text_list)

        # 初始化API客户端
        client = ZhipuAI(api_key="05748176082447f483438dfd914cc299.NcsCifhTarCch6es")

        prompts = []
        has_contexts = batch[0]["chains"] is not None

        # 第一步：遍历batch中的所有example，收集所有context_text_list
        context_text_lists = []
        questions = []

        for example in batch:
            if has_contexts:
                context_text_list = self.get_contexts(example)
                context_text_lists.append(context_text_list)
                questions.append(example["question"])
            else:
                context_text_lists.append([])
                questions.append(example["question"])

        # 第二步：多线程调用API获取分数
        batch_texts = []

        if has_contexts and context_text_lists:
            with ThreadPoolExecutor(max_workers=len(batch)) as executor:
                # 提交所有任务
                future_to_index = {}
                for i, (context_text_list, question) in enumerate(zip(context_text_lists, questions)):
                    if context_text_list:  # 只有当context_text_list不为空时才调用API
                        future = executor.submit(call_api_for_scoring, context_text_list, question, client)
                        future_to_index[future] = i
                    else:
                        batch_texts.append("")  # 空的context对应空字符串

                # 收集结果
                results = [None] * len(batch)
                for future in as_completed(future_to_index):
                    index = future_to_index[future]
                    try:
                        scores = future.result()
                        results[index] = scores
                    except Exception as e:
                        print(f"线程 {index} 执行失败: {e}")
                        results[index] = [(10, 10)] * len(context_text_lists[index])

                # 第三步：将分数拼接到context_text_list中
                for i, (context_text_list, scores) in enumerate(zip(context_text_lists, results)):
                    if context_text_list and scores:
                        final_paras = []
                        for j, (context_text, (cred_score, useful_score)) in enumerate(zip(context_text_list, scores)):
                            # 计算融合分数：(可信度分数 × 有用性分数) / 10，保留1位小数
                            fusion_score = round((cred_score * useful_score) / 10, 1)
                            formatted_context = f"[CUFScore: {fusion_score}]: {context_text}"
                            final_paras.append(formatted_context)

                        context_text = "\n".join(final_paras)
                        batch_texts.append(context_text)
                    else:
                        batch_texts.append("")
        else:
            # 如果没有contexts，则填充空字符串
            batch_texts = [""] * len(batch)

        # 第四步：生成最终的prompts
        for i, example in enumerate(batch):
            if has_contexts:
                instruction = """You are an assistant that answers questions using multiple paragraphs. Each paragraph has a credibility–usefulness fusion score (CUFScore, 0.0–10.0), which reflects that paragraph’s contribution to the required reasoning steps (usefulness) and its factual credibility. Use all paragraphs, prioritizing information from higher CUFScores when resolving conflicts. If the paragraphs do not contain enough information, answer based on your own internal knowledge.Given the paragraphs (each with a CUFScore) and the question, please only output the answer to the question. Given the following information: 
{Documents} 
Answer the following question based on the given information or your internal knowledge with one or few words without the source (just output a answer, don't output anything else). 
Question: {Question} 
the correct answer is:"""

                # 将评分后的上下文文本插入到instruction中
                user_input_text = instruction.format(
                    Documents=batch_texts[i] if batch_texts[i] else "No documents provided",
                    Question=example["question"]
                )
            else:
                instruction = "Given a question, please only output the answer to the question."
                user_input_text = example["question"] + "\n" + "the correct answer is:"
            if has_contexts:
                prompts.append(
                    [
                        {"role": "user", "content": user_input_text}
                    ]
                )
            else:
                prompts.append(
                    [
                        {"role": "system", "content": instruction},
                        {"role": "user", "content": user_input_text}
                    ]
                )
        return prompts

    def tokenizer_encode(self, prompts):
        """
        Tokenize the prompts for the model

        Args:
            prompts: Prompts in chat format

        Returns:
            dict: Dictionary containing input_ids and attention_mask for the model
        """
        texts = self.tokenizer.apply_chat_template(prompts, tokenize=False, add_generation_prompt=True)
        batch_dict = self.tokenizer(texts, max_length=self.text_maxlength, padding=True, truncation=True,
                                    return_tensors='pt')
        tokenizer_outputs = {"input_ids": batch_dict["input_ids"], "attention_mask": batch_dict["attention_mask"]}
        return tokenizer_outputs

    def __call__(self, batch):
        """
        Process a batch of examples for answer generation

        Args:
            batch: Batch of examples

        Returns:
            tuple: (index, inputs) for the model
        """
        batch_size = len(batch)
        index = torch.tensor([example['index'] for example in batch])
        prompts = self.get_prompts_chat_format(batch)
        inputs = self.tokenizer_encode(prompts)
        return index, inputs


class CollatorWithChains(CollatorWithChainsChatFormat):
    """
    Alternative collator implementation for non-chat models

    This class inherits from CollatorWithChainsChatFormat but implements
    a different prompt formatting approach for traditional language models
    without a chat template.
    """

    def __init__(self, tokenizer, text_maxlength, answer_maxlength=25, context_type="triples", **kwargs):

        self.tokenizer = tokenizer
        self.tokenizer.padding_side = "left"
        self.text_maxlength = text_maxlength
        self.answer_maxlength = answer_maxlength
        assert context_type in ["triples", "documents", "all_documents"]
        self.context_type = context_type
        self.kwargs = kwargs

    def get_prompts(self, batch):
        """
        Generate prompts in standard format (non-chat) for answer generation

        Args:
            batch: Batch of examples to process

        Returns:
            list: List of formatted prompts ready for the model
        """
        has_contexts = batch[0]["chains"] is not None
        if has_contexts:
            instruction = "Given some contexts and a question, please only output the answer to the question.\n"
        else:
            instruction = "Given a question, please only output the answer to the question.\n"

        prompts_list = []
        for example in batch:
            question = example["question"]
            if has_contexts:
                context = "context:\n{}".format(self.get_contexts(example))
                prompt = context + "\n" + question
                prompts_list.append(prompt)
            else:
                prompts_list.append(question)

        prompts_list = ["{}{}\nthe correct answer is:".format(instruction, prompt) for prompt in prompts_list]
        return prompts_list

    def tokenizer_encode(self, prompts):
        """
        Tokenize the standard (non-chat) prompts for the model

        Args:
            prompts: List of prompt strings

        Returns:
            dict: Dictionary containing input_ids and attention_mask for the model
        """
        batch_dict = self.tokenizer(prompts, max_length=self.text_maxlength, padding=True, truncation=True,
                                    return_tensors='pt')
        tokenizer_outputs = {"input_ids": batch_dict["input_ids"], "attention_mask": batch_dict["attention_mask"]}
        return tokenizer_outputs

    def __call__(self, batch):
        """
        Process a batch of examples for answer generation

        Args:
            batch: Batch of examples

        Returns:
            tuple: (index, inputs) for the model
        """
        batch_size = len(batch)
        index = torch.tensor([example['index'] for example in batch])
        prompts = self.get_prompts(batch)
        inputs = self.tokenizer_encode(prompts)
        return index, inputs