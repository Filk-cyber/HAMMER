import os
import json
import argparse
import numpy as np
import torch
from torch import Tensor
from typing import Dict, List
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from retrievers.e5_mistral import get_e5_mistral_embeddings_for_query, get_e5_mistral_embeddings_for_document
from readers.metrics import ems, f1_score,accuracy

# 全局变量
device = "cuda" if torch.cuda.is_available() else "cpu"


def load_json(file_path):
    """加载JSON文件"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


# ============ Llama3-8B-Instruct模型加载器 ============
def load_llama3_model_tokenizer(model_path):
    """
    加载Llama3-8B-Instruct模型和tokenizer

    Args:
        model_path: 模型路径

    Returns:
        tuple: (model, tokenizer)
    """
    print(f"正在加载Llama3-8B-Instruct模型: {model_path}")

    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        padding_side="left"
    )

    # 设置pad_token
    if tokenizer.pad_token is None or tokenizer.pad_token_id is None:
        print("设置padding token为eos_token")
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # 加载模型
    if device == "cuda":
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float32
        )
        model.to(device)

    model.eval()
    print("模型加载完成!")
    return model, tokenizer


# ============ Gemma-7B模型加载器 ============
def load_gemma_model_tokenizer(model_path):
    """
    加载Gemma-7B模型和tokenizer

    Args:
        model_path: 模型路径

    Returns:
        tuple: (model, tokenizer)
    """
    print(f"正在加载Gemma-7B模型: {model_path}")

    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        padding_side="left",
        trust_remote_code=True
    )

    # 设置pad_token
    if tokenizer.pad_token is None or tokenizer.pad_token_id is None:
        print("设置padding token为eos_token")
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # 加载模型
    if device == "cuda":
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float32,
            trust_remote_code=True
        )
        model.to(device)

    model.eval()
    print("Gemma模型加载完成!")
    return model, tokenizer

# ============ Mistral-7B模型加载器 ============
def load_mistral_model_tokenizer(model_path):
    """
    加载Mistral-7B模型和tokenizer

    Args:
        model_path: 模型路径

    Returns:
        tuple: (model, tokenizer)
    """
    print(f"正在加载Mistral-7B模型: {model_path}")

    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        padding_side="left",
        trust_remote_code=True
    )

    # 设置pad_token
    if tokenizer.pad_token is None or tokenizer.pad_token_id is None:
        print("设置padding token为eos_token")
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # 加载模型
    if device == "cuda":
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float32,
            trust_remote_code=True
        )
        model.to(device)

    model.eval()
    print("Gemma模型加载完成!")
    return model, tokenizer

# ============ 基于相似度的检索函数 ============
def retrieve_documents_by_similarity(question: str, ctxs: List[Dict], args) -> List[Dict]:
    """
    基于相似度的检索函数：为单个问题检索最相关的文档
    只使用相似度分数，不考虑truthful_score

    Args:
        question: 问题文本
        ctxs: 候选文档列表
        args: 参数配置

    Returns:
        List[Dict]: 检索到的top-k文档，包含text字段
    """
    # 提取所有候选文档
    documents = []
    end_index = len(ctxs) - 3
    ctxs = ctxs[:end_index + args.fake_num]
    for ctx in ctxs:
        documents.append("title: {}, text: {}".format(ctx["title"], ctx["text"]))

    # 计算文档embeddings
    doc_embeddings = get_e5_mistral_embeddings_for_document(documents, max_length=256, batch_size=2)
    question_embedding = get_e5_mistral_embeddings_for_query("retrieve_relevant_documents", [question],
                                                             max_length=128, batch_size=1)

    # 归一化embeddings
    doc_embeddings = torch.nn.functional.normalize(doc_embeddings, p=2, dim=-1)
    question_embedding = torch.nn.functional.normalize(question_embedding, p=2, dim=-1)

    # 计算相似度分数
    similarities = torch.matmul(question_embedding, doc_embeddings.T).squeeze(0)

    # 选择top-k文档
    topk_scores, topk_indices = torch.topk(similarities, k=min(args.context_nums, len(documents)), dim=0)

    # 构建检索结果
    retrieved_documents = []
    for idx in topk_indices.tolist():
        retrieved_documents.append({
            "text": documents[idx]
        })

    return retrieved_documents


def retrieve_documents_by_similarity_score(question: str, ctxs: List[Dict], args, ideal_setting: bool = False) -> List[
    Dict]:
    """
    单跳检索函数：为单个问题检索最相关的文档
    使用相似度与truthful_score相乘的评分方式

    Args:
        question: 问题文本
        ctxs: 候选文档列表
        args: 参数配置

    Returns:
        List[Dict]: 检索到的top-k文档，包含text和truthful_score字段
    """
    # 提取所有候选文档和真实性分数
    documents, truthful_scores = [], []
    end_index = len(ctxs) - 3
    ctxs = ctxs[:end_index + args.fake_num]
    for i, ctx in enumerate(ctxs):
        # 只使用text内容作为检索文档
        documents.append("title: {}, text: {}".format(ctx["title"], ctx["text"]))
        if ideal_setting:
            if i < end_index:
                truthful_scores.append(10)
            else:
                truthful_scores.append(1)
        else:
            # 获取文档的truthful_score
            truthful_scores.append(ctx["text_truthful_score"])

    # 将truthful_scores转换为tensor
    truthful_scores_tensor = torch.tensor(truthful_scores, dtype=torch.bfloat16)
    # 计算文档embeddings
    doc_embeddings = get_e5_mistral_embeddings_for_document(documents, max_length=256, batch_size=2)
    question_embedding = get_e5_mistral_embeddings_for_query("retrieve_relevant_documents", [question],
                                                             max_length=128, batch_size=1)
    doc_embeddings = torch.nn.functional.normalize(doc_embeddings, p=2, dim=-1)
    question_embedding = torch.nn.functional.normalize(question_embedding, p=2, dim=-1)

    # 计算相似度分数
    similarities = torch.matmul(question_embedding, doc_embeddings.T).squeeze(0)
    final_scores = similarities * truthful_scores_tensor
    # 选择top-k文档
    topk_scores, topk_indices = torch.topk(final_scores, k=min(args.context_nums, len(documents)), dim=0)

    # 构建检索结果，只返回text和truthful_score字段
    retrieved_documents = []
    for i, idx in enumerate(topk_indices.tolist()):
        retrieved_documents.append({
            "text": documents[idx],
            "credibility": truthful_scores[idx]
        })
    if not ideal_setting and args.exclusion:
        retrieved_documents = [doc for doc in retrieved_documents if doc["credibility"] > 3]
    return retrieved_documents


# ============ Llama3数据处理器 ============
class Llama3DataProcessor:
    """Llama3数据处理器"""

    def __init__(self, args):
        self.args = args

    def get_contexts(self, retrieved_docs: List[Dict]) -> str:
        """
        格式化检索到的文档为上下文

        Args:
            retrieved_docs: 检索到的文档列表

        Returns:
            str: 格式化后的上下文字符串
        """
        contexts = []
        for i, doc in enumerate(retrieved_docs):
            text = doc["text"]
            contexts.append(("Passage-%d: " % i) + text)

        return "\n".join(contexts)

    def get_contexts_score(self, retrieved_docs: List[Dict]) -> str:
        """
        格式化检索到的文档为上下文
        与get_contexts不同，此方法在每个Passage后添加truthful score
        Args:
            retrieved_docs: 检索到的文档列表

        Returns:
            str: 格式化后的上下文字符串
        """
        contexts = []
        for i, doc in enumerate(retrieved_docs):
            text = doc["text"]
            score = doc["credibility"]
            contexts.append(("Passage-%d: " % i) + text + (f'\nTruthful score: {score}\n'))

        return "\n".join(contexts)

    def create_llama3_prompt(self, question: str, retrieved_docs: List[Dict]) -> List[Dict]:
        """
        创建符合Llama3格式的prompt

        Args:
            question: 问题文本
            retrieved_docs: 检索到的文档列表

        Returns:
            List[Dict]: 聊天格式的prompt
        """
        if retrieved_docs:
            contexts = self.get_contexts(retrieved_docs)
            prompt = f'Given the following information: \n{contexts}\nAnswer the following question based on the given information or your internal knowledge with one or few words without the source.\nQuestion: {question}\nAnswer: '
        else:
            print("norag")
            prompt = f'Answer the following question based on your internal knowledge with one or few words.\nQuestion: {question}\nAnswer: '

        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant!"
            },
            {
                "role": "user",
                "content": prompt
            },
        ]

        return messages

    def create_llama3_prompt_score(self, question: str, retrieved_docs: List[Dict]) -> List[Dict]:
        """
        创建符合Llama3格式的prompt

        Args:
            question: 问题文本
            retrieved_docs: 检索到的文档列表

        Returns:
            List[Dict]: 聊天格式的prompt
        """
        if retrieved_docs:
            contexts = self.get_contexts_score(retrieved_docs)
            prompt = f'You are an assistant capable of answering questions based on provided passages. Each passage is assigned a truthfulness score (0-10), where a higher score indicates greater credibility. Your answer need to combine multiple passages and their credibility. Given the following information: \n{contexts}\nAnswer the following question based on the given information or your internal knowledge with one or few words without the source (just output a answer, don\'t output anything else).\nQuestion: {question}\nAnswer: '
        else:
            prompt = f'Answer the following question based on your internal knowledge with one or few words.\nQuestion: {question}\nAnswer: '

        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant!"
            },
            {
                "role": "user",
                "content": prompt
            },
        ]

        return messages


# ============ Gemma数据处理器 ============
class GemmaDataProcessor:
    """Gemma数据处理器"""

    def __init__(self, args):
        self.args = args

    def get_contexts(self, retrieved_docs: List[Dict]) -> str:
        """
        格式化检索到的文档为上下文

        Args:
            retrieved_docs: 检索到的文档列表

        Returns:
            str: 格式化后的上下文字符串
        """
        contexts = []
        for i, doc in enumerate(retrieved_docs):
            text = doc["text"]
            contexts.append(("Passage-%d: " % i) + text)

        return "\n".join(contexts)

    def get_contexts_score(self, retrieved_docs: List[Dict]) -> str:
        """
        格式化检索到的文档为上下文
        与get_contexts不同，此方法在每个Passage后添加truthful score
        Args:
            retrieved_docs: 检索到的文档列表

        Returns:
            str: 格式化后的上下文字符串
        """
        contexts = []
        for i, doc in enumerate(retrieved_docs):
            text = doc["text"]
            score = doc["credibility"]
            contexts.append(("Passage-%d: " % i) + text + (f'\nTruthful score: {score}\n'))

        return "\n".join(contexts)

    def create_gemma_prompt(self, question: str, retrieved_docs: List[Dict]) -> str:
        """
        创建符合Gemma格式的prompt

        Args:
            question: 问题文本
            retrieved_docs: 检索到的文档列表

        Returns:
            List[Dict]: 聊天格式的prompt
        """
        if retrieved_docs:
            contexts = self.get_contexts(retrieved_docs)
            prompt = f'Given the following information: \n{contexts}\nplease only output the answer to the question.\nQuestion:{question}\nthe correct answer is:'
        else:
            print("norag")
            prompt = f'Answer the following question based on your internal knowledge with one or few words.\nQuestion: {question}\nthe correct answer is:'

        return prompt

    def create_gemma_prompt_score(self, question: str, retrieved_docs: List[Dict]) -> str:
        """
        创建符合Gemma格式的prompt

        Args:
            question: 问题文本
            retrieved_docs: 检索到的文档列表

        Returns:
            List[Dict]: 聊天格式的prompt
        """
        if retrieved_docs:
            contexts = self.get_contexts_score(retrieved_docs)
            prompt = f'You are an assistant capable of answering questions based on provided passages. Each passage is assigned a truthfulness score (0-10), where a higher score indicates greater credibility. You should consider truthfulness score of the passage, if the score is low, you should not trust it. Given the following information: \n{contexts}\nAnswer the following question based on the given information or your internal knowledge with one or few words without the source (just output a answer, don\'t output anything else).\nQuestion: {question}\nthe correct answer is:'
        else:
            prompt = f'Answer the following question based on your internal knowledge with one or few words.\nQuestion: {question}\nthe correct answer is:'

        return prompt

# ============ Mistral数据处理器 ============
class MistralDataProcessor:
    """Mistral数据处理器"""

    def __init__(self, args):
        self.args = args

    def get_contexts(self, retrieved_docs: List[Dict]) -> str:
        """
        格式化检索到的文档为上下文

        Args:
            retrieved_docs: 检索到的文档列表

        Returns:
            str: 格式化后的上下文字符串
        """
        contexts = []
        for i, doc in enumerate(retrieved_docs):
            text = doc["text"]
            contexts.append(("Passage-%d: " % i) + text)

        return "\n".join(contexts)

    def get_contexts_score(self, retrieved_docs: List[Dict]) -> str:
        """
        格式化检索到的文档为上下文
        与get_contexts不同，此方法在每个Passage后添加truthful score
        Args:
            retrieved_docs: 检索到的文档列表

        Returns:
            str: 格式化后的上下文字符串
        """
        contexts = []
        for i, doc in enumerate(retrieved_docs):
            text = doc["text"]
            score = doc["credibility"]
            contexts.append(("Passage-%d: " % i) + text + (f'\nTruthful score: {score}\n'))

        return "\n".join(contexts)

    def create_mistral_prompt(self, question: str, retrieved_docs: List[Dict]) -> str:
        """
        创建符合Mistral格式的prompt

        Args:
            question: 问题文本
            retrieved_docs: 检索到的文档列表

        Returns:
            List[Dict]: 聊天格式的prompt
        """
        if retrieved_docs:
            contexts = self.get_contexts(retrieved_docs)
            prompt = f'Given the following information: \n{contexts}\nplease only output the answer to the question.\nQuestion:{question}\nthe correct answer is:'
        else:
            print("norag")
            prompt = f'Answer the following question based on your internal knowledge with one or few words.\nQuestion: {question}\nthe correct answer is:'

        return prompt

    def create_mistral_prompt_score(self, question: str, retrieved_docs: List[Dict]) -> str:
        """
        创建符合Mistral格式的prompt

        Args:
            question: 问题文本
            retrieved_docs: 检索到的文档列表

        Returns:
            List[Dict]: 聊天格式的prompt
        """
        if retrieved_docs:
            contexts = self.get_contexts_score(retrieved_docs)
            prompt = f'You are an assistant capable of answering questions based on provided passages. Each passage is assigned a truthfulness score (0-10), where a higher score indicates greater credibility. You should consider truthfulness score of the passage, if the score is low, you should not trust it. Given the following information: \n{contexts}\nAnswer the following question based on the given information or your internal knowledge with one or few words without the source (just output a answer, don\'t output anything else).\nQuestion: {question}\nthe correct answer is:'
        else:
            prompt = f'Answer the following question based on your internal knowledge with one or few words.\nQuestion: {question}\nthe correct answer is:'

        return prompt

def parse_generated_answer_chat_format(answer):

    if "answer is" in answer:
        idx = answer.find("answer is")
        answer = answer[idx+len("answer is"): ].strip()
        if answer.startswith(":"):
            answer = answer[1:].strip()
    return answer

def parse_gemma_mistral_answer(answer):

    candidate_answers = answer.split("\n")
    answer = ""
    i = 0
    while len(answer) < 1 and i<len(candidate_answers):
        answer = candidate_answers[i].strip()
        i += 1
    answer = parse_generated_answer_chat_format(answer)
    return answer


# ============ Llama3评估函数 ============
def evaluate_with_llama3(args, model, tokenizer, data):
    """
    使用Llama3模型评估检索结果

    Args:
        args: 参数配置
        model: Llama3模型
        tokenizer: Llama3 tokenizer
        data: 测试数据

    Returns:
        Dict: 评估指标
    """
    em_scores_list, f1_scores_list = [], []
    processor = Llama3DataProcessor(args)
    retrieved_docs = None
    print(f"开始评估 {len(data)} 个样本...")

    for i, example in enumerate(tqdm(data, desc="正在评估")):
        question = example["question"]
        gold_answers = example["answers"]
        ctxs = example["ctxs"]
        if args.norag:
            prompt = processor.create_llama3_prompt(question, retrieved_docs)
        else:
            if args.prompt_based:
                retrieved_docs = retrieve_documents_by_similarity_score(question, ctxs, args)
                prompt = processor.create_llama3_prompt_score(question, retrieved_docs)
            else:
                retrieved_docs = retrieve_documents_by_similarity(question, ctxs, args)
                prompt = processor.create_llama3_prompt(question, retrieved_docs)

        # 将聊天格式转换为模型输入
        input_text = tokenizer.apply_chat_template(
            prompt,
            add_generation_prompt=True,
            tokenize=False
        )

        # 编码输入
        encoded = tokenizer(
            [input_text],
            truncation=True,
            padding=True,
            return_tensors="pt"
        ).to(device)

        model_inputs = {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"]
        }

        # 生成答案
        with torch.no_grad():
            outputs = model.generate(
                **model_inputs,
                max_new_tokens=args.answer_maxlength,
                do_sample=False,
                temperature=1.0,
                use_cache=True
            )

        # 解码生成的答案
        generated_ids = outputs[0, model_inputs["input_ids"].shape[1]:].detach().cpu()
        predicted_answer = tokenizer.decode(generated_ids, skip_special_tokens=True)

        print(f"问题: {question}")
        print(f"预测答案: {predicted_answer}")
        print("-" * 50)

        # 计算评估指标
        em_score = ems(predicted_answer, gold_answers)
        em_scores_list.append(em_score)

        if not em_score and i < 5:  # 只打印前5个错误案例
            print(f"\n错误案例 {i + 1}:")
            print(f"问题: {question}")
            print(f"预测答案: {predicted_answer}")
            print(f"正确答案: {gold_answers}")
            print("-" * 50)

        f1, precision, recall = f1_score(predicted_answer, gold_answers[0])
        f1_scores_list.append(f1)

    # 计算最终指标
    metrics = {
        "exact_match": np.mean(em_scores_list),
        "f1": np.mean(f1_scores_list)
    }

    return metrics


# ============ Gemma评估函数 ============
def evaluate_with_gemma(args, model, tokenizer, data):
    """
    使用Gemma模型评估检索结果

    Args:
        args: 参数配置
        model: Gemma模型
        tokenizer: Gemma tokenizer
        data: 测试数据

    Returns:
        Dict: 评估指标
    """
    em_scores_list, f1_scores_list = [], []
    processor = GemmaDataProcessor(args)
    retrieved_docs = None
    print(f"开始评估 {len(data)} 个样本...")

    for i, example in enumerate(tqdm(data, desc="正在评估")):
        question = example["question"]
        gold_answers = example["answers"]
        ctxs = example["ctxs"]
        if args.norag:
            messages = processor.create_gemma_prompt(question, retrieved_docs)
        else:
            if args.prompt_based:
                retrieved_docs = retrieve_documents_by_similarity_score(question, ctxs, args)
                messages = processor.create_gemma_prompt_score(question, retrieved_docs)
            else:
                retrieved_docs = retrieve_documents_by_similarity(question, ctxs, args)
                messages = processor.create_gemma_prompt(question, retrieved_docs)

        # 编码输入
        encoded = tokenizer(
            messages,
            truncation=True,
            padding=True,
            return_tensors="pt"
        ).to(device)

        model_inputs = {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"]
        }

        # 生成答案
        with torch.no_grad():
            outputs = model.generate(
                **model_inputs,
                max_new_tokens=args.answer_maxlength,
                do_sample=False,
                temperature=1.0,
                use_cache=True
            )

        # 解码生成的答案
        generated_ids = outputs[0, model_inputs["input_ids"].shape[1]:].detach().cpu()
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        predicted_answer = parse_gemma_mistral_answer(generated_text)

        print(f"问题: {question}")
        print(f"预测答案: {predicted_answer}")
        print("-" * 50)

        # 计算评估指标
        em_score = ems(predicted_answer, gold_answers)
        em_scores_list.append(em_score)

        if not em_score and i < 5:  # 只打印前5个错误案例
            print(f"\n错误案例 {i + 1}:")
            print(f"问题: {question}")
            print(f"预测答案: {predicted_answer}")
            print(f"正确答案: {gold_answers}")
            print("-" * 50)

        f1, precision, recall = f1_score(predicted_answer, gold_answers[0])
        f1_scores_list.append(f1)

    # 计算最终指标
    metrics = {
        "exact_match": np.mean(em_scores_list),
        "f1": np.mean(f1_scores_list)
    }

    return metrics

# ============ Mistral评估函数 ============
def evaluate_with_mistral(args, model, tokenizer, data):
    """
    使用Mistral模型评估检索结果

    Args:
        args: 参数配置
        model: Mistral模型
        tokenizer: Mistral tokenizer
        data: 测试数据

    Returns:
        Dict: 评估指标
    """
    em_scores_list, f1_scores_list = [], []
    processor = MistralDataProcessor(args)
    retrieved_docs = None
    print(f"开始评估 {len(data)} 个样本...")

    for i, example in enumerate(tqdm(data, desc="正在评估")):
        question = example["question"]
        gold_answers = example["answers"]
        ctxs = example["ctxs"]
        if args.norag:
            messages = processor.create_mistral_prompt(question, retrieved_docs)
        else:
            if args.prompt_based:
                retrieved_docs = retrieve_documents_by_similarity_score(question, ctxs, args)
                messages = processor.create_mistral_prompt_score(question, retrieved_docs)
            else:
                retrieved_docs = retrieve_documents_by_similarity(question, ctxs, args)
                messages = processor.create_mistral_prompt(question, retrieved_docs)

        # 编码输入
        encoded = tokenizer(
            messages,
            truncation=True,
            padding=True,
            return_tensors="pt"
        ).to(device)

        model_inputs = {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"]
        }

        # 生成答案
        with torch.no_grad():
            outputs = model.generate(
                **model_inputs,
                max_new_tokens=args.answer_maxlength,
                do_sample=False,
                temperature=1.0,
                use_cache=True
            )

        # 解码生成的答案
        generated_ids = outputs[0, model_inputs["input_ids"].shape[1]:].detach().cpu()
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        predicted_answer = parse_gemma_mistral_answer(generated_text)

        print(f"问题: {question}")
        print(f"预测答案: {predicted_answer}")
        print("-" * 50)

        # 计算评估指标
        em_score = ems(predicted_answer, gold_answers)
        em_scores_list.append(em_score)

        if not em_score and i < 5:  # 只打印前5个错误案例
            print(f"\n错误案例 {i + 1}:")
            print(f"问题: {question}")
            print(f"预测答案: {predicted_answer}")
            print(f"正确答案: {gold_answers}")
            print("-" * 50)

        f1, precision, recall = f1_score(predicted_answer, gold_answers[0])
        f1_scores_list.append(f1)

    # 计算最终指标
    metrics = {
        "exact_match": np.mean(em_scores_list),
        "f1": np.mean(f1_scores_list)
    }

    return metrics

# ============ CAG-7B模型加载 ============
def load_cag_model_tokenizer(args):
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        padding_side="left"
    )
    if device == "cuda":
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
    model.eval()
    return model, tokenizer


# ============ 数据处理 ============
class CagDataProcessor:
    """单跳数据处理器"""

    def __init__(self, args):
        self.args = args

    def format_documents(self, retrieved_docs: List[Dict]) -> str:
        """
        格式化检索到的文档，包含可信度分数

        Args:
            retrieved_docs: 检索到的文档列表

        Returns:
            str: 格式化后的文档字符串
        """
        formatted_docs = []

        for i, doc in enumerate(retrieved_docs):
            score = doc["credibility"]
            text = doc["text"]
            if score <= 3:
                credibility = "Low credibility of text"
            elif score > 3 and score < 7:
                credibility = "Medium credibility of text"
            elif score >= 7:
                credibility = "High credibility of text"
            # 格式化文档：包含可信度分数和内容
            formatted_doc = f"{credibility}: {text} "
            formatted_docs.append(formatted_doc)

        return "\n".join(formatted_docs)

    def create_prompt(self, question: str, retrieved_docs: List[Dict]) -> str:
        """
        创建符合指定格式的prompt

        Args:
            question: 问题文本
            retrieved_docs: 检索到的文档列表

        Returns:
            List[Dict]: 聊天格式的prompt
        """


        user_input = '''You are an assistant who can answer questions based on the given passages. Each passage has a credibility score that indicates the relevance and accuracy of the passage to the question. Your answer need to combine multiple passages and their credibility.Question:{question}\nDocs:{paras}\n\nYour answer should based on the given information or your internal knowledge with one or few words without the source  (just output a answer, don\'t output anything else). Answer:'''

        if retrieved_docs:
            paras = self.format_documents(retrieved_docs)
            user_input = user_input.format(question=question, paras=paras)
        else:
            user_input = f"Question: {question}\n\nAnswer (one or few words only):"

        return user_input


def parse_cag_answer(answer):
    if "Answer:" in answer:
        idx = answer.find("Answer:")
        answer = answer[idx + len("Answer:"):].strip()
        if answer.startswith(":"):
            answer = answer[1:].strip()
    return answer


# ============ 评估函数 ============
def evaluate_with_cag(args, cag_tokenizer, cag_model, data):
    """
    评估单跳检索结果

    Args:
        args: 参数配置
        cag_tokenizer: CAG模型的tokenizer
        cag_model: CAG模型
        data: 测试数据

    Returns:
        Dict: 评估指标
    """
    em_scores_list, f1_scores_list, accuracy_list = [], [], []
    processor = CagDataProcessor(args)

    print(f"开始评估 {len(data)} 个样本...")

    for i, example in enumerate(tqdm(data, desc="正在评估")):
        question = example["question"]
        gold_answers = example["answers"]
        ctxs = example["ctxs"]

        # 执行单跳检索
        retrieved_doc = retrieve_documents_by_similarity_score(question, ctxs, args)

        # 创建prompt
        prompt = processor.create_prompt(question, retrieved_doc)



        # 编码输入
        encoded = cag_tokenizer(
            prompt,
            truncation=True,
            padding=True,
            return_tensors="pt"
        ).to(device)

        model_inputs = {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"]
        }
        # 生成答案
        with torch.no_grad():
            outputs = cag_model.generate(
                **model_inputs,
                max_new_tokens=args.answer_maxlength,
                do_sample=False,
                temperature=1.0
            )

        # 解码生成的答案
        generated_ids = outputs[0, model_inputs["input_ids"].shape[1]:].detach().cpu()
        generated_text = cag_tokenizer.decode(generated_ids, skip_special_tokens=True)
        predicted_answer = generated_text.strip()
        predicted_answer = parse_cag_answer(predicted_answer)
        print(predicted_answer)
        # 计算评估指标
        acc = accuracy(predicted_answer, gold_answers)
        accuracy_list.append(acc)
        # if not acc and i < 5:  # 只打印前5个案例
        #     print(f"\n错误案例 {i + 1}:")
        #     print(f"问题: {question}")
        #     print(f"预测答案: {predicted_answer}")
        #     print(f"正确答案: {gold_answers}")
        #     print("-" * 50)
        em_score = ems(predicted_answer, gold_answers)
        em_scores_list.append(em_score)
        if not em_score and i < 5:  # 只打印前5个案例
            print(f"\n错误案例 {i + 1}:")
            print(f"问题: {question}")
            print(f"预测答案: {predicted_answer}")
            print(f"正确答案: {gold_answers}")
            print("-" * 50)

        f1, precision, recall = f1_score(predicted_answer, gold_answers[0])
        f1_scores_list.append(f1)

    metrics = {
        "exact_match": np.mean(em_scores_list),
        "f1": np.mean(f1_scores_list),
        "accuracy": np.mean(accuracy_list),
    }

    return metrics


# ============ 参数设置 ============
def setup_parser():
    """设置命令行参数"""
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input_data_file", type=str, default="data/hotpotqa/dev_with_kgs.json",
                        help="输入数据文件路径")
    parser.add_argument("--model_type", type=str, choices=["cag", "llama3", "gemma","mistral"], default="llama3",
                        help="模型类型：cag表示CAG-7B，llama3表示Llama3-8B-Instruct")
    parser.add_argument("--model_path", type=str, required=True, help="CAG-7B模型路径")
    parser.add_argument("--context_nums", type=int, default=5, help="检索的文档数量")
    parser.add_argument("--answer_maxlength", type=int, default=25, help="答案最大长度")
    parser.add_argument("--fake_num", type=int, default=1)
    parser.add_argument("--prompt_based", action="store_true",
                        help="Run prompt based")
    parser.add_argument("--norag", action="store_true",
                        help="Run inference without context")
    parser.add_argument("--exclusion", action="store_true",
                        help="This option will filter documents with credibility scores below the threshold and perform inference.")
    args = parser.parse_args()
    return args


# ============ 主函数 ============
def main():
    """主函数"""
    # 设置参数
    args = setup_parser()

    print("=" * 80)
    print("=" * 80)
    print(f"检索文档数: {args.context_nums}")
    print(f"模型路径: {args.model_path}")
    print("=" * 80)

    # 加载测试数据
    print("步骤1: 加载测试数据...")
    data = load_json(args.input_data_file)
    print(f"数据集大小: {len(data)} 个样本")
    if args.model_type == "llama3":
        model, tokenizer = load_llama3_model_tokenizer(args.model_path)
        print("步骤3: 开始评估（使用Llama3）...")
        metrics = evaluate_with_llama3(args, model, tokenizer, data)
    elif args.model_type == "gemma":
        model, tokenizer = load_gemma_model_tokenizer(args.model_path)
        print("步骤3: 开始评估（使用Gemma-7B）...")
        metrics = evaluate_with_gemma(args, model, tokenizer, data)
    elif args.model_type == "mistral":
        model, tokenizer = load_mistral_model_tokenizer(args.model_path)
        print("步骤3: 开始评估（使用Mistral-7B）...")
        metrics = evaluate_with_mistral(args, model, tokenizer, data)
    else:
        # 加载CAG-7B模型
        print("步骤2: 加载CAG-7B模型...")
        cag_model, cag_tokenizer = load_cag_model_tokenizer(args)

        # 执行评估
        print("步骤3: 开始检索和评估...")
        metrics = evaluate_with_cag(args, cag_tokenizer, cag_model, data)

    # 输出结果
    print("\n" + "=" * 80)
    print("评估结果:")
    print("=" * 80)
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    print("=" * 80)


if __name__ == "__main__":
    main()
