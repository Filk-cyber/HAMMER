import os
import logging
import argparse
import numpy as np
from tqdm import tqdm
import json

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import logging as hf_logging

hf_logging.set_verbosity_error()

from readers.datasets import ReaderDatasetWithChains
from readers.collators import CollatorWithChainsChatFormat, CollatorWithChains
from readers.metrics import ems, f1_score

from utils.const import *
from utils.utils import seed_everything, setup_logger, to_device

logger = logging.getLogger(__file__)

READER_NAME_TO_MODEL_NAME_MAP = {
    "llama3": "/home/jiangjp/models/Meta-Llama-3-8B-Instruct",
    "mistral": "mistralai/Mistral-7B-v0.1",
    "gemma": "google/gemma-7b",
}

COLLATOR_MAP = {
    "llama3": CollatorWithChainsChatFormat,
    "mistral": CollatorWithChains,
    "gemma": CollatorWithChains,
}


def setup_parser():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # 数据集参数
    parser.add_argument("--reader", type=str, default="llama3")
    parser.add_argument("--text_maxlength", type=int, default=4096)
    parser.add_argument("--answer_maxlength", type=int, default=25)
    parser.add_argument("--test_file", type=str, required=True,
                        help="输入文件路径，如 chain_w0.json, chain_w1.json 等")
    parser.add_argument("--n_context", type=int, default=None)
    parser.add_argument("--context_type", type=str, default=None)

    # 实验参数
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_dir", type=str, default="checkpoint")
    parser.add_argument("--save_IE_file", type=str, default="checkpoint")
    parser.add_argument("--name", type=str, default="llama3")
    # 权重参数 - 从文件名自动解析或手动指定
    parser.add_argument("--weight_w", type=float, default=None,
                        help="权重值，如果不指定则从文件名自动解析")

    args = parser.parse_args()
    assert args.reader in COLLATOR_MAP.keys()

    # 如果没有手动指定权重，尝试从文件名解析
    if args.weight_w is None:
        try:

            filename = os.path.basename(args.test_file)

            # 寻找 "_w" 模式，然后提取权重值
            if "_w" in filename:
                # 找到 "_w" 的位置
                w_index = filename.find("modify_w")
                # 从 "_w" 之后开始提取
                w_part = filename[w_index + 8:]  # 跳过 "_w"

                # 提取数字部分（直到遇到非数字字符或文件扩展名）
                w_digits = ""
                for char in w_part:
                    if char.isdigit():
                        w_digits += char
                    else:
                        break

                if w_digits:
                    args.weight_w = float(f"{w_digits[0]}.{w_digits[1]}")
                    print(f"从文件名自动解析得到权重: w = {args.weight_w}")
                else:
                    args.weight_w = 1.0  # 默认值
                    print(f"无法从文件名解析出数字权重，使用默认值: w = {args.weight_w}")
            else:
                args.weight_w = 1.0  # 默认值
                print(f"文件名中未找到权重标识(_w)，使用默认值: w = {args.weight_w}")
        except Exception as e:
            args.weight_w = 1.0
            print(f"权重解析失败 ({e})，使用默认值: w = {args.weight_w}")
    return args


def load_tokenizer(model_name_or_path, padding_side="left"):
    print(f"正在加载分词器: \"{model_name_or_path}\"，padding_side: \"{padding_side}\"")
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, padding_side=padding_side, token=HF_TOKEN)
    if tokenizer.pad_token is None or tokenizer.pad_token_id is None:
        print("缺少padding token，设置padding token为eos_token")
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer


def parse_generated_answer_chat_format(answer):
    if "answer is" in answer:
        idx = answer.find("answer is")
        answer = answer[idx + len("answer is"):].strip()
        if answer.startswith(":"):
            answer = answer[1:].strip()
    return answer


def parse_generated_answer(answer):
    candidate_answers = answer.split("\n")
    answer = ""
    i = 0
    while len(answer) < 1 and i < len(candidate_answers):
        answer = candidate_answers[i].strip()
        i += 1
    answer = parse_generated_answer_chat_format(answer)
    return answer


PARSE_FUNCTION_MAP = {
    "llama3": parse_generated_answer_chat_format,
    "mistral": parse_generated_answer,
    "gemma": parse_generated_answer
}


def get_single_answer_log_probability(model, tokenizer, input_ids, attention_mask, answer_text):
    """
    计算模型生成单个答案的对数概率
    """
    # 将答案进行分词
    try:
        answer_tokens = tokenizer(answer_text, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
    except:
        return float('-inf')

    if len(answer_tokens) == 0:
        return float('-inf')

    # 准备输入
    current_input_ids = input_ids[0:1].clone()
    current_attention_mask = attention_mask[0:1].clone()

    total_log_prob = 0.0

    # 强制解码
    with torch.no_grad():
        for i, target_token_id in enumerate(answer_tokens):
            # 模型前向传播
            outputs = model(input_ids=current_input_ids, attention_mask=current_attention_mask)
            next_token_logits = outputs.logits[:, -1, :]

            # 转换为对数概率
            log_probs = torch.nn.functional.log_softmax(next_token_logits, dim=-1)

            # 获取目标token的对数概率
            token_log_prob = log_probs[0, target_token_id].item()
            total_log_prob += token_log_prob

            # 更新输入序列
            target_token_tensor = target_token_id.unsqueeze(0).unsqueeze(0).to(current_input_ids.device)
            current_input_ids = torch.cat([current_input_ids, target_token_tensor], dim=1)

            # 更新注意力掩码
            new_attention = torch.ones((1, 1), device=current_attention_mask.device)
            current_attention_mask = torch.cat([current_attention_mask, new_attention], dim=1)

    return total_log_prob


def get_multiple_answers_probability(model, tokenizer, input_ids, attention_mask, correct_answers):
    """
    计算模型对多个正确答案的概率分布

    Args:
        model: 语言模型
        tokenizer: 分词器
        input_ids: 输入token IDs
        attention_mask: 注意力掩码
        correct_answers: 正确答案列表，如 ["Barack Obama", "Obama", "President Obama"]

    Returns:
        dict: 包含每个答案的概率信息和总体统计信息
    """

    # 处理输入的答案格式
    if isinstance(correct_answers, str):
        answer_list = [correct_answers]
    elif isinstance(correct_answers, list):
        answer_list = [str(ans).strip() for ans in correct_answers if str(ans).strip()]
    else:
        return {
            'individual_probs': {},
            'individual_log_probs': {},
            'max_prob': 0.0,
            'max_log_prob': float('-inf'),
            'best_answer': '',
            'total_prob': 0.0,
            'answer_count': 0
        }

    if not answer_list:
        return {
            'individual_probs': {},
            'individual_log_probs': {},
            'max_prob': 0.0,
            'max_log_prob': float('-inf'),
            'best_answer': '',
            'total_prob': 0.0,
            'answer_count': 0
        }

    individual_log_probs = {}
    individual_probs = {}

    # 计算每个答案的概率
    for answer in answer_list:
        log_prob = get_single_answer_log_probability(model, tokenizer, input_ids, attention_mask, answer)
        individual_log_probs[answer] = log_prob

        if log_prob != float('-inf'):
            individual_probs[answer] = np.exp(log_prob)
        else:
            individual_probs[answer] = 0.0

    # 找到最佳答案（概率最高的）
    if individual_probs:
        best_answer = max(individual_probs.keys(), key=lambda k: individual_probs[k])
        max_prob = individual_probs[best_answer]
        max_log_prob = individual_log_probs[best_answer]
    else:
        best_answer = answer_list[0] if answer_list else ''
        max_prob = 0.0
        max_log_prob = float('-inf')

    # 计算所有正确答案的总概率（在某些情况下可能有用）
    valid_probs = [prob for prob in individual_probs.values() if prob > 0]
    total_prob = sum(valid_probs) if valid_probs else 0.0

    return {
        'individual_probs': individual_probs,
        'individual_log_probs': individual_log_probs,
        'max_prob': max_prob,
        'max_log_prob': max_log_prob,
        'best_answer': best_answer,
        'total_prob': total_prob,
        'answer_count': len(answer_list),
        'valid_answer_count': len(valid_probs)
    }


def evaluate_single_weight(args, tokenizer, dataloader, model):
    """
    评估单个权重值，收集概率分布信息（支持多个正确答案）
    """

    em_scores_list = []
    f1_scores_list = []
    precision_scores_list = []
    recall_scores_list = []
    sample_info_list = []

    model.eval()

    print(f"正在评估权重 w = {args.weight_w}")
    print(f"输入文件: {args.test_file}")
    print(f"数据集大小: {len(dataloader.dataset)}")

    for batch_idx, batch_inputs in tqdm(dataloader, desc=f"评估 w={args.weight_w}", total=len(dataloader)):
        batch_inputs = to_device(batch_inputs, DEVICE)

        # 生成答案
        batch_outputs = model.generate(
            **batch_inputs,
            max_new_tokens=args.answer_maxlength,
            do_sample=False,
            temperature=1.0
        )
        batch_generated_token_ids = batch_outputs[:, batch_inputs["input_ids"].shape[1]:].detach().cpu()

        # 处理每个样本
        batch_size = batch_inputs["input_ids"].shape[0]
        for i in range(batch_size):

            # 解析生成的答案
            generated_tokens = batch_generated_token_ids[i]
            generated_answer = tokenizer.decode(generated_tokens, skip_special_tokens=True)
            parsed_answer = PARSE_FUNCTION_MAP[args.reader](generated_answer)

            # 获取标准答案
            sample_idx = batch_idx[i]
            try:
                example = dataloader.dataset.get_example(sample_idx)
                gold_answers = example["answers"]

                # 确保gold_answers是列表格式
                if isinstance(gold_answers, str):
                    gold_answers_list = [gold_answers]
                elif isinstance(gold_answers, list):
                    gold_answers_list = gold_answers
                else:
                    gold_answers_list = [str(gold_answers)]

            except Exception as e:
                print(f"警告：无法获取样本 {sample_idx} 的标准答案: {e}")
                continue

            # 计算评估指标
            em_score = ems(parsed_answer, gold_answers)
            em_scores_list.append(em_score)

            # 对于F1分数，我们使用第一个正确答案作为参考
            reference_answer = gold_answers_list[0] if gold_answers_list else ""
            f1, precision, recall = f1_score(parsed_answer, reference_answer)
            f1_scores_list.append(f1)
            precision_scores_list.append(precision)
            recall_scores_list.append(recall)

            # 计算所有正确答案的概率分布
            sample_input_ids = batch_inputs["input_ids"][i:i + 1]
            sample_attention_mask = batch_inputs["attention_mask"][i:i + 1]

            prob_info = get_multiple_answers_probability(
                model=model,
                tokenizer=tokenizer,
                input_ids=sample_input_ids,
                attention_mask=sample_attention_mask,
                correct_answers=gold_answers_list
            )

            # 存储样本详细信息
            sample_info_list.append({
                'sample_idx': int(sample_idx),  # 确保是int
                'generated_answer': parsed_answer,
                'gold_answers': gold_answers_list,
                'reference_answer': reference_answer,
                'em_score': float(em_score),  # 确保是float
                'f1_score': float(f1),  # 确保是float
                'precision': float(precision),  # 确保是float
                'recall': float(recall),  # 确保是float

                # 概率信息 - 确保都是Python原生类型
                'individual_probs': {k: float(v) for k, v in prob_info['individual_probs'].items()},
                'individual_log_probs': {k: float(v) if v != float('-inf') else -999999
                                         for k, v in prob_info['individual_log_probs'].items()},
                'max_prob': float(prob_info['max_prob']),
                'max_log_prob': float(prob_info['max_log_prob']) if prob_info['max_log_prob'] != float(
                    '-inf') else -999999,
                'best_answer': str(prob_info['best_answer']),
                'total_prob': float(prob_info['total_prob']),
                'answer_count': int(prob_info['answer_count']),
                'valid_answer_count': int(prob_info['valid_answer_count']),
                'weight_w': float(args.weight_w)
            })

            # 打印错误案例和概率信息
            if not em_score:
                print(f"样本{sample_idx} 错误:")
                print(f"  生成答案: '{parsed_answer}'")
                print(f"  正确答案: {gold_answers_list}")
                print(f"  最佳正确答案概率: {prob_info['max_prob']:.6f} ('{prob_info['best_answer']}')")
            else:
                # 对于正确的样本，也可以打印概率信息（可选）
                if sample_idx % 50 == 0:  # 每50个样本打印一次
                    print(f"样本{sample_idx} 正确:")
                    print(f"  生成答案: '{parsed_answer}'")
                    print(f"  最佳正确答案概率: {prob_info['max_prob']:.6f} ('{prob_info['best_answer']}')")

    # 计算整体指标
    valid_max_log_probs = [info['max_log_prob'] for info in sample_info_list if info['max_log_prob'] != float('-inf')]
    valid_max_probs = [info['max_prob'] for info in sample_info_list if info['max_prob'] > 0]
    valid_total_probs = [info['total_prob'] for info in sample_info_list if info['total_prob'] > 0]

    metrics = {
        "weight_w": args.weight_w,
        "input_file": args.test_file,
        "total_samples": len(sample_info_list),
        "exact_match": np.mean(em_scores_list),
        "f1": np.mean(f1_scores_list),
        "precision": np.mean(precision_scores_list),
        "recall": np.mean(recall_scores_list),

        # 最佳答案的概率统计
        "avg_max_log_prob": np.mean(valid_max_log_probs) if valid_max_log_probs else float('-inf'),
        "avg_max_prob": np.mean(valid_max_probs) if valid_max_probs else 0.0,
        "avg_total_prob": np.mean(valid_total_probs) if valid_total_probs else 0.0,

        # 有效样本统计
        "valid_max_prob_samples": len(valid_max_probs),
        "invalid_max_prob_samples": len(sample_info_list) - len(valid_max_probs),
        "avg_answer_count": np.mean([info['answer_count'] for info in sample_info_list]),
        "avg_valid_answer_count": np.mean([info['valid_answer_count'] for info in sample_info_list])
    }

    return metrics, sample_info_list


if __name__ == "__main__":

    args = setup_parser()
    seed_everything(args.seed)

    # 创建保存目录
    checkpoint_path = os.path.join(args.save_dir, args.name)
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path, exist_ok=True)

    log_file = os.path.join(checkpoint_path, f"evaluation_w_{args.weight_w}.log")
    setup_logger(-1, log_file)

    print(f"权重值: {args.weight_w}")
    print(f"输入文件: {args.test_file}")
    print(f"日志文件: {log_file}")

    # 加载数据集
    print("正在加载数据集...")
    dataset = ReaderDatasetWithChains(
        data_path=args.test_file,
        n_context=args.n_context,
        chain_key="chains"
    )

    # 加载分词器和整理器
    model_name_or_path = READER_NAME_TO_MODEL_NAME_MAP[args.reader]
    tokenizer = load_tokenizer(model_name_or_path, padding_side="left")
    collator = COLLATOR_MAP[args.reader](
        tokenizer=tokenizer,
        text_maxlength=args.text_maxlength,
        answer_maxlength=args.answer_maxlength,
        context_type=args.context_type
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        drop_last=False,
        shuffle=False,
        collate_fn=collator
    )

    # 加载模型
    print("正在加载模型...")
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, token=HF_TOKEN)
    model.to(DEVICE)

    # 执行评估
    print("开始评估...")
    metrics, sample_info = evaluate_single_weight(args, tokenizer, dataloader, model)

    # 输出结果
    print("\n" + "=" * 60)
    print("评估结果")
    print("=" * 60)
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key}: {value:.6f}")
        else:
            print(f"{key}: {value}")
    print("=" * 60)

    # 记录到日志
    logger.info("评估完成")
    logger.info("权重值: {}".format(args.weight_w))
    logger.info("输入文件: {}".format(args.test_file))
    logger.info("评估结果: {}".format(metrics))

    # 保存详细结果
    output_file = os.path.join(args.save_IE_file, f"sample_correct_probabilities_w_{args.weight_w}_{args.context_type}.json")

    # 处理无法序列化的值
    serializable_info = []
    for info in sample_info:
        new_info = info.copy()

        # 处理对数概率中的无穷大值
        if new_info['max_log_prob'] == float('-inf'):
            new_info['max_log_prob'] = -999999

        # 处理individual_log_probs中的无穷大值
        new_individual_log_probs = {}
        for ans, log_prob in new_info['individual_log_probs'].items():
            if log_prob == float('-inf'):
                new_individual_log_probs[ans] = -999999
            else:
                new_individual_log_probs[ans] = log_prob
        new_info['individual_log_probs'] = new_individual_log_probs

        serializable_info.append(new_info)

    # 保存到文件
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            'metrics': metrics,
            'sample_info': serializable_info
        }, f, ensure_ascii=False, indent=2)

    print(f"\n结果已保存至: {output_file}")
    logger.info(f"结果已保存至: {output_file}")

    # 打印一些统计信息
    print(f"\n统计信息:")
    print(f"平均每个问题有 {metrics['avg_answer_count']:.1f} 个正确答案")
    print(f"平均每个问题有 {metrics['avg_valid_answer_count']:.1f} 个有效概率的答案")
    print(f"有效概率样本: {metrics['valid_max_prob_samples']} / {metrics['total_samples']}")