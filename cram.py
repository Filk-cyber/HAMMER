import os
import json
import argparse
import numpy as np
import torch
from typing import Dict, List
from tqdm import tqdm
from retrievers.e5_mistral import get_e5_mistral_embeddings_for_query, get_e5_mistral_embeddings_for_document
from readers.metrics import ems, f1_score,accuracy
from re_weighting import Re_Weighting_Strategy, Find_Best_Heads

# 全局变量
device = "cuda" if torch.cuda.is_available() else "cpu"


def load_json(file_path):
    """加载JSON文件"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def retrieve_documents_by_similarity_for_find_best_heads(question: str, ctxs: List[Dict], args):
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
    ctxs_copy = ctxs[:end_index]
    for i, ctx in enumerate(ctxs_copy):
        documents.append(ctx["text"])

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
    topk_scores, topk_indices = torch.topk(similarities, k=min(args.context_nums - 1, len(documents)), dim=0)

    # 构建检索结果
    retrieved_documents = []
    scores = []
    retrieved_documents.append(ctxs[end_index]["text"])
    scores.append(0)
    for i, idx in enumerate(topk_indices.tolist()):
        retrieved_documents.append(documents[idx])
        scores.append(1)
    return retrieved_documents, scores


def retrieve_documents_by_similarity_for_re_weighting(question: str, ctxs: List[Dict], args,
                                                      ideal_setting: bool = True):
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
    print("ideal_setting:",ideal_setting)
    documents, real_documents, real_documents_truthful_scores = [], [], []
    end_index = len(ctxs) - 3
    ctxs_copy = ctxs[:end_index]
    for i, ctx in enumerate(ctxs_copy):
        documents.append(ctx["text"])
        real_documents.append(ctx["text"])
        if ideal_setting:
            real_documents_truthful_scores.append(10)
        else:
            # 获取文档的truthful_score
            real_documents_truthful_scores.append(ctx["text_truthful_score"])
    for ctx in ctxs[end_index:end_index + args.fake_num]:
        documents.append(ctx["text"])
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
    topk_scores, topk_indices = torch.topk(similarities[:-args.fake_num], k=min(args.context_nums - args.fake_num, len(real_documents)), dim=0)

    # 构建检索结果
    retrieved_documents = []
    truthful_scores=[]
    relevant_scores=[]
    for i, ctx in enumerate(ctxs[end_index:end_index + args.fake_num]):
        if ideal_setting:
            retrieved_documents.append(ctx["text"])
            truthful_scores.append(1)
        else:
            retrieved_documents.append(ctx["text"])
            truthful_scores.append(ctx["text_truthful_score"])
            relevant_scores.append(similarities[end_index + i].item())

    for i, idx in enumerate(topk_indices.tolist()):
        retrieved_documents.append(real_documents[idx])
        truthful_scores.append(real_documents_truthful_scores[idx])
        relevant_scores.append(topk_scores[i].item())

    scores = np.array([int(score) for score in truthful_scores])
    if np.max(scores) != 0:
        scores = (scores / np.max(scores))
    else:
        scores = (scores + 1)
    combined = list(zip(scores, retrieved_documents))
    combined_sorted = sorted(combined, key=lambda x: x[0])
    scores, contexts = zip(*combined_sorted)
    scores = list(scores)
    contexts = list(contexts)
    return contexts, scores


def cram(args, model_path: str, output_dir: str = "./results_heads_scores"):
    chat_model_re_weighting = None
    print("args.cram_type", args.cram_type)
    print(type(args.cram_type))
    print(args.cram_type == "find_best_heads")
    data = load_json(args.input_data_file)
    if "Llama-3" in model_path:
        llm = "llama3"
    elif "Qwen" in model_path:
        llm = 'qwen'
    elif "gemma" in model_path:
        llm = 'gemma'
    elif "Mistral" in model_path:
        llm = 'Mistral'
    if args.cram_type == "find_best_heads":
        chat_model_re_weighting = Find_Best_Heads(model_name=model_path)
        output_dir = os.path.join(output_dir, args.datasets)
        file_path = os.path.join(output_dir, llm, f"heads_scores.json")
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding="utf8") as file:
                all_prob_changes = json.load(file)
        else:
            all_prob_changes = []
        for i, example in enumerate(tqdm(data, desc="正在记录注意力头分数")):
            if i < len(all_prob_changes):
                continue
            question = example["question"]
            ctxs = example["ctxs"]
            wrong_answer = example["wrong_answer"]
            contexts,scores = retrieve_documents_by_similarity_for_find_best_heads(question, ctxs, args)
            prob_change = chat_model_re_weighting.cal_logits(question=question, paras=contexts, scores=scores,
                                           wrong_answer=wrong_answer)
            all_prob_changes.append(prob_change)

            folder = os.path.join(output_dir, llm)
            file_path = os.path.join(output_dir, llm, f"heads_scores.json")
            if not os.path.exists(folder):
                os.makedirs(folder)
            with open(file_path, 'w', encoding="utf8") as file:
                json.dump(all_prob_changes, file, ensure_ascii=False, indent=4)

    else:
        input_path = os.path.join(f'./results_heads_scores', args.datasets, llm)
        input_filepath = os.path.join(input_path, 'selected_heads.json')
        with open(input_filepath, "r", encoding="utf-8") as f:
            layers_to_be_modified = json.load(f)
        chat_model_re_weighting = Re_Weighting_Strategy(model_name=model_path,
                                                        layers_to_be_modified=layers_to_be_modified)
        em_scores_list, f1_scores_list, accuracy_list = [], [], []
        for i, example in enumerate(tqdm(data, desc="正在进行修改注意力头评估：")):
            question = example["question"]
            ctxs = example["ctxs"]
            wrong_answer = example["wrong_answer"]
            gold_answers = example["answers"]
            contexts,scores = retrieve_documents_by_similarity_for_re_weighting(question, ctxs, args)
            prompt, predicted_answer=chat_model_re_weighting.run_RAG_with_attention_weighting(question=question, paras=contexts, scores=scores)
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
            "accuracy": np.mean(accuracy_list)
        }

        return metrics

def setup_parser():
    """设置命令行参数"""
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input_data_file", type=str, default="data/hotpotqa/dev_with_kgs.json",
                        help="输入数据文件路径")
    parser.add_argument("--datasets", type=str, default="hotpotqa",
                        help="数据集类型")
    parser.add_argument("--cram_type", type=str, default="find_best_heads",
                        help="cram type")
    parser.add_argument("--model_path", type=str, required=True, help="llama3-8b-instruct、qwen-7b模型路径")
    parser.add_argument("--context_nums", type=int, default=5, help="检索的文档数量")
    parser.add_argument("--answer_maxlength", type=int, default=25, help="答案最大长度")
    parser.add_argument("--fake_num", type=int, default=1)
    args = parser.parse_args()
    return args

def main():
    """主函数"""
    # 设置参数
    args = setup_parser()
    if args.cram_type == "find_best_heads":
        cram(args, args.model_path)
    else:
        metrics = cram(args, args.model_path)
        # 输出结果
        print("\n" + "=" * 80)
        print("评估结果:")
        print("=" * 80)
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")
        print("=" * 80)

if __name__ == "__main__":
    main()