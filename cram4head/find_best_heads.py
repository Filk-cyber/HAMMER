import os
import fire

# os.environ["CUDA_VISIBLE_DEVICES"] = "5,7"
# import torch
from tqdm import tqdm
import json
import numpy as np

from .re_weighting_modify import Find_Best_Heads


def casual_tracing_per_head(
        filepath: str = "nq_100_bge.json",
        LLM: str = "Meta-Llama-3-8B-Instruct",
        output_dir: str = "datasets/nq",
        contexts_type: list = ["ori_fake", "reranked_dense"],
        # contexts_type: list = ["reranked_dense"],
        topk: int = 4):
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if "Llama-2-13b" in LLM:
        model_name = "llama2_13b_chat"
    elif "Llama-3" in LLM:
        model_name = "llama3"
    elif "Qwen" in LLM:
        model_name = 'qwen'

    file_path = os.path.join(output_dir, model_name, f"heads_scores.json")
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding="utf8") as file:
            all_prob_changes = json.load(file)
    else:
        all_prob_changes = []

    model = Find_Best_Heads(model_name=LLM)
    for idx, sample in enumerate(tqdm(data)):
        if idx < len(all_prob_changes):
            continue
        contexts = []
        scores = []
        if "ori_fake" in contexts_type:
            contexts.extend(sample["ori_fake"])
            scores.extend([0])
        if "reranked_dense" in contexts_type:
            contexts.extend(sample["reranked_dense_ctxs"][:topk])
            scores.extend([1, 1, 1, 1])

        prob_change = model.cal_logits(question=sample["question"], paras=contexts, scores=scores, wrong_answer=sample["wrong answer"])
        all_prob_changes.append(prob_change)

        folder = os.path.join(output_dir, model_name)
        file_path = os.path.join(output_dir, model_name, f"heads_scores.json")
        if not os.path.exists(folder):
            os.makedirs(folder)
        with open(file_path, 'w', encoding="utf8") as file:
            json.dump(all_prob_changes, file, ensure_ascii=False, indent=4)


def casual_tracing_per_head_with_position(
        filepath: str = "nq_100_bge.json",
        LLM: str = "Meta-Llama-3-8B-Instruct",
        output_dir: str = "datasets/nq",
        contexts_type: list = ["ori_fake", "reranked_dense"],
        # contexts_type: list = ["reranked_dense"],
        topk: int = 4):
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if "Llama-2-13b" in LLM:
        model_name = "llama2_13b_chat"
    elif "Llama-3" in LLM:
        model_name = "llama3"

    all_prob_changes = []
    model = Find_Best_Heads(model_name=LLM)
    for sample in tqdm(data):
        all_prob_changes_position = []
        ori_fake = sample["ori_fake"]
        reranked_dense_ctxs = sample["reranked_dense_ctxs"][:topk]

        for position in range(5):
            contexts = []
            scores = []
            
            if "reranked_dense" in contexts_type:
                contexts.extend(reranked_dense_ctxs[:position])
                scores.extend([1] * position)

            
            if "ori_fake" in contexts_type:
                contexts.extend(ori_fake)
                scores.append(0)

            
            if "reranked_dense" in contexts_type:
                contexts.extend(reranked_dense_ctxs[position:])
                scores.extend([1] * (4 - position))

    
            prob_change = model.cal_logits(question=sample["question"], paras=contexts, scores=scores, wrong_answer=sample["wrong answer"])
            all_prob_changes_position.append(prob_change)
        all_prob_changes.append(all_prob_changes_position)
        folder = os.path.join(output_dir, model_name)
        file_path = os.path.join(output_dir, model_name, f"heads_scores_pos.json")
        if not os.path.exists(folder):
            os.makedirs(folder)
        with open(file_path, 'w', encoding="utf8") as file:
            json.dump(all_prob_changes, file, ensure_ascii=False, indent=4)


def casual_tracing_combine_all(input_path: str = "datasets/nq/Llama-3"):
    file_path = os.path.join(input_path, f"heads_scores.json")
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    all_data = np.array(data)
    mean_data = np.mean(all_data, axis=0)
    flat_mean_data = mean_data.flatten()
    sorted_indices = np.argsort(flat_mean_data)[::-1]  
    sorted_values = flat_mean_data[sorted_indices]  


    sorted_2d_indices = np.unravel_index(sorted_indices, mean_data.shape)

    #将scores即之前和更改后的概率差值展平成一个一维列表进行从高到低排序，[(分数，(行索引，列索引))] for for i in range(len(sorted_values)
    sorted_data_with_indices = [(sorted_values[i], (sorted_2d_indices[0][i].item(), sorted_2d_indices[1][i].item())) for i in range(len(sorted_values))]

    output_path = os.path.join(input_path, f"heads_scores_mean.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(sorted_data_with_indices, f, ensure_ascii=False, indent=4)
    return sorted_data_with_indices


def find_top_k_heads(input_path: str = "datasets/triviaqa", topk: int = 10):
    file_path = os.path.join(input_path, f"heads_scores_mean.json")
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    selected_heads = [x[1] for x in data[:topk]]
    selected_heads_dict = dict()
    for x in selected_heads:
        if x[0] not in selected_heads_dict:
            selected_heads_dict[x[0]] = [x[1]]
        else:
            selected_heads_dict[x[0]].append(x[1])
    # 原始格式：[(5, 7), (2, 3), (5, 1), (8, 2), (2, 0)]
    # 转换后：
    '''
    {
        "5": [7, 1],  # 第5层的第7个和第1个头
        "2": [3, 0],  # 第2层的第3个和第0个头
        "8": [2]  # 第8层的第2个头
    }
    '''
    output_path = os.path.join(input_path, f"selected_heads.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(selected_heads_dict, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    fire.Fire(find_top_k_heads)
