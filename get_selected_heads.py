import os
import json
def find_top_k_heads(input_path: str = "results_heads_scores/hotpotqa/llama3", topk: int = 894):
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
    find_top_k_heads(input_path="results_heads_scores/musique/Mistral")