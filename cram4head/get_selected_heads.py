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
    
    output_path = os.path.join(input_path, f"selected_heads.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(selected_heads_dict, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    find_top_k_heads(input_path="results_heads_scores/musique/Mistral")
