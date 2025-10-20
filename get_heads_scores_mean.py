import os
import json
import numpy as np
def casual_tracing_combine_all(input_path: str = "results_heads_scores/hotpotqa/llama3"):
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

if __name__ == "__main__":
    casual_tracing_combine_all(input_path="results_heads_scores/musique/Mistral")