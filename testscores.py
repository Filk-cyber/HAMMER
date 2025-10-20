import json
from tqdm import tqdm
input_file="/home/jiangjp/trace-idea/data/hotpotqa/hotpotqa_dev100_add_truthful_scores_with_kgs_optimized.json"
try:
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
except Exception as e:
    print(f"读取输入文件失败: {e}")
ideal_setting=True
fake_num=4
for example in tqdm(data[:1], desc="Generating reasoning chains", total=len(data[:1])):
    question = example["question"]
    triples, triple_positions = [], []

    end_index = len(example["ctxs"]) - (3 - fake_num)
    ctxs = example["ctxs"][:end_index]
    for i, ctx in enumerate(ctxs):
        truthful_scores = []
        for triple_item in ctx["triples"]:
            triples.append("<{}; {}; {}>".format(triple_item["head"], triple_item["relation"], triple_item["tail"]))
            triple_positions.append(triple_item["position"])
            if ideal_setting:
                if i < (end_index - fake_num):
                    truthful_scores.append(10)
                else:
                    truthful_scores.append(1)
            else:
                truthful_scores.append(triple_item["triple_truthful_score"])
        print("i: ",i,"scores:",truthful_scores)
        print("=========================================")
