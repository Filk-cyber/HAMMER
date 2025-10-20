import json
input_file="/home/jiangjp/trace-idea/data/hotpotqa/hotpotqa_dev100_add_truthful_scores_with_kgs.json"
try:
    with open(input_file, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
except Exception as e:
    print(f"读取输入文件失败: {e}")

maxtriples=0
max_idx=0
max_ctx_idx=0
for idx, item in enumerate(dataset):
    if idx==0:
        print(item["question"])
    for ctx_idx, ctx in enumerate(item['ctxs']):
        triplescnt=0
        for triple_idx, triple in enumerate(ctx['triples']):
            triplescnt += 1
        if triplescnt > maxtriples:
            maxtriples = triplescnt
            max_idx=idx
            max_ctx_idx=ctx_idx
print(maxtriples)
print(max_idx)
print(max_ctx_idx)
