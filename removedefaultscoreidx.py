import json
from copy import deepcopy

# 输入和输出文件路径
input_file = "/home/jiangjp/trace-idea/data/musique/musique_test1000_add_truthful_scores_with_kgs9.json"
output_file = "/home/jiangjp/trace-idea/data/musique/musique_test1000_add_truthful_scores_with_kgs_filtered.json"

try:
    with open(input_file, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
except Exception as e:
    print(f"读取输入文件失败: {e}")
    exit(1)

textsdefaults = []
triplesdefailt = []
count = 0

# 收集需要删除的信息
for itemid, item in enumerate(dataset):
    count += 1
    for i, ctx in enumerate(item['ctxs']):
        if ctx['text_truthful_score'] == 12:
            textsdefaults.append({
                "itemid": itemid,
                "itemstr": item['id'],
                "ctxid": i,
                "text": ctx['text']
            })
        for j, triple in enumerate(ctx['triples']):
            if triple['triple_truthful_score'] == 12:
                triplesdefailt.append({
                    "itemid": itemid,
                    "itemstr": item['id'],
                    "ctxid": ctx['id'],
                    "tripleid": j,
                    "triple": triple
                })

print(f"总共处理了 {count} 个项目")
print(f"找到 {len(textsdefaults)} 个需要删除的文本")
print(f"找到 {len(triplesdefailt)} 个需要删除的三元组")

# 创建新的数据集副本
new_dataset = deepcopy(dataset)

# 收集需要删除的ctxid（按itemid分组）
ctx_to_remove = {}
for triple in triplesdefailt:
    itemid = triple['itemid']
    ctxid = triple['ctxid']

    if itemid not in ctx_to_remove:
        ctx_to_remove[itemid] = set()
    ctx_to_remove[itemid].add(ctxid)

print(f"需要删除的ctx分布: {dict(ctx_to_remove)}")

# 删除对应的ctx
removed_count = 0
for itemid, ctx_ids in ctx_to_remove.items():
    if itemid < len(new_dataset):
        original_ctx_count = len(new_dataset[itemid]['ctxs'])

        # 过滤掉需要删除的ctx
        new_dataset[itemid]['ctxs'] = [
            ctx for ctx in new_dataset[itemid]['ctxs']
            if ctx['id'] not in ctx_ids
        ]

        new_ctx_count = len(new_dataset[itemid]['ctxs'])
        removed_count += (original_ctx_count - new_ctx_count)

        print(f"Item {itemid}: 删除了 {original_ctx_count - new_ctx_count} 个ctx，剩余 {new_ctx_count} 个ctx")

print(f"总共删除了 {removed_count} 个ctx")

# 保存到新的JSON文件
try:
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(new_dataset, f, ensure_ascii=False, indent=2)
    print(f"已成功保存到文件: {output_file}")
except Exception as e:
    print(f"保存文件失败: {e}")

# 验证结果
print("\n验证结果:")
print(f"原始数据集大小: {len(dataset)}")
print(f"新数据集大小: {len(new_dataset)}")

# 统计每个item的ctx数量变化
total_original_ctx = sum(len(item['ctxs']) for item in dataset)
total_new_ctx = sum(len(item['ctxs']) for item in new_dataset)
print(f"原始ctx总数: {total_original_ctx}")
print(f"新ctx总数: {total_new_ctx}")
print(f"删除的ctx总数: {total_original_ctx - total_new_ctx}")