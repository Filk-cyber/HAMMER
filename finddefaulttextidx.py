import json
input_file="/home/jiangjp/trace-idea/data/2wikimultihopqa/test1000_optimized5.json"
try:
    with open(input_file, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
except Exception as e:
    print(f"读取输入文件失败: {e}")
textsdefaults=[]
triplesdefailt=[]
for itemid,item in enumerate(dataset):
    for i, ctx in enumerate(item['ctxs']):
        if ctx['text_truthful_score']==12:
            textsdefaults.append({
                "itemid":itemid,
                "ctxid":ctx['id'],
                "text":ctx['text']})
        # for j, triple in enumerate(ctx['triples']):
        #     if triple['triple_truthful_score']==12:
        #         triplesdefailt.append({
        #             "itemid":itemid,
        #             "ctxid":i,
        #             "tripleid":j,
        #             "triple":triple})

for item in textsdefaults:
    print(item)
print("=========================================================================")
# print(triplesdefailt)