import json
input_file="/home/jiangjp/trace-idea/data/musique/musique_test1000_add_truthful_scores_with_kgs9.json"
try:
    with open(input_file, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
except Exception as e:
    print(f"读取输入文件失败: {e}")
textsdefaults=[]
triplesdefailt=[]
count=0
for itemid,item in enumerate(dataset):
    count+=1
    for i, ctx in enumerate(item['ctxs']):
        if ctx['text_truthful_score']==12:
            textsdefaults.append({
                "itemid":itemid,
                "itemstr":item['id'],
                "ctxid":i,
                "text":ctx['text']})
        for j, triple in enumerate(ctx['triples']):
            if triple['triple_truthful_score']==12:
                triplesdefailt.append({
                    "itemid":itemid,
                    "itemstr": item['id'],
                    "ctxid":ctx['id'],
                    "tripleid":j,
                    "triple":triple})
print(count)
print(textsdefaults)
print("=========================================================================")
for triple in triplesdefailt:
    print(triple)
