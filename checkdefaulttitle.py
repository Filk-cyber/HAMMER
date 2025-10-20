import json
input_file="/home/jiangjp/trace-idea/data/hotpotqa/hotpotqa_test1000_add_ctxs.json"
try:
    with open(input_file, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
except Exception as e:
    print(f"读取输入文件失败: {e}")
titledefaults=[]
flag=False
for itemid,item in enumerate(dataset):
    for i, ctx in enumerate(item['ctxs'][-3:]):
        if ctx['title']=="DEFAULT_TITLE_PLACEHOLDER"or not ctx['text'].startswith("CNN News:"):
            titledefaults.append({
                "itemid":itemid,
                "ctxid":ctx['id'],
                "text":ctx['text']})
            flag=True


for item in titledefaults:
    print(item)
print(flag)
print("=========================================================================")
