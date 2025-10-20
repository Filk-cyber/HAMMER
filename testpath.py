# import os
# input_path = os.path.join(f'./results_heads_scores', "hotpotqa", "llama3")
# print(input_path)
# list1=["a","b","c","d","e","f"]
# fake_num=2
# ideal_setting=True
# end_index = len(list1) - (3 - fake_num)
# for i,a in enumerate(list1[:end_index]):
#     if ideal_setting:
#         if i < (end_index - fake_num):
#             print("this is i",i,"and the item is: ",a," and the score is :",10)
#         else:
#             print("this is i", i, "and the item is: ", a, " and the score is :", 1)
#     else:
#         print("this is truthful score: xxxxxxxxx")

import matplotlib.font_manager as fm

# 修正后的代码
fonts = [f.name for f in fm.fontManager.ttflist if 'zh' in f.name.lower() or 'chinese' in f.name.lower() or 'song' in f.name.lower() or 'kai' in f.name.lower()]
print("可用的中文字体:")
for font in set(fonts):
    print(f"  - {font}")

import torch

A = torch.tensor([[1, 2], [3, 4]])
B = torch.tensor([[5, 6], [7, 8]])

C = A * B  # 逐元素乘
# 或 C = torch.mul(A, B)
import numpy as np
print(C)
thresholds = np.arange(0, 11, 1)
print(thresholds)
print("===================")
model_name="/home/jiangjp/models/gemma-7b/"
if model_name.find('gemma') != -1:
    print(model_name.find('gemma'))
    print("yes")

print("===================")
pppp=[1,2,3,4,5,6,7]
print(pppp[-3:])
n=len(pppp)
print(pppp[:max(0, n-3)])

strpred="poo ws"
strgold=["wxs"]
if strgold[0] in strpred:
    print("yes")
else:
    print("no")
ems_list=[False,False,False,False,False]
s=np.mean(ems_list)
ss=max(ems_list)
print(s)
print(ss)
from readers.metrics import normalize_answer
def accuracy(prediction, ground_truths):
    """
    只要 prediction 中包含 ground_truths 里的任意一个元素（归一化后），就算正确。
    """
    norm_pred = normalize_answer(prediction)
    for gt in ground_truths:
        if normalize_answer(gt) in norm_pred:
            return True
    return False
print(accuracy(strpred,strgold))
import numpy as np
score=[9.9,9.2,8.1,7.322]
scores = np.round(score, 2)
print(scores)
paths=[[1,2,3,4],[0,9,2]]
new_paths=[]
new_paths.append(paths[0]+[5])
print(new_paths)
print("You are an assistant capable of answering questions based on provided passages. Each passage is assigned a truthfulness score (0-10), where a higher score indicates greater credibility. You should consider truthfulness score of the passage, if the score is low, you should not trust it. Given the following information: \nPassage-0: CNN news: Recent studies have shown that eukaryotes have approximately 10,000 origins of DNA replication, rather than the previously believed number of up to 100,000. This finding is based on extensive research and analysis conducted by leading scientists in the field. The discovery of around 10,000 origins of DNA replication in eukaryotes has significant implications for our understanding of genetic processes and cellular functions. It suggests that the replication of DNA in eukaryotic cells is more tightly regulated and controlled than previously thought. This new information challenges previous assumptions and opens up new avenues for further research and exploration in the field of molecular biology.Truthful socre: 1\n\nPassage-1: is more")
print("===================")
instruction=""
prompt="123"
print("{}{}\nthe correct answer is:".format(instruction, prompt))
print("===================")
print(torch.__version__)
print(torch.version.cuda)
print(torch.backends.cudnn.version())
indice2triple={"a":"ss",0:"sss",1:"sssaa"}
print(indice2triple[1])
print(indice2triple["a"])
import pandas as pd
table1_path = '/home/jiangjp/trace-idea/data/cram_ourmethod_fakenum(Mistral-7B).xlsx'
df = pd.read_excel(table1_path, sheet_name="HotPotQA")
df1 = df.iloc[:, 0]
print(df1)
