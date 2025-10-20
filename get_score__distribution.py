import json
import matplotlib.pyplot as plt
from collections import defaultdict

def stat_score_distribution(file_paths):
    # 用于统计分数分布
    text_err_score = defaultdict(int)
    text_wiki_score = defaultdict(int)
    triple_err_score = defaultdict(int)
    triple_wiki_score = defaultdict(int)

    for file_path in file_paths:
        with open(file_path, 'r', encoding='utf-8') as fin:
            dataset = json.load(fin)
        for item in dataset:
            ctxs = item.get('ctxs', [])
            n = len(ctxs)
            # 错误文档: ctxs[-3:]
            for ctx in ctxs[-3:]:
                score = ctx.get('text_truthful_score', -1)
                if 0 <= score <= 10:
                    text_err_score[score] += 1
                for triple in ctx.get('triples', []):
                    t_score = triple.get('triple_truthful_score', -1)
                    if 0 <= t_score <= 10:
                        triple_err_score[t_score] += 1
            # 维基百科文档: ctxs[0 : n-3],即第1个到倒数第4个
            for ctx in ctxs[:max(0, n-3)]:
                score = ctx.get('text_truthful_score', -1)
                if 0 <= score <= 10:
                    text_wiki_score[score] += 1
                for triple in ctx.get('triples', []):
                    t_score = triple.get('triple_truthful_score', -1)
                    if 0 <= t_score <= 10:
                        triple_wiki_score[t_score] += 1

    # 画图
    def draw(score_dict, title, xlabel, ylabel, filename, bar_color='skyblue'):
        x = list(range(0, 11))
        y = [score_dict.get(i, 0) for i in x]
        plt.figure()
        plt.bar(x, y, color=bar_color)
        plt.title(title, x=0.489)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel, labelpad=1)
        plt.xticks(x)
        plt.savefig(filename)
        plt.close()

    # Text分布
    draw(text_err_score,
         "Credibility scores distribution of misinformation docs",
         "Credibility scores", "Count",
         "text_error_score_dist.png",
         bar_color='skyblue')
    draw(text_wiki_score,
         "Credibility scores distribution of Wikipedia docs",
         "Credibility scores", "Count",
         "text_wiki_score_dist.png",
         bar_color='#90EE90')  # 学术浅绿色
    # Triple分布
    draw(triple_err_score,
         "Credibility scores distribution of triples extracted from misinformation docs",
         "Credibility scores", "Count",
         "triple_error_score_dist.png",
         bar_color='skyblue')
    draw(triple_wiki_score,
         "Credibility scores distribution of triples extracted from Wikipedia docs",
         "Credibility scores", "Count",
         "triple_wiki_score_dist.png",
         bar_color='#90EE90')  # 学术浅绿色

if __name__ == '__main__':
    # 请将下面的路径替换成你自己的数据集文件路径
    file_paths = [
        "/home/jiangjp/trace-idea/data/hotpotqa/hotpotqa_test1000_add_truthful_scores_with_kgs_final.json",
        "/home/jiangjp/trace-idea/data/2wikimultihopqa/wiki_test1000_add_truthful_scores_with_kgs_final.json",
        "/home/jiangjp/trace-idea/data/musique/musique_test1000_add_truthful_scores_with_kgs_final.json"
    ]
    stat_score_distribution(file_paths)