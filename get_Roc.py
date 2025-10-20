import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import auc

def extract_scores_and_labels(datasets, score_key):
    scores = []
    labels = []
    for dataset in datasets:
        for item in dataset:
            ctxs = item["ctxs"]
            n = len(ctxs)
            for idx, ctx in enumerate(ctxs):
                # 标定text标签
                if idx < n - 3:
                    text_label = 1  # 正样本
                else:
                    text_label = 0  # 负样本

                if score_key == "text_truthful_score":
                    score = ctx.get("text_truthful_score", 0)
                    scores.append(score)
                    labels.append(text_label)
                elif score_key == "triple_truthful_score":
                    for triple in ctx.get("triples", []):
                        triple_score = triple.get("triple_truthful_score", 0)
                        scores.append(triple_score)
                        labels.append(text_label)  # triple 的标签与文本一致
    return np.array(scores), np.array(labels)

def manual_roc_curve(scores, labels, thresholds):
    tpr_list = []
    fpr_list = []
    for thresh in thresholds:
        pred = (scores >= thresh).astype(int)
        tp = np.sum((pred == 1) & (labels == 1))
        fp = np.sum((pred == 1) & (labels == 0))
        fn = np.sum((pred == 0) & (labels == 1))
        tn = np.sum((pred == 0) & (labels == 0))
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        tpr_list.append(tpr)
        fpr_list.append(fpr)
    return np.array(fpr_list), np.array(tpr_list)

if __name__ == "__main__":
    dataset_files = [
        "/home/jiangjp/trace-idea/data/hotpotqa/hotpotqa_test1000_add_truthful_scores_with_kgs_final.json",
        "/home/jiangjp/trace-idea/data/2wikimultihopqa/wiki_test1000_add_truthful_scores_with_kgs_final.json",
        "/home/jiangjp/trace-idea/data/musique/musique_test1000_add_truthful_scores_with_kgs_final.json"
    ]

    datasets = []
    for file_name in dataset_files:
        with open(file_name, "r", encoding="utf-8") as f:
            datasets.append(json.load(f))

    for score_key, key_name, color in [("text_truthful_score", "Text", "#1f77b4"), ("triple_truthful_score", "Triple", "#ff7f0e")]:
        scores, labels = extract_scores_and_labels(datasets, score_key)
        thresholds = np.arange(0, 11, 1)
        fpr, tpr = manual_roc_curve(scores, labels, thresholds)

        fpr = np.concatenate(([0], fpr, [1]))
        tpr = np.concatenate(([0], tpr, [1]))

        order = np.argsort(fpr)
        fpr = fpr[order]
        tpr = tpr[order]

        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color=color, lw=2, label=f'AUC = {roc_auc:.2f}')
        plt.plot([0, 1], [0, 1], 'k--', lw=1)
        plt.xlabel('False Positive Rate (FPR)')
        plt.ylabel('True Positive Rate (TPR)')
        plt.title(f'ROC Curve ({key_name})')
        plt.legend(loc="lower right")
        plt.xlim(0, 1)  # 让x轴0和1与边界重合
        plt.ylim(0, 1)  # 让y轴0和1与边界重合（推荐加上）
        plt.grid(True)  # 添加网格
        plt.tight_layout()
        plt.savefig(f'roc_curve_{key_name.lower()}_with_true_label_manual_int_threshold.png')
        plt.show()
        print(f"All datasets {key_name} AUC: {roc_auc:.4f}")