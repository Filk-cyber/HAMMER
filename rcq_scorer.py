# =================================================================
# 文件: rcq_scorer.py (新创建)
# 职责: 实现RCQ-Score四大指标的计算逻辑。
# =================================================================
import re
from collections import defaultdict
import torch


# 这是一个辅助函数，从三元组字符串中提取头实体和尾实体
def _get_entities_from_triple(triple_str: str) -> set:
    """内部辅助函数，从'<h; r; t>'格式中提取头实体和尾实体"""
    try:
        content = triple_str.strip('<>')
        head, _, tail = content.rsplit(';', 2)
        return {head.strip(), tail.strip()}
    except ValueError:
        return set()


# ==================================================
# 第二部分：结构连贯性 (Coherence)
# ==================================================
def get_coherence_score(chain: list) -> float:
    """计算单条推理链的结构连贯性得分。"""
    triples = [item['triple'] for item in chain]
    if len(triples) <= 1:
        return 1.0

    connected_pairs = 0
    num_pairs = len(triples) - 1

    for i in range(num_pairs):
        entities_i = _get_entities_from_triple(triples[i])
        entities_i_plus_1 = _get_entities_from_triple(triples[i + 1])
        if entities_i and entities_i_plus_1 and not entities_i.isdisjoint(entities_i_plus_1):
            connected_pairs += 1

    return connected_pairs / num_pairs if num_pairs > 0 else 1.0


# ==================================================
# 第三部分：问题相关性 (Relevance)
# ==================================================
def get_relevance_score(chain: list, question: str, get_embedding_func) -> float:
    """
    (重构版) 计算推理链与问题的全局语义相关性。
    接收一个外部的嵌入函数作为参数，以重用 e5_mistral。
    """
    if not chain:
        return 0.0

    chain_text = ". ".join([item['triple'].strip('<>').replace(';', ' ') for item in chain])

    try:
        question_embedding = get_embedding_func([question])
        chain_embedding = get_embedding_func([chain_text])
        cosine_score = torch.matmul(question_embedding, chain_embedding.T)
        return max(0.0, cosine_score.item())
    except Exception as e:
        print(f"在 get_relevance_score 中计算嵌入时出错: {e}")
        return 0.0


# ==================================================
# 第四部分：逻辑完整性 (Completeness) - 改进版
# ==================================================
def get_completeness_score(chain: list, question: str, answer: str, get_embedding_func) -> float:
    """
    (改进版) 融合“首尾呼应”和“问题类型驱动”来评估逻辑完整性。
    """
    if not chain or not answer:
        return 0.0

    question_lower = question.lower()
    is_comparison = ' or ' in question_lower or 'which' in question_lower or 'compare' in question_lower

    if is_comparison:
        entities_in_question = set(re.findall(r'"([^"]*)"|\b[A-Z][a-zA-Z]+\b', question))
        if len(entities_in_question) >= 2:
            chain_text = " ".join([item['triple'] for item in chain])
            found_entities_count = sum(1 for entity in entities_in_question if entity in chain_text)
            if found_entities_count >= 2:
                return 1.0
            elif found_entities_count == 1:
                return 0.2
            else:
                return 0.0

    last_triple_str = chain[-1]['triple']
    last_triple_entities = _get_entities_from_triple(last_triple_str)
    answer_alignment_score = 1.0 if answer in last_triple_str or not last_triple_entities.isdisjoint(
        set(answer.split())) else 0.0

    start_alignment_score = 0.5
    try:
        first_triple_str = chain[0]['triple']
        q_embedding = get_embedding_func([question])
        first_triple_embedding = get_embedding_func([first_triple_str])
        start_alignment_score = max(0.0, torch.matmul(q_embedding, first_triple_embedding.T).item())
    except Exception as e:
        print(f"在 get_completeness_score 中计算嵌入时出错: {e}")
        start_alignment_score = 0.5

    final_score = (0.66 * answer_alignment_score + 0.34 * start_alignment_score)
    return final_score


# ==================================================
# 第五部分：事实一致性 (Consistency)
# ==================================================
def get_consistency_score(chain: list) -> float:
    """通过检测内部属性冲突来近似评估事实一致性。"""
    if not chain:
        return 1.0

    entity_attributes = defaultdict(dict)
    conflicts = 0

    for item in chain:
        try:
            content = item['triple'].strip('<>')
            head, relation, tail = content.rsplit(';', 2)
            head, relation, tail = head.strip(), relation.strip(), tail.strip()

            if relation in entity_attributes[head] and entity_attributes[head][relation] != tail:
                conflicts += 1
            else:
                entity_attributes[head][relation] = tail
        except ValueError:
            continue

    consistency_score = 0.9 ** conflicts
    return consistency_score


# ==================================================
# 总分计算函数
# ==================================================
def calculate_rcq_score(chain: list, question: str, answer: str, get_embedding_func) -> dict:
    """计算一条链的所有RCQ分数，并将嵌入函数传递下去。"""
    s_coh = get_coherence_score(chain)
    s_rel = get_relevance_score(chain, question, get_embedding_func)
    s_com = get_completeness_score(chain, question, answer, get_embedding_func)
    s_con = get_consistency_score(chain)

    # 最终加权总分，权重可以根据实验调整
    total_score = 0.4 * s_coh + 0.3 * s_rel + 0.2 * s_com + 0.1 * s_con

    return {
        "total": total_score,
        "coherence": s_coh,
        "relevance": s_rel,
        "completeness": s_com,
        "consistency": s_con
    }