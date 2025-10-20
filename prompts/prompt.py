from .prompts_templates import *


def get_prompt(context: list = [], question: str = '', answer: str = '', type: str = 'with_contexts', scores: list = []):
    prompt = prompt_dict['qa'][type]
    if type == 'with_contexts'or type == 'with_contexts_gemma_mistral':
        paras = ''
        final_paras = []
        for i, para in enumerate(context):
            final_paras.append(("Passage-%d: " % i) + para)
        paras = "\n".join(final_paras)
        prompt = prompt.format(question=question, paras=paras, answer=answer)
    elif type == 'with_contexts_score' or type == 'with_contexts_score_llama2':
        paras = ''
        final_paras = []
        for i, para in enumerate(context):
            final_paras.append(("Passage-%d: " % i) + para + (f'Truthful socre: {scores[i]}\n'))
        paras = "\n".join(final_paras)
        prompt = prompt.format(question=question, paras=paras, answer=answer)
    elif type == 'with_contexts_cag':
        paras = ''
        final_paras = []
        for i, para in enumerate(context):
            if scores[i] <= 3:
                credibility = "Low credibility of text"
            elif scores[i] > 3 and scores[i] < 7:
                credibility = "Medium credibility of text"
            elif scores[i] >= 7:
                credibility = "High credibility of text"
            final_paras.append(f"{credibility}: {para} ")
        paras = "\n".join(final_paras)
        prompt = prompt.format(question=question, paras=paras, answer=answer)
    elif type == 'without_contexts':
        prompt = prompt.format(question=question)
    return prompt


def get_prompt_truthful_judge(context: str = '', question: str = ''):
    prompt = truthful_judge.format(para=context, question=question)
    return prompt
