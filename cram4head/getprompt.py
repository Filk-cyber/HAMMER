from .prompts_templates import *
def get_prompt(context: list = [], question: str = '', answer: str = '', type: str = 'with_contexts', scores: list = []):
    prompt = prompt_dict['qa'][type]
    if type == 'with_contexts':
        paras = ''
        final_paras = []
        for i, para in enumerate(context):
            final_paras.append(("Passage-%d: " % i) + para)
        paras = "\n".join(final_paras)
        prompt = prompt.format(question=question, paras=paras, answer=answer)

    return prompt
