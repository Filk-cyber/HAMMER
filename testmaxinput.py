import json
import time
from zhipuai import ZhipuAI
client = ZhipuAI(api_key="05748176082447f483438dfd914cc299.NcsCifhTarCch6es")  # 请填写您自己的APIKey
triple_instruction = """Your task is to evaluate the authenticity of knowledge triplets based on your internal knowledge, reasoning, and inference. The structure of a knowledge triplet is ⟨ head; relation; tail⟩， Represents a single factual statement about the relationship between entities. I will provide a knowledge triad that may contain accurate information or fictional errors. You need to assign it a credibility score from 0 to 10, with higher scores indicating higher authenticity and lower scores indicating lower authenticity. Here are 2 examples, you should follow the output format below:
##########
Triple:
head: Albert Einstein
relation: was the first recipient in 1921 of
tail: the Nobel Prize in Physics

Analysis:
Head Accuracy: Albert Einstein is correct. Einstein was a real historical figure and a renowned physicist.
Tail Accuracy: the Nobel Prize in Physics is valid. This award exists and has been granted since 1901.
Relation Errors: Albert Einstein was not the first Nobel laureate in Physics. The inaugural prize was awarded in 1901 to Wilhelm Conrad Roentgen for his discovery of X-rays. Einstein did receive the Nobel Prize, but it was awarded retrospectively in 1922 (for the year 1921) and recognized his work on the photoelectric effect, not relativity.

Credibility Score: 0


Triple:
head: Wilhelm Conrad Roentgen
relation: was the first recipient in 1901 of
tail: the Nobel Prize in Physics

Analysis:
Head Accuracy: Wilhelm Conrad Roentgen is correct. Roentgen was a German physicist who discovered X-rays.
Tail Accuracy: the Nobel Prize in Physics is factual and well-documented.
Relation Accuracy: Roentgen was indeed the first laureate in Physics, as confirmed by Nobel Prize archives. The inaugural award was correctly granted in 1901. The prize honored his discovery of X-rays, which revolutionized medicine and physics.

Credibility Score: 10
##########
"""
text_input_template="""Passage:
{text}

Credibility Score: 
"""
triple_input_template = """Triple:
head: {head}
relation: {relation}
tail: {tail}

Credibility Score: 
"""
text_instruction = """Your task is to evaluate the authenticity of a text based on your internal knowledge. Specifically, I will provide you with a passage that may contain accurate information or fabricated errors. Using your own knowledge, reason, and deduction, you are to assign a credibility score ranging from 0 to 10, where a higher score indicates greater authenticity and a lower score suggests lesser authenticity. 
Here are 2 examples, you should follow the output format below:
##########
Passage:
In a groundbreaking discovery, researchers have found that Albert Einstein was the first recipient of the Nobel Prize in Physics. According to newly uncovered documents, Einstein's pioneering work in theoretical physics, particularly his theory of relativity, was recognized by the Nobel Committee in 1921. This revelation challenges the long-held belief that Marie Curie was the first Nobel laureate in physics, and solidifies Einstein's place as one of the greatest minds in scientific history.

Analysis:
1. Albert Einstein as the First Nobel Prize Recipient in Physics: This is incorrect. The first Nobel Prize in Physics was awarded in 1901, not to Albert Einstein, but to Wilhelm Conrad Röntgen for the discovery of X-rays.
2. Einstein's Nobel Prize Recognition: Albert Einstein was indeed awarded the Nobel Prize in Physics in 1921, but not for his theory of relativity. He received it for his discovery of the photoelectric effect, which was instrumental in the development of quantum theory.
3. Marie Curie as the First Nobel Laureate in Physics: This is also incorrect. Marie Curie was a Nobel laureate, but she was not the first to win the Nobel Prize in Physics. Her first Nobel Prize was in Physics in 1903, shared with her husband Pierre Curie and Henri Becquerel for their work on radioactivity. Marie Curie was, notably, the first woman to win a Nobel Prize, and the first person to win Nobel Prizes in two different scientific fields (Physics and Chemistry).
4. Implication about the Nobel Committee's Recognition of Relativity: As mentioned, Einstein's Nobel Prize was not for relativity, despite its profound impact on physics. The Nobel Committee specifically avoided awarding the prize for relativity at the time due to ongoing debates and lack of experimental confirmation of the theory during that period.

Credibility Score: 0


Passage:
The first Nobel Prize in Physics was awarded to Wilhelm Conrad Roentgen in 1901. Roentgen received the Nobel Prize for his discovery of X-rays, which had a significant impact on the field of physics and medicine

Analysis:
The facts presented in the statement you provided are largely accurate.

Credibility Score: 10
##########
"""
user_input_list=[]
input_file="/home/jiangjp/trace-idea/data/hotpotqa/hotpotqa_dev100_add_truthful_scores_with_kgs.json"
try:
    with open(input_file, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
except Exception as e:
    print(f"读取输入文件失败: {e}")
# for i, ctx in enumerate(dataset[0]['ctxs']):
#     user_input_list.append(text_input_template.format(text=ctx['text']))
for i ,ctx in enumerate(dataset[0]['ctxs']):
    for j,triple in enumerate(ctx['triples'][:8]):
        user_input_list.append(triple_input_template.format(head=triple['head'], relation=triple['relation'], tail=triple['tail']))
user_input="\n".join(user_input_list)
print(user_input)
print("=========================以下是模型输出：================================")
response = client.chat.completions.create(
                model="glm-4-flash",
                messages=[
                    {"role": "system", "content": triple_instruction},
                    {"role": "user", "content": user_input},
                ],
                stream=True,
            )

full_response_content = ""
for chunk in response:
    delta = chunk.choices[0].delta
    if delta.content:
        full_response_content += delta.content

print(full_response_content)
