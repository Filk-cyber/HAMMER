from zhipuai import ZhipuAI
import re
from typing import List, Tuple
client = ZhipuAI(api_key="05748176082447f483438dfd914cc299.NcsCifhTarCch6es")  # 请填写您自己的APIKey
instruction = """Your task is to evaluate the authenticity and usefulness of the text based on your internal knowledge. Specifically, I will provide you with a question and multiple paragraphs that may contain accurate information or fabricated errors. Using your own knowledge, reasoning, and inference, you are to assign two scores: a credibility score and a usefulness score, both on a scale from 0 to 10.A higher credibility score indicates higher authenticity, while a lower score indicates lower authenticity.A higher usefulness score means the text provides greater help in the logical steps required to answer the question. A lower usefulness score means it provides less help. Here is an example, you should follow the following output format:
##########
Question: 
Who was the U.S. President when the scientist who discovered penicillin was knighted?
Paragraph:
Sir Alexander Fleming, a Scottish physician and microbiologist, was formally knighted for his services to science and medicine in 1944 by King George VI at Buckingham Palace.

Analysis: This passage is historically accurate. It correctly identifies the scientist and the year he was knighted, which is 1944. This year is the most critical piece of information needed to solve the multi-hop question. Therefore, the passage is both highly credible and highly useful. 

Credibility Score: 10 
Usefulness Score: 10


Paragraph:
In 1951, during the presidency of Harry S. Truman, the renowned scientist Alexander Fleming was knighted for his discovery of antibiotics. The ceremony was widely celebrated as a symbol of post-war scientific achievement.

Analysis: This passage contains critical factual errors. Alexander Fleming was knighted in 1944, not 1951. Consequently, the U.S. President at the time of his knighthood was Franklin D. Roosevelt, not Harry S. Truman (who took office in 1945). Although the passage appears relevant by mentioning a scientist, a knighthood, and a U.S. President, its core facts are incorrect, making it highly misleading and not useful.

Credibility Score: 1 
Usefulness Score: 0
##########
"""
question_template="""Question:
{question}"""
context_input_template = """Paragraph:
{paragraph}

Credibility Score:
Usefulness Score:
"""
context_input_list=[]
context_list2=["Stenomesson is a genus of bulbous plants in the family Amaryllidaceae.","All the species are native to western South America (Colombia, Ecuador, Peru and northern Chile).","Clinantheae is a tribe (in the family Amaryllidaceae, subfamily Amaryllidoideae), where it forms part of the Andean clade, one of two American clades.","The tribe was described in 2000 by Alan Meerow \"et al.\" as a result of a molecular phylogenetic study of the American Amaryllidoideae.","Michelia punduana is a species of plant in the Magnoliaceae family."]
context_list=["Michelia ingrata is a species of plant in the Magnoliaceae family. It is endemic to China.","Michelia wilsonii is a species of plant in the Magnoliaceae family. It is endemic to China. It is threatened by habitat loss.","Michelia odora is a species of plant in the Magnoliaceae family. It is found in China and Vietnam. It is threatened by habitat loss.","Stenomesseae was a tribe (in the family Amaryllidaceae, subfamily Amaryllidoideae), where it forms part of the Andean clade, one of two American clades. The tribe was originally described by Traub in his monograph on the Amaryllidaceae in 1963, as Stenomessae based on the type genus \"Stenomesson\". In 1995 it was recognised that Eustephieae was a distinct group separate from the other Stenomesseae. Subsequently, the M\u00fcller-Doblies' (1996) divided tribe Eustephieae into two subtribes, Stenomessinae and Eustephiinae.","Stenomesson is a genus of bulbous plants in the family Amaryllidaceae. All the species are native to western South America (Colombia, Ecuador, Peru and northern Chile)."]
question_input=question_template.format(question="Are the genuses Michelia and Stenomesson in the same family?")
for ctx in context_list2:
    context_input_list.append(context_input_template.format(paragraph=ctx))
context_input="\n".join(context_input_list)
user_input = f"{question_input}\n{context_input}"
print(user_input)
print("================================================")
response = client.chat.completions.create(
    model="glm-4-flash",  # 请填写您要调用的模型名称
    messages=[
        {"role": "system", "content": instruction},
        {"role": "user", "content": user_input},
    ],
    stream=True,
)
full_response_content = ""
for chunk in response:
    delta = chunk.choices[0].delta
    if delta.content:
        full_response_content += delta.content

print(f"拼接后的完整结果: {full_response_content}")

def extract_scores_simple(text: str) -> List[Tuple[int, int]]:
    """
    简化版本：直接按顺序提取所有Credibility和Usefulness评分

    Args:
        text (str): 包含评分信息的文本

    Returns:
        List[Tuple[int, int]]: 每个元组包含(Credibility, Usefulness)评分对
    """
    # 查找所有Credibility评分
    credibility_scores = re.findall(r'Credibility Score:\s*(\d+)', text)
    # 查找所有Usefulness评分
    usefulness_scores = re.findall(r'Usefulness Score:\s*(\d+)', text)

    # 将两个列表配对
    scores = []
    for i in range(min(len(credibility_scores), len(usefulness_scores))):
        credibility = int(credibility_scores[i])
        usefulness = int(usefulness_scores[i])
        scores.append((credibility, usefulness))

    return scores
final_answer = extract_scores_simple(full_response_content)
print(f"截取后的最终答案: {final_answer}")
for i, (cred, useful) in enumerate(final_answer, 1):
        print(f"Context {i}: Credibility={cred}, Usefulness={useful}")