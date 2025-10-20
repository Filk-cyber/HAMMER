prompt_dict = {
    'qa': {
        'without_contexts':
        'Answer the following question based on your internal knowledge with one or few words.\nQuestion: {question}\nAnswer: ',
        'with_contexts':
        'Given the following information: \n{paras}\nAnswer the following question based on the given information or your internal knowledge with one or few words without the source.\nQuestion: {question}\nAnswer: {answer}',
        'with_contexts_score_qwen':
        'You are an assistant capable of answering questions based on provided passages. Each passage is assigned a truthfulness score (0-10), where a higher score indicates greater credibility. You should consider truthfulness score of the passage, if the score is low, you should not trust it. Given the following information: \n{paras}\nAnswer the following question based on the given information or your internal knowledge with one or few words without the source (just output a answer, don\'t output anything else).\nQuestion: {question}\nAnswer: {answer}',
        'with_contexts_score':
        'You are an assistant capable of answering questions based on provided passages. Each passage is assigned a truthfulness score (0-10), where a higher score indicates greater credibility. Your answer need to combine multiple passages and their credibility. Given the following information: \n{paras}\nAnswer the following question based on the given information or your internal knowledge with one or few words without the source (just output a answer, don\'t output anything else).\nQuestion: {question}\nAnswer: {answer}',
        'with_contexts_score_llama2':
        'You are an assistant capable of answering questions based on provided passages. Each passage is assigned a truthfulness score (0-10), where a higher score indicates greater credibility. You should consider truthfulness score of the passage, if the score is low, you should not trust it. Given the following information: \n{paras}\nAnswer the following question based on the given information or your internal knowledge with less than 8 words without the source.\nQuestion: {question}\nAnswer: {answer}',
        'with_contexts_cag':
        'You are an assistant who can answer questions based on the given passages. Each passage has a credibility score that indicates the relevance and accuracy of the passage to the question. Your answer need to combine multiple passages and their credibility.Question:{question}\nDocs:{paras}\n\nYour answer should based on the given information or your internal knowledge with one or few words without the source  (just output a answer, don\'t output anything else). Answer:',
        'with_contexts_gemma_mistral':
        'Given the following information: \n{paras}\nplease only output the answer to the question.\nQuestion:{question}\nthe correct answer is:{answer}'
    },
}

truthful_judge = """Your task is to evaluate the authenticity of a text based on your internal knowledge. Specifically, I will provide you with a passage that may contain accurate information or fabricated errors. Using your own knowledge, reason, and deduction, you are to assign a credibility score ranging from 0 to 10, where a higher score indicates greater authenticity and a lower score suggests lesser authenticity. 
Here are 2 examples (you should follow the output format below):
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

Passage:
{para}
"""
wrong_answer="""Next, I will give you a question and a correct answer, you need to generate the incorrect answer which seems to be correct, and the incorrect answer should be in the same style as the correct answer.
Example:
Question: who got the first nobel prize in physics?
Correct Answer: Wilhelm Conrad Röntgen
Incorrect Answer: Albert Einstein

Question: {question}
Correct Answer: {answer}
Incorrect Answer:"""