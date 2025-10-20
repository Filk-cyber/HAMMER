import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
device = "cuda" if torch.cuda.is_available() else "cpu"
from typing import Dict, List

def load_cag_model_tokenizer(model_path):
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        padding_side="left"
    )
    if device == "cuda":
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto"
        )
    model.eval()
    return model, tokenizer

def format_documents(self, retrieved_docs: List[Dict]) -> str:
    """
    格式化检索到的文档，包含可信度分数

    Args:
        retrieved_docs: 检索到的文档列表

    Returns:
        str: 格式化后的文档字符串
    """
    formatted_docs = []

    for i, doc in enumerate(retrieved_docs):
        score = doc["credibility"]
        text = doc["text"]
        if score <= 3:
            credibility = "Low credibility of text"
        elif score > 3 and score < 7:
            credibility = "Medium credibility of text"
        elif score >= 7:
            credibility = "High credibility of text"
        # 格式化文档：包含可信度分数和内容
        formatted_doc = f"{credibility}: {text} "
        formatted_docs.append(formatted_doc)

    return "\n".join(formatted_docs)

def create_prompt(question: str, retrieved_docs: List[Dict]) -> List[Dict]:
    """
    创建符合指定格式的prompt

    Args:
        question: 问题文本
        retrieved_docs: 检索到的文档列表

    Returns:
        List[Dict]: 聊天格式的prompt
    """
    # 添加Few-shot示例
    few_shot_examples = '''Examples:

Question: What is the capital of France?
Documents:
High credibility of text: France is a country in Western Europe. Paris is the capital and largest city of France.
Answer: Paris

Question: Who wrote Romeo and Juliet?
Documents:
High credibility of text: William Shakespeare was an English playwright and poet. He wrote many famous plays including Romeo and Juliet.
Answer: Shakespeare

Question: Is the sky blue?
Documents:
High credibility of text: The sky appears blue during clear weather due to light scattering in the atmosphere.
Answer: Yes

Question: What color is grass?
Documents:
High credibility of text: Grass is typically green in color due to chlorophyll in the plant cells.
Answer: Green

'''

    user_input = '''You are an assistant who answers questions based on given passages. Each passage has a credibility score indicating relevance and accuracy.

    IMPORTANT: Your answer must be EXACTLY one or few words only. Do not provide explanations, reasoning, or additional context.

    {examples}

    Question: {question}

    Documents:
    {paras}

    Answer (one or few words only):'''

    if retrieved_docs:
        paras = format_documents(retrieved_docs)
        user_input = user_input.format(examples=few_shot_examples, question=question, paras=paras)
    else:
        user_input = f'Answer the following question based on your internal knowledge with one or few words.\nQuestion: {question}\nAnswer: '

    return [{"role": "user", "content": user_input}]

cag_model, cag_tokenizer = load_cag_model_tokenizer("/home/jiangjp/models/cag-7b")
user_question = "Are 111 Murray Street and 220 Central Park South both located in Manhattan?"
messages = f'Answer the following question based on your internal knowledge with one or few words.\nQuestion: {user_question}\nAnswer: '
# # 将聊天格式转换为模型输入
# input_text = cag_tokenizer.apply_chat_template(
#     messages,
#     add_generation_prompt=True,  # 自动添加生成提示符
#     tokenize=False
# )

# 编码输入
encoded = cag_tokenizer(
    messages,
    truncation=True,
    padding=True,
    return_tensors="pt"
).to(device)

model_inputs = {
    "input_ids": encoded["input_ids"],
    "attention_mask": encoded["attention_mask"]
}
# 生成答案
with torch.no_grad():
    outputs = cag_model.generate(
        **model_inputs,
        max_new_tokens=25,
        do_sample=False,
        temperature=1.0,
        eos_token_id=cag_tokenizer.convert_tokens_to_ids("</s>")
    )

# 解码生成的答案
generated_ids = outputs[0, model_inputs["input_ids"].shape[1]:].detach().cpu()
generated_text = cag_tokenizer.decode(generated_ids, skip_special_tokens=True)
predicted_answer = generated_text.strip()

print(f"问题: {user_question}")
print(f"预测答案: {predicted_answer}")
print("-" * 50)