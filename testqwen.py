import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
device = "cuda" if torch.cuda.is_available() else "cpu"
from typing import Dict, List
def load_qwen_model_tokenizer(model_path):
    """
    加载Qwen1.5-7B模型和tokenizer

    Args:
        model_path: 模型路径

    Returns:
        tuple: (model, tokenizer)
    """
    print(f"正在加载Qwen1.5-7B模型: {model_path}")

    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        padding_side="left",
        trust_remote_code=True
    )

    # 设置pad_token
    if tokenizer.pad_token is None or tokenizer.pad_token_id is None:
        print("设置padding token为eos_token")
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # 加载模型
    if device == "cuda":
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float32,
            trust_remote_code=True
        )
        model.to(device)

    model.eval()
    print("Qwen模型加载完成!")
    return model, tokenizer

def parse_chat_answer(answer):
    """
    解析Qwen生成的答案

    Args:
        answer: 模型生成的原始答案

    Returns:
        str: 清理后的答案
    """
    # 清理常见的前缀
    if "Answer:" in answer:
        idx = answer.find("Answer:")
        answer = answer[idx + len("Answer:"):].strip()
        if answer.startswith(":"):
            answer = answer[1:].strip()
    return answer

def parse_gemma_answer(answer):

    candidate_answers = answer.split("\n")
    answer = ""
    i = 0
    while len(answer) < 1 and i<len(candidate_answers):
        answer = candidate_answers[i].strip()
        i += 1
    return answer
def get_contexts(self, retrieved_docs: List[Dict]) -> str:
    """
    格式化检索到的文档为上下文

    Args:
        retrieved_docs: 检索到的文档列表

    Returns:
        str: 格式化后的上下文字符串
    """
    contexts = []
    for i, doc in enumerate(retrieved_docs):
        text = doc["text"]
        contexts.append(("Passage-%d: " % i) + text)

    return "\n".join(contexts)

def create_qwen_prompt(question: str, retrieved_docs: List[Dict]) -> List[Dict]:
    """
    创建符合Qwen格式的prompt

    Args:
        question: 问题文本
        retrieved_docs: 检索到的文档列表

    Returns:
        List[Dict]: 聊天格式的prompt
    """
    if retrieved_docs:
        contexts = get_contexts(retrieved_docs)
        prompt = f'Given the following information: \n{contexts}\nAnswer the following question based on the given information or your internal knowledge with one or few words without the source.\nQuestion: {question}\nAnswer: '
    else:
        prompt = f'Answer the following question based on your internal knowledge with one or few words.\nQuestion: {question}\nAnswer: '

    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant."
        },
        {
            "role": "user",
            "content": prompt
        },
    ]

    return messages

model, tokenizer = load_qwen_model_tokenizer("/home/jiangjp/mydatastore/Mistral-7B-v0.1")
user_question = "What disease is the Wolf Prize winner and Vadim  Bereinskii-inspired Nobel laureate suffering from?"
answer=''
paras=''
input_text=f'Given the following information: \n{paras}\nplease only output the answer to the question.\nQuestion: {user_question}\nthe correct answer is:{answer}'
# messages = create_qwen_prompt(user_question, None)
# # 将聊天格式转换为模型输入
# input_text = tokenizer.apply_chat_template(
#     messages,
#     add_generation_prompt=True,
#     tokenize=False,
# )

# 编码输入
encoded = tokenizer(
    input_text, padding=True, truncation=True, return_tensors='pt'
).to(device)

model_inputs = {
    "input_ids": encoded["input_ids"],
    "attention_mask": encoded["attention_mask"]
}

# 生成答案
with torch.no_grad():
    outputs = model.generate(
        **model_inputs,
        max_new_tokens=25,
        do_sample=False,
        temperature=1.0,
        use_cache=True,
    )

# 解码生成的答案
generated_ids = outputs[0, model_inputs["input_ids"].shape[1]:].detach().cpu()
generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
print("original:", generated_text)
predicted_answer = parse_gemma_answer(generated_text)

print(f"问题: {user_question}")
print(f"预测答案: {predicted_answer}")
print("-" * 50)
