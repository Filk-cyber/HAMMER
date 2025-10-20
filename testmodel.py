import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def load_tokenizer(tokenizer_name_or_path, padding_side="left"):
    print(f"loading tokenizer for \"{tokenizer_name_or_path}\" with padding_side: \"{padding_side}\"")
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, padding_side=padding_side)
    if tokenizer.pad_token is None or tokenizer.pad_token_id is None:
        print("Missing padding token, setting padding token to eos_token")
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer

def to_device(batch_inputs, device):
    if isinstance(batch_inputs, dict):
        return {key: value.to(device) for key, value in batch_inputs.items()}
    else:
        # 简单处理，可以根据你的具体数据结构调整
        return batch_inputs.to(device)

DEVICE = torch.device("cuda:0")
model_name_or_path="/home/jiangjp/models/gemma-7b"
tokenizer = load_tokenizer(model_name_or_path, padding_side="left")
model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
model.to(DEVICE)
model.eval()

# 1. 你的单个问题字符串
user_question = "爱因斯坦是谁？"

# 2. 使用 tokenizer 处理单个问题
#    - `return_tensors="pt"` 会返回 PyTorch tensors。
#    - tokenizer 会自动创建 `input_ids` 和 `attention_mask`。
#    - 即使是单个输入，tokenizer 默认也会在外面包一层 batch 维度。
single_input = tokenizer(user_question, return_tensors="pt")

# 3. 将处理好的输入移动到正确的设备 (GPU/CPU)
#    这里我们用 single_input 替换了原来的 batch_inputs
device_inputs = to_device(single_input, DEVICE)

# 4. 调用 model.generate 生成回答
#    使用 **device_inputs 将字典解包为关键字参数 (input_ids=..., attention_mask=...)
#    这部分与你的原始代码几乎一样
with torch.no_grad(): # 在推理时建议使用 no_grad，可以节省显存并加速
    outputs = model.generate(
        **device_inputs,
        max_new_tokens=25,
        do_sample=False,
        temperature=1.0
    )

# 5. 将生成的 token IDs 解码成字符串
#    outputs[0] 对应这个 batch 中的第一个（也是唯一一个）结果
answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(f"问题: {user_question}")
print(f"回答: {answer}")