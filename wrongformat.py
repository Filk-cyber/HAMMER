import json
input_file="/home/jiangjp/trace-idea/data/2wikimultihopqa/wiki_dev100ideal_with_reasoning_chains_modify_w00.json"
try:
    with open(input_file, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
except Exception as e:
    print(f"读取输入文件失败: {e}")

def clean_wrong_answer(wrong_answer):
    """
    清理GLM-4-FLASH生成的错误答案格式

    输入示例:
    - "['Canadian']"
    - "['yes']"
    - "['no']"
    - "['Rock All Night' (1980) and 'Death On The Job' (1982) were both released, but 'Death On The Job' was actually released first, in 1982.]"

    输出: 清理后的答案文本
    """
    if not isinstance(wrong_answer, str):
        return str(wrong_answer).strip()

    # 去除首尾空白
    cleaned = wrong_answer.strip()

    # 如果以 [' 开头并以 '] 结尾，去除这些符号
    if cleaned.startswith("['") and cleaned.endswith("']"):
        cleaned = cleaned[2:-2]  # 去除 ['  和 ']
    # 如果以 [ 开头并以 ] 结尾，去除这些符号
    elif cleaned.startswith("[") and cleaned.endswith("]"):
        cleaned = cleaned[1:-1]  # 去除 [  和 ]

    # 最终清理首尾空白
    cleaned = cleaned.strip()

    return cleaned
# for itemid,item in enumerate(dataset):
#     print(item["wrong_answer"])

# 测试清理函数
print(f"\n错误答案清理测试:")
test_cases = [
    "['Canadian']",
    "['yes']",
    "['no']",
    "['Rock All Night' (1980) and 'Death On The Job' (1982) were both released, but 'Death On The Job' was actually released first, in 1982.]"
]

for test in test_cases:
    cleaned = clean_wrong_answer(test)
    print(f"原始: {test}")
    print(f"清理: {cleaned}")
    print("---")