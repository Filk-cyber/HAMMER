import json
from zhipuai import ZhipuAI
from retrievers.e5_mistral import get_e5_mistral_embeddings_for_query, get_e5_mistral_embeddings_for_document
import torch
# 初始化智谱AI客户端
client = ZhipuAI(api_key="05748176082447f483438dfd914cc299.NcsCifhTarCch6es")  # 请填写您自己的APIKey

# GPT生成标题的指令和模板
instruction = """Your task is to generate a single concise title for the given English paragraph. The generated title should be less than 10 words.
Here are 2 examples, you should follow the output format below:
##########
Passage:
Boston College (also referred to as BC) is a private Jesuit Catholic research university located in the affluent village of Chestnut Hill, Massachusetts, United States, 6 mi west of downtown Boston. It has 9,100 full-time undergraduates and almost 5,000 graduate students. The university's name reflects its early history as a liberal arts college and preparatory school (now Boston College High School) in Dorchester. It is a member of the 568 Group and the Association of Jesuit Colleges and Universities. Its main campus is a historic district and features some of the earliest examples of collegiate gothic architecture in North America.

Title: Boston College



Passage:
The Rideau River Residence Association (RRRA) is the student organization that represents undergraduate students living in residence at Carleton University. It was founded in 1968 as the Carleton University Residence Association. Following a protracted fight with the university in the mid-1970s, it was renamed in its present form. It is a non-profit corporation that serves as Canada's oldest and largest residence association. Its membership consists of roughly 3,600 undergraduate students enrolled at the university living in residence. With an annual budget of approximately $1.4 million and three executives alongside volunteer staff, RRRA serves as an advocate for residence students and provides a variety of services, events, and programs to its members.

Title: Rideau River Residence Association
##########
"""

user_input_template = """Passage: {passage}
Title: 
"""


def get_dataset_demonstrations(dataset):
    if dataset == "hotpotqa":
        from prompts import generate_knowledge_triples_hotpotqa_examplars
        demonstrations = generate_knowledge_triples_hotpotqa_examplars
    elif dataset == "2wikimultihopqa":
        from prompts import generate_knowledge_triples_2wikimultihopqa_examplars
        demonstrations = generate_knowledge_triples_2wikimultihopqa_examplars
    elif dataset == "musique":
        from prompts import generate_knowledge_triples_musique_examplars
        demonstrations = generate_knowledge_triples_musique_examplars
    else:
        raise ValueError(f"{dataset} is not a supported dataset!")

    return demonstrations

def generate_title(passage):
    """
    使用智谱AI生成段落标题
    """
    try:
        user_input = user_input_template.format(passage=passage)
        response = client.chat.completions.create(
            model="glm-4-flash",
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

        return full_response_content.strip()
    except Exception as e:
        print(f"生成标题时出错: {e}")
        return "Generated Title"  # 返回默认标题


def split_sentences(text):
    """
    根据句号拆分句子，保留句号
    """
    # 按句号分割
    parts = text.split('.')
    sentences = []

    for i, part in enumerate(parts):
        part = part.strip()
        if part:  # 如果部分不为空
            # 除了最后一个部分，其他部分都要加回句号
            if i < len(parts) - 1:
                sentences.append(part + '.')
            else:
                # 最后一个部分，如果原文本以句号结尾，也要加句号
                if text.endswith('.'):
                    sentences.append(part + '.')
                else:
                    sentences.append(part)

    return sentences


def convert_paragraph_to_ctx_format(paragraph, existing_ctxs):
    """
    将段落转换为ctxs格式的json对象
    """
    # 生成唯一ID（可以基于现有ctxs的数量）
    new_id = len(existing_ctxs)

    # 生成标题
    title = generate_title(paragraph)

    # 拆分句子
    sentences = split_sentences(paragraph)

    # 构造新的ctx对象
    new_ctx = {
        "id": str(new_id),
        "title": title,
        "text": paragraph,
        "sentences": sentences
    }

    return new_ctx


def process_dataset_with_demonstration_rank(data, dataset):
    """
    处理数据集并添加相似度排行（优化版本：批量计算相似度）
    """
    # 获取demonstration embeddings
    dataset_demonstrations = get_dataset_demonstrations(dataset)
    demonstration_texts = ["title: {} text: {}".format(demo["title"], demo["text"]) for demo in dataset_demonstrations]
    demonstration_embeddings = get_e5_mistral_embeddings_for_document(
        doc_list=demonstration_texts,
        max_length=256,
        batch_size=4,
    )

    processed_data = []

    for item in data:
        new_item = item.copy()
        existing_ctxs = item.get('ctxs', [])
        ori_fake = item.get('ori_fake', [])

        # 第一步：为所有ori_fake段落生成ctx对象（不包含相似度排行）
        new_ctxs = []
        new_documents_texts = []

        for paragraph in ori_fake:
            if paragraph.strip():
                # 生成基础的ctx对象
                new_ctx = convert_paragraph_to_ctx_format(paragraph, existing_ctxs)
                new_ctxs.append(new_ctx)

                # 构建用于相似度计算的文本
                document_text = f"title: {new_ctx['title']} text: {new_ctx['text']}"
                new_documents_texts.append(document_text)

                # 更新existing_ctxs计数（用于下一个ID生成）
                existing_ctxs.append(new_ctx)

        # 第二步：如果有新的ctx，批量计算相似度
        if new_documents_texts:
            print(f"正在为 {len(new_documents_texts)} 个新段落计算相似度...")

            # 一次性计算所有新文档的嵌入向量
            documents_embeddings = get_e5_mistral_embeddings_for_query(
                "retrieve_semantically_similar_text",
                query_list=new_documents_texts,
                max_length=256,
                batch_size=4,
            )

            # 一次性计算所有相似度
            similarities = torch.matmul(documents_embeddings, demonstration_embeddings.T)
            demonstration_ranks = torch.argsort(similarities, dim=1, descending=True)

            # 第三步：将计算结果分别赋值给每个ctx
            for i, new_ctx in enumerate(new_ctxs):
                new_ctx["ranked_prompt_indices"] = demonstration_ranks[i].tolist()
                print(f"已处理段落，生成标题: {new_ctx['title']}")

        # 更新ctxs字段（这里的existing_ctxs已经包含了所有新的ctx）
        new_item['ctxs'] = existing_ctxs
        processed_data.append(new_item)

    return processed_data


def main():
    """
    主函数
    """
    # 读取JSON数据集
    input_file = "/home/jiangjp/trace-idea/data/hotpotqa/hotpotqa_dev100_add_orifake.json"  # 请替换为您的输入文件路径
    output_file = "/home/jiangjp/trace-idea/data/hotpotqa/hotpotqa_dev100_add_ctxs.json"  # 输出文件路径

    try:
        # 读取数据
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        print(f"开始处理数据集，共有 {len(data)} 个项目...")

        # 处理数据
        processed_data = process_dataset_with_demonstration_rank(data,"hotpotqa")

        # 保存处理后的数据
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(processed_data, f, ensure_ascii=False, indent=2)

        print(f"数据处理完成！结果已保存到 {output_file}")

    except FileNotFoundError:
        print(f"错误：找不到输入文件 {input_file}")
    except json.JSONDecodeError:
        print("错误：JSON文件格式不正确")
    except Exception as e:
        print(f"处理过程中出现错误: {e}")


# 示例用法
if __name__ == "__main__":
    # 处理文件中的数据
    main()