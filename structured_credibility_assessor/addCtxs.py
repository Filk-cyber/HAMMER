import json
import time
import concurrent.futures
from threading import Lock
from zhipuai import ZhipuAI
from retrievers.e5_mistral import get_e5_mistral_embeddings_for_query, get_e5_mistral_embeddings_for_document
import torch
from typing import List, Dict, Any, Tuple


class OptimizedTitleGenerator:
    def __init__(self, api_key: str, max_workers: int = 10):
        """
        初始化优化版标题生成器

        Args:
            api_key: ZhipuAI的API密钥
            max_workers: 最大并发线程数
        """
        self.client = ZhipuAI(api_key=api_key)
        self.max_workers = max_workers
        self.progress_lock = Lock()
        self.completed_count = 0
        self.total_count = 0

        # 默认标识
        self.DEFAULT_TITLE = "DEFAULT_TITLE_PLACEHOLDER"
        self.EMPTY_TITLE = ""

        # 最小配置值
        self.MIN_WORKERS = 1

        # 保持原始的提示词模板
        self.instruction = """Your task is to generate a single concise title for the given English paragraph. The generated title should be less than 10 words.
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

        self.user_input_template = """Passage: {passage}
Title: 
"""

    def get_dataset_demonstrations(self, dataset):
        """获取数据集演示样例"""
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

    def split_sentences(self, text):
        """根据句号拆分句子，保留句号"""
        parts = text.split('.')
        sentences = []

        for i, part in enumerate(parts):
            part = part.strip()
            if part:
                if i < len(parts) - 1:
                    sentences.append(part + '.')
                else:
                    if text.endswith('.'):
                        sentences.append(part + '.')
                    else:
                        sentences.append(part)
        return sentences

    def generate_title_single(self, passage: str) -> str:
        """
        为单个段落生成标题

        Args:
            passage: 段落文本

        Returns:
            生成的标题
        """
        try:
            user_input = self.user_input_template.format(passage=passage)
            response = self.client.chat.completions.create(
                model="glm-4-flash",
                messages=[
                    {"role": "system", "content": self.instruction},
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
            print(f"生成标题失败: {str(e)}")
            return self.DEFAULT_TITLE

    def call_api_with_retry(self, passage: str, max_retries: int = 3) -> str:
        """
        调用API并重试的方法

        Args:
            passage: 段落文本
            max_retries: 最大重试次数

        Returns:
            生成的标题
        """
        for attempt in range(max_retries):
            try:
                result = self.generate_title_single(passage)
                if result != self.DEFAULT_TITLE and result.strip():
                    return result
                else:
                    print(f"第 {attempt + 1} 次尝试得到空或默认结果")
            except Exception as e:
                print(f"API调用第 {attempt + 1} 次尝试失败: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)

        print(f"API调用最终失败，返回默认值")
        return self.DEFAULT_TITLE

    def generate_titles_only(self, passage_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        仅生成标题的单线程处理函数

        Args:
            passage_data: 包含段落信息的字典

        Returns:
            包含标题的结果字典
        """
        try:
            item_idx = passage_data['item_idx']
            paragraph_idx = passage_data['paragraph_idx']
            paragraph = passage_data['paragraph']

            # 生成标题
            title = self.call_api_with_retry(paragraph)

            with self.progress_lock:
                self.completed_count += 1
                print(
                    f"标题生成进度: {self.completed_count}/{self.total_count} - 项目 {item_idx + 1}, 段落 {paragraph_idx + 1}")

            return {
                'item_idx': item_idx,
                'paragraph_idx': paragraph_idx,
                'paragraph': paragraph,
                'title': title,
                'success': True
            }

        except Exception as e:
            print(f"处理项目 {item_idx}, 段落 {paragraph_idx} 时发生错误: {e}")
            return {
                'item_idx': item_idx,
                'paragraph_idx': paragraph_idx,
                'paragraph': passage_data['paragraph'],
                'title': self.DEFAULT_TITLE,
                'success': False
            }

    def generate_title_for_ctx(self, passage_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        为单个ctx生成标题

        Args:
            passage_data: 包含段落信息的字典

        Returns:
            包含标题的结果字典
        """
        try:
            item_idx = passage_data['item_idx']
            ctx_idx = passage_data['ctx_idx']
            paragraph = passage_data['paragraph']

            # 生成标题
            title = self.call_api_with_retry(paragraph)

            with self.progress_lock:
                self.completed_count += 1
                print(
                    f"标题生成进度: {self.completed_count}/{self.total_count} - 项目 {item_idx + 1}, ctx {ctx_idx + 1}")

            return {
                'item_idx': item_idx,
                'ctx_idx': ctx_idx,
                'paragraph': paragraph,
                'title': title,
                'success': True
            }

        except Exception as e:
            print(f"处理项目 {item_idx}, ctx {ctx_idx} 时发生错误: {e}")
            return {
                'item_idx': item_idx,
                'ctx_idx': ctx_idx,
                'paragraph': paragraph,
                'title': self.DEFAULT_TITLE,
                'success': False
            }

    def collect_all_paragraphs(self, dataset: List[Dict]) -> List[Dict[str, Any]]:
        """
        收集所有需要处理的段落数据

        Args:
            dataset: 数据集

        Returns:
            所有段落的数据列表
        """
        all_paragraphs_data = []

        for item_idx, item in enumerate(dataset):
            # 检查是否有ori_fake字段
            if 'ori_fake' in item and isinstance(item['ori_fake'], list):
                for paragraph_idx, paragraph in enumerate(item['ori_fake']):
                    if paragraph.strip():
                        all_paragraphs_data.append({
                            'item_idx': item_idx,
                            'paragraph_idx': paragraph_idx,
                            'paragraph': paragraph
                        })

        return all_paragraphs_data

    def collect_default_title_paragraphs(self, dataset: List[Dict]) -> List[Dict[str, Any]]:
        """
        收集具有默认标题或空标题的段落数据

        Args:
            dataset: 数据集

        Returns:
            需要重新生成标题的段落数据列表
        """
        paragraphs_data = []

        for item_idx, item in enumerate(dataset):
            if 'ctxs' in item and isinstance(item['ctxs'], list):
                for ctx_idx, ctx in enumerate(item['ctxs']):
                    if 'title' in ctx and 'text' in ctx:
                        title = ctx['title']
                        if title == self.DEFAULT_TITLE or title.strip() == self.EMPTY_TITLE:
                            paragraphs_data.append({
                                'item_idx': item_idx,
                                'ctx_idx': ctx_idx,
                                'paragraph': ctx['text']
                            })

        return paragraphs_data

    def collect_missing_title_paragraphs(self, dataset: List[Dict]) -> List[Dict[str, Any]]:
        """
        收集缺失标题字段的段落数据

        Args:
            dataset: 数据集

        Returns:
            需要添加标题的段落数据列表
        """
        paragraphs_data = []

        for item_idx, item in enumerate(dataset):
            if 'ctxs' in item and isinstance(item['ctxs'], list):
                for ctx_idx, ctx in enumerate(item['ctxs']):
                    if 'title' not in ctx and 'text' in ctx:
                        paragraphs_data.append({
                            'item_idx': item_idx,
                            'ctx_idx': ctx_idx,
                            'paragraph': ctx['text']
                        })

        return paragraphs_data

    def stage1_generate_all_titles(self, dataset: List[Dict], output_file: str) -> List[Dict[str, Any]]:
        """
        第一阶段：批量生成所有标题

        Args:
            dataset: 数据集
            output_file: 输出文件路径

        Returns:
            所有标题生成结果
        """
        print("=" * 80)
        print("第一阶段：批量生成所有标题")
        print("=" * 80)

        # 收集所有段落
        all_paragraphs_data = self.collect_all_paragraphs(dataset)
        self.total_count = len(all_paragraphs_data)
        self.completed_count = 0

        print(f"总共需要生成 {self.total_count} 个标题")

        if self.total_count == 0:
            print("没有需要处理的段落")
            return []

        # 使用多线程生成标题
        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_paragraph = {
                executor.submit(self.generate_titles_only, paragraph_data): paragraph_data
                for paragraph_data in all_paragraphs_data
            }

            for future in concurrent.futures.as_completed(future_to_paragraph):
                result = future.result()
                results.append(result)

        success_count = sum(1 for r in results if r['success'])
        print(f"标题生成完成: {success_count}/{len(results)} 成功")

        # 保存标题生成结果
        self.save_titles_results(results, output_file, "stage1_titles")

        return results

    def stage1_generate_titles_for_ctxs(self, paragraphs_data: List[Dict[str, Any]], output_file: str) -> List[Dict[str, Any]]:
        """
        第一阶段：为现有ctxs生成标题

        Args:
            paragraphs_data: 段落数据列表
            output_file: 输出文件路径

        Returns:
            标题生成结果
        """
        print("=" * 80)
        print("第一阶段：为现有ctxs生成标题")
        print("=" * 80)

        self.total_count = len(paragraphs_data)
        self.completed_count = 0

        print(f"总共需要生成 {self.total_count} 个标题")

        if self.total_count == 0:
            print("没有需要处理的段落")
            return []

        # 使用多线程生成标题
        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_paragraph = {
                executor.submit(self.generate_title_for_ctx, paragraph_data): paragraph_data
                for paragraph_data in paragraphs_data
            }

            for future in concurrent.futures.as_completed(future_to_paragraph):
                result = future.result()
                results.append(result)

        success_count = sum(1 for r in results if r['success'])
        print(f"标题生成完成: {success_count}/{len(results)} 成功")

        return results

    def save_titles_results(self, titles_results: List[Dict], output_file: str, stage: str):
        """
        保存标题生成结果

        Args:
            titles_results: 标题生成结果列表
            output_file: 输出文件路径
            stage: 阶段标识
        """
        try:
            temp_file = f"{output_file}.{stage}.json"
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(titles_results, f, ensure_ascii=False, indent=2)
            print(f"{stage}阶段标题结果已保存到: {temp_file}")
        except Exception as e:
            print(f"保存{stage}阶段标题结果失败: {e}")

    def stage2_calculate_similarities(self, titles_results: List[Dict], dataset_name: str, output_file: str) -> List[Dict]:
        """
        第二阶段：计算所有相似度

        Args:
            titles_results: 标题生成结果
            dataset_name: 数据集名称
            output_file: 输出文件路径

        Returns:
            包含相似度的结果
        """
        print("=" * 80)
        print("第二阶段：计算所有相似度")
        print("=" * 80)

        if not titles_results:
            print("没有标题结果需要处理")
            return []

        # 获取demonstration embeddings
        dataset_demonstrations = self.get_dataset_demonstrations(dataset_name)
        demonstration_texts = ["title: {} text: {}".format(demo["title"], demo["text"]) for demo in
                               dataset_demonstrations]

        print(f"正在计算demonstration embeddings...")
        demonstration_embeddings = get_e5_mistral_embeddings_for_document(
            doc_list=demonstration_texts,
            max_length=256,
            batch_size=4,
        )

        # 构建文档文本列表
        document_texts = []
        for result in titles_results:
            if result['success']:
                document_text = f"title: {result['title']} text: {result['paragraph']}"
                document_texts.append(document_text)
            else:
                document_texts.append("")  # 占位符

        print(f"正在计算 {len(document_texts)} 个文档的embeddings...")

        # 分批处理避免显存溢出
        batch_size = 13  # 可根据显存情况调整
        all_similarities = []

        for i in range(0, len(document_texts), batch_size):
            batch_texts = document_texts[i:i + batch_size]
            # 过滤掉空文本
            valid_texts = [text for text in batch_texts if text.strip()]

            if valid_texts:
                print(f"处理批次 {i // batch_size + 1}/{(len(document_texts) + batch_size - 1) // batch_size}")

                # 计算当前批次的嵌入向量
                documents_embeddings = get_e5_mistral_embeddings_for_query(
                    "retrieve_semantically_similar_text",
                    query_list=valid_texts,
                    max_length=256,
                    batch_size=4,
                )

                # 计算相似度
                similarities = torch.matmul(documents_embeddings, demonstration_embeddings.T)
                demonstration_ranks = torch.argsort(similarities, dim=1, descending=True)

                # 添加到总结果中，处理空文本的情况
                valid_idx = 0
                for j, text in enumerate(batch_texts):
                    if text.strip():
                        all_similarities.append(demonstration_ranks[valid_idx].tolist())
                        valid_idx += 1
                    else:
                        all_similarities.append([])  # 空文本的占位符

                # 清理显存
                del documents_embeddings, similarities, demonstration_ranks
                torch.cuda.empty_cache() if torch.cuda.is_available() else None

        # 将相似度结果添加到titles_results中
        enhanced_results = []
        for i, result in enumerate(titles_results):
            enhanced_result = result.copy()
            if i < len(all_similarities):
                enhanced_result['ranked_prompt_indices'] = all_similarities[i]
            else:
                enhanced_result['ranked_prompt_indices'] = []
            enhanced_results.append(enhanced_result)

        print(f"相似度计算完成")

        # 保存包含相似度的结果
        self.save_titles_results(enhanced_results, output_file, "stage2_similarities")

        return enhanced_results

    def stage3_apply_to_dataset(self, enhanced_results: List[Dict], dataset: List[Dict]) -> List[Dict]:
        """
        第三阶段：将结果应用到数据集

        Args:
            enhanced_results: 包含标题和相似度的结果
            dataset: 原始数据集

        Returns:
            更新后的数据集
        """
        print("=" * 80)
        print("第三阶段：将结果应用到数据集")
        print("=" * 80)

        if not enhanced_results:
            print("没有结果需要应用")
            return dataset

        # 按item_idx分组结果
        results_by_item = {}
        for result in enhanced_results:
            item_idx = result['item_idx']
            if item_idx not in results_by_item:
                results_by_item[item_idx] = []
            results_by_item[item_idx].append(result)

        # 应用结果到数据集
        processed_items = 0
        for item_idx, item_results in results_by_item.items():
            if item_idx < len(dataset):
                existing_ctxs = dataset[item_idx].get('ctxs', [])

                # 为每个段落创建ctx对象
                for result in item_results:
                    if result['success'] and result.get('ranked_prompt_indices'):
                        # 生成唯一ID
                        new_id = len(existing_ctxs)

                        # 拆分句子
                        sentences = self.split_sentences(result['paragraph'])

                        # 构造新的ctx对象
                        new_ctx = {
                            "id": str(new_id),
                            "title": result['title'],
                            "text": result['paragraph'],
                            "sentences": sentences,
                            "ranked_prompt_indices": result['ranked_prompt_indices']
                        }

                        existing_ctxs.append(new_ctx)

                # 更新数据集
                dataset[item_idx]['ctxs'] = existing_ctxs
                processed_items += 1

        print(f"数据集更新完成，处理了 {processed_items} 个项目")
        return dataset

    def stage3_apply_to_existing_ctxs(self, enhanced_results: List[Dict], dataset: List[Dict]) -> List[Dict]:
        """
        第三阶段：将标题和相似度结果应用到现有ctxs

        Args:
            enhanced_results: 包含标题和相似度的结果
            dataset: 数据集

        Returns:
            更新后的数据集
        """
        print("=" * 80)
        print("第三阶段：将标题和相似度应用到现有ctxs")
        print("=" * 80)

        processed_count = 0
        for result in enhanced_results:
            if result['success'] and result.get('ranked_prompt_indices'):
                item_idx = result['item_idx']
                ctx_idx = result['ctx_idx']
                title = result['title']
                ranked_prompt_indices = result['ranked_prompt_indices']

                try:
                    if item_idx < len(dataset) and 'ctxs' in dataset[item_idx]:
                        ctxs = dataset[item_idx]['ctxs']
                        if ctx_idx < len(ctxs):
                            ctxs[ctx_idx]['title'] = title
                            ctxs[ctx_idx]['ranked_prompt_indices'] = ranked_prompt_indices
                            processed_count += 1
                except (IndexError, KeyError) as e:
                    print(f"应用结果到项目 {item_idx}, ctx {ctx_idx} 时发生错误: {e}")

        print(f"标题和相似度应用完成，处理了 {processed_count} 个ctx")
        return dataset

    def check_default_or_empty_titles(self, dataset: List[Dict]) -> List[Dict]:
        """
        检查数据集中是否有默认值或空值的title，并提取出来

        Args:
            dataset: 数据集

        Returns:
            包含默认值或空值的项目数据
        """
        failed_items = []

        for item_idx, item in enumerate(dataset):
            if 'ctxs' in item and isinstance(item['ctxs'], list):
                has_default_or_empty = False
                for ctx in item['ctxs']:
                    if 'title' in ctx:
                        title = ctx['title']
                        if title == self.DEFAULT_TITLE or title.strip() == self.EMPTY_TITLE:
                            has_default_or_empty = True
                            break

                if has_default_or_empty:
                    failed_items.append({
                        'item_idx': item_idx,
                        'item': item,
                        'dataset': 'hotpotqa'  # 默认数据集
                    })

        return failed_items

    def count_default_or_empty_titles(self, dataset: List[Dict]) -> Tuple[int, int]:
        """
        统计默认值或空值的数量

        Args:
            dataset: 数据集

        Returns:
            (items_with_issues, total_titles_with_issues): 有问题的条目数量和标题数量
        """
        items_with_issues = 0
        total_titles_with_issues = 0

        for item in dataset:
            if 'ctxs' in item and isinstance(item['ctxs'], list):
                item_has_issues = False
                for ctx in item['ctxs']:
                    if 'title' in ctx:
                        title = ctx['title']
                        if title == self.DEFAULT_TITLE or title.strip() == self.EMPTY_TITLE:
                            total_titles_with_issues += 1
                            item_has_issues = True

                if item_has_issues:
                    items_with_issues += 1

        return items_with_issues, total_titles_with_issues

    def check_missing_title_fields(self, dataset: List[Dict]) -> List[Dict]:
        """
        检查数据集中是否有缺失title字段的ctx，并提取出来

        Args:
            dataset: 数据集

        Returns:
            缺失title字段的项目数据
        """
        missing_items = []

        for item_idx, item in enumerate(dataset):
            if 'ctxs' in item and isinstance(item['ctxs'], list):
                has_missing_title = False
                for ctx in item['ctxs']:
                    if 'title' not in ctx:
                        has_missing_title = True
                        break

                if has_missing_title:
                    missing_items.append({
                        'item_idx': item_idx,
                        'item': item,
                        'dataset': 'hotpotqa'  # 默认数据集
                    })

        return missing_items

    def count_missing_title_fields(self, dataset: List[Dict]) -> int:
        """
        统计缺失title字段的数量

        Args:
            dataset: 数据集

        Returns:
            缺失title字段的ctx数量
        """
        missing_count = 0

        for item in dataset:
            if 'ctxs' in item and isinstance(item['ctxs'], list):
                for ctx in item['ctxs']:
                    if 'title' not in ctx:
                        missing_count += 1

        return missing_count

    def process_default_title_check(self, input_file: str, output_file: str, dataset_name: str = "hotpotqa"):
        """
        执行默认标题检查并修复（完整三阶段处理）

        Args:
            input_file: 输入文件路径
            output_file: 输出文件路径
            dataset_name: 数据集名称
        """
        print("🔍 执行默认标题检查模式（完整三阶段处理）")
        print("=" * 80)

        # 读取输入文件
        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                dataset = json.load(f)
        except Exception as e:
            print(f"读取输入文件失败: {e}")
            return

        if not isinstance(dataset, list):
            dataset = [dataset]

        # 统计默认标题数量
        items_count, titles_count = self.count_default_or_empty_titles(dataset)
        print(f"发现默认或空标题: {items_count} 个项目, {titles_count} 个标题")

        if titles_count == 0:
            print("✅ 没有发现默认或空标题，无需处理")
            return

        # 第一阶段：收集需要重新生成标题的段落并生成标题
        paragraphs_data = self.collect_default_title_paragraphs(dataset)
        print(f"收集到 {len(paragraphs_data)} 个需要重新生成标题的段落")

        titles_results = self.stage1_generate_titles_for_ctxs(paragraphs_data, output_file)

        # 第二阶段：计算相似度
        enhanced_results = self.stage2_calculate_similarities(titles_results, dataset_name, output_file)

        # 第三阶段：应用到数据集
        updated_dataset = self.stage3_apply_to_existing_ctxs(enhanced_results, dataset)

        # 保存结果
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(updated_dataset, f, ensure_ascii=False, indent=2)
            print(f"✅ 默认标题检查完成！结果已保存到: {output_file}")

            # 清理临时文件
            import os
            for stage in ["stage1_titles", "stage2_similarities"]:
                temp_file = f"{output_file}.{stage}.json"
                if os.path.exists(temp_file):
                    os.remove(temp_file)
                    print(f"已清理临时文件: {temp_file}")

            # 最终统计
            final_items_count, final_titles_count = self.count_default_or_empty_titles(updated_dataset)
            print(f"🏁 处理后统计:")
            print(f"   - 剩余默认/空标题: {final_items_count} 个项目, {final_titles_count} 个标题")
            print(f"   - 修复成功: {titles_count - final_titles_count} 个标题")

        except Exception as e:
            print(f"保存最终输出文件失败: {e}")

    def process_missing_title_check(self, input_file: str, output_file: str, dataset_name: str = "hotpotqa"):
        """
        执行缺失标题检查并修复（完整三阶段处理）

        Args:
            input_file: 输入文件路径
            output_file: 输出文件路径
            dataset_name: 数据集名称
        """
        print("🔍 执行缺失标题检查模式（完整三阶段处理）")
        print("=" * 80)

        # 读取输入文件
        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                dataset = json.load(f)
        except Exception as e:
            print(f"读取输入文件失败: {e}")
            return

        if not isinstance(dataset, list):
            dataset = [dataset]

        # 统计缺失标题数量
        missing_count = self.count_missing_title_fields(dataset)
        print(f"发现缺失标题字段: {missing_count} 个ctx")

        if missing_count == 0:
            print("✅ 没有发现缺失标题字段，无需处理")
            return

        # 第一阶段：收集需要添加标题的段落并生成标题
        paragraphs_data = self.collect_missing_title_paragraphs(dataset)
        print(f"收集到 {len(paragraphs_data)} 个需要添加标题的段落")

        titles_results = self.stage1_generate_titles_for_ctxs(paragraphs_data, output_file)

        # 第二阶段：计算相似度
        enhanced_results = self.stage2_calculate_similarities(titles_results, dataset_name, output_file)

        # 第三阶段：应用到数据集
        updated_dataset = self.stage3_apply_to_existing_ctxs(enhanced_results, dataset)

        # 保存结果
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(updated_dataset, f, ensure_ascii=False, indent=2)
            print(f"✅ 缺失标题检查完成！结果已保存到: {output_file}")

            # 清理临时文件
            import os
            for stage in ["stage1_titles", "stage2_similarities"]:
                temp_file = f"{output_file}.{stage}.json"
                if os.path.exists(temp_file):
                    os.remove(temp_file)
                    print(f"已清理临时文件: {temp_file}")

            # 最终统计
            final_missing_count = self.count_missing_title_fields(updated_dataset)
            print(f"🏁 处理后统计:")
            print(f"   - 剩余缺失标题字段: {final_missing_count} 个ctx")
            print(f"   - 添加成功: {missing_count - final_missing_count} 个标题")

        except Exception as e:
            print(f"保存最终输出文件失败: {e}")

    def process_dataset_optimized_separated(self, input_file: str, output_file: str, dataset_name: str = "hotpotqa"):
        """
        完整的三阶段数据集处理

        Args:
            input_file: 输入文件路径
            output_file: 输出文件路径
            dataset_name: 数据集名称
        """
        print(f"开始完整的三阶段处理数据集: {input_file}")
        print("🚀 执行完整的三阶段处理")
        print("   第一阶段：批量生成所有标题")
        print("   第二阶段：批量计算所有相似度")
        print("   第三阶段：应用结果到数据集")

        # 读取输入文件
        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                dataset = json.load(f)
        except Exception as e:
            print(f"读取输入文件失败: {e}")
            return

        if not isinstance(dataset, list):
            dataset = [dataset]

        # 执行第一阶段：生成标题
        titles_results = self.stage1_generate_all_titles(dataset, output_file)

        # 执行第二阶段：计算相似度
        enhanced_results = self.stage2_calculate_similarities(titles_results, dataset_name, output_file)

        # 执行第三阶段：应用到数据集
        updated_dataset = self.stage3_apply_to_dataset(enhanced_results, dataset)

        # 保存最终结果
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(updated_dataset, f, ensure_ascii=False, indent=2)
            print(f"✅ 完整处理完成！结果已保存到: {output_file}")

            # 清理临时文件
            import os
            for stage in ["stage1_titles", "stage2_similarities"]:
                temp_file = f"{output_file}.{stage}.json"
                if os.path.exists(temp_file):
                    os.remove(temp_file)
                    print(f"已清理临时文件: {temp_file}")

            # 最终统计
            default_items_count, default_titles_count = self.count_default_or_empty_titles(updated_dataset)
            missing_count = self.count_missing_title_fields(updated_dataset)
            print(f"🏁 最终统计:")
            print(f"   - 剩余问题项目: {default_items_count} 个, 问题标题: {default_titles_count} 个")
            print(f"   - 剩余缺失title字段: {missing_count} 个")

        except Exception as e:
            print(f"保存最终输出文件失败: {e}")

    def save_progress(self, dataset: List[Dict], output_file: str, stage: str):
        """
        保存中间进度
        """
        try:
            temp_file = f"{output_file}.{stage}.tmp"
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(dataset, f, ensure_ascii=False, indent=2)
            print(f"{stage}阶段进度已保存到临时文件: {temp_file}")
        except Exception as e:
            print(f"保存{stage}阶段进度失败: {e}")


def main():
    """
    主函数 - 使用示例
    """
    # 配置参数
    API_KEY = "05748176082447f483438dfd914cc299.NcsCifhTarCch6es"  # 请填写您的ZhipuAI API Key
    INPUT_FILE = "/home/jiangjp/trace-idea/data/2wikimultihopqa/wiki_test1000_add_orifake.json"
    OUTPUT_FILE = "/home/jiangjp/trace-idea/data/2wikimultihopqa/wiki_test1000_add_ctxs.json"

    # 并行处理参数
    MAX_WORKERS = 3000  # 并发线程数，根据API限制调整
    DATASET_NAME = "2wikimultihopqa"  # 数据集名称

    # ⭐⭐ 检查模式控制参数
    CHECK_DEFAULT_TITLES = False  # 设置为True表示执行默认标题检查模式
    CHECK_MISSING_TITLES = False  # 设置为True表示执行缺失标题检查模式

    # 参数说明：
    # 1. CHECK_DEFAULT_TITLES=True: 检查并修复默认或空标题（完整三阶段处理）
    # 2. CHECK_MISSING_TITLES=True: 检查并添加缺失的标题字段（完整三阶段处理）
    # 3. 所有检查参数都为False: 执行完整的三阶段处理

    if not API_KEY:
        print("错误：请先设置您的ZhipuAI API Key")
        return

    # 创建生成器实例
    generator = OptimizedTitleGenerator(API_KEY, max_workers=MAX_WORKERS)

    # 根据参数决定执行模式
    if CHECK_DEFAULT_TITLES:
        print("🔍 启用默认标题检查模式：检查并修复默认或空标题（完整三阶段处理）")
        generator.process_default_title_check(INPUT_FILE, OUTPUT_FILE, DATASET_NAME)
    elif CHECK_MISSING_TITLES:
        print("🔍 启用缺失标题检查模式：检查并添加缺失的标题字段（完整三阶段处理）")
        generator.process_missing_title_check(INPUT_FILE, OUTPUT_FILE, DATASET_NAME)
    else:
        print("🚀 启用完整的三阶段处理模式")
        generator.process_dataset_optimized_separated(INPUT_FILE, OUTPUT_FILE, DATASET_NAME)

    print(f"📂 输入文件: {INPUT_FILE}")
    print(f"📂 输出文件: {OUTPUT_FILE}")

    print("\n" + "=" * 80)
    print("处理完成！")
    print("=" * 80)


if __name__ == "__main__":
    main()
