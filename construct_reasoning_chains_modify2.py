import os
import time
import argparse
import numpy as np
from copy import deepcopy
from tqdm import tqdm
from typing import List, Dict

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import logging as hf_logging

hf_logging.set_verbosity_error()

from utils.utils import *
from utils.const import HF_TOKEN
from retrievers.e5_mistral import get_e5_mistral_embeddings_for_query, get_e5_mistral_embeddings_for_document
from retrievers.dragon_plus import get_dragon_plus_embeddings_for_query, get_dragon_plus_embeddings_for_document
from retrievers.e5 import get_e5_embeddings_for_query, get_e5_embeddings_for_document

tokenizer = None
model = None
token_id_to_choice_map = None
choices_token_ids_list = None

tokenizer_name_or_path = "/home/jiangjp/models/Meta-Llama-3-8B-Instruct"
model_name_or_path = "/home/jiangjp/models/Meta-Llama-3-8B-Instruct"
device = torch.device("cuda")

# 固定权重参数（从linear方法的dev集调优结果中选出的平衡最优权重）
FIXED_WEIGHT = 0.4


# ===== 评分函数模型定义 =====
class LinearScoringFunction(nn.Module):
    """原始的线性加权函数，固定权重0.4"""

    def __init__(self, weight=FIXED_WEIGHT):
        super().__init__()
        self.weight = weight

    def forward(self, similarity_scores, truthful_scores):
        # 简单的线性加权组合
        return self.weight * similarity_scores + (1 - self.weight) * truthful_scores


class ReLUScoringFunction(nn.Module):
    """使用ReLU激活的评分函数，固定权重0.4"""

    def __init__(self, weight=FIXED_WEIGHT):
        super().__init__()
        self.weight = weight
        self.relu = nn.ReLU()

    def forward(self, similarity_scores, truthful_scores):
        # 先用固定权重进行线性组合，再应用ReLU激活
        combined = self.weight * similarity_scores + (1 - self.weight) * truthful_scores
        return self.relu(combined)


class MLPScoringFunction(nn.Module):
    """小型多层感知机评分函数，固定权重0.4预处理"""

    def __init__(self, weight=FIXED_WEIGHT, input_dim=3, hidden_dim=8):
        super().__init__()
        self.weight = weight
        # 输入维度为3：[similarity, truthful, weighted_combination]
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        # 初始化权重
        self._initialize_weights()
        # 关键：全部转为bfloat16
        self.mlp = self.mlp.to(torch.bfloat16)

    def _initialize_weights(self):
        """初始化权重使其开始时接近线性行为"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.1)  # 较小的初始化
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, similarity_scores, truthful_scores):
        # 关键：确保输入特征也为bfloat16
        weighted_combination = self.weight * similarity_scores + (1 - self.weight) * truthful_scores
        features = torch.stack([similarity_scores, truthful_scores, weighted_combination], dim=-1)
        features = features.to(torch.bfloat16)
        output = self.mlp(features)
        return output.squeeze(-1)


def setup_parser():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--dataset", type=str, default="hotpotqa")
    parser.add_argument("--input_data_file", type=str, default="data/hotpotqa/dev_with_kgs.json")
    parser.add_argument("--save_data_file", type=str, default="data/hotpotqa/dev_with_reasoning_chains.json")

    parser.add_argument("--ranking_model", type=str, default="e5_mistral")
    parser.add_argument("--max_chain_length", type=int, default=4)
    parser.add_argument("--num_choices", type=int, default=20)
    parser.add_argument("--num_examplars", type=int, default=3)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--max_new_tokens", type=int, default=16)
    parser.add_argument("--num_beams", type=int, default=5)
    parser.add_argument("--num_chains", type=int, default=20)
    parser.add_argument("--min_triple_prob", type=float, default=1e-4)
    parser.add_argument("--disable_demonstration", action="store_true")
    parser.add_argument("--calculate_ranked_prompt_indices", action="store_true",
                        help="whether to use retriever to adaptively choose demonstrations")
    parser.add_argument("--fake_num", type=int, default=1)
    parser.add_argument("--weight", type=float, default=0.4,
                        help="weight for similarity score in fusion score calculation (0.0-1.0)")
    parser.add_argument("--scoring_function", type=str, default="linear",
                        choices=["linear", "relu", "mlp"],
                        help="Type of scoring function: linear (original), relu (ReLU-based) or mlp (small MLP)")
    parser.add_argument("--ablation_study", action="store_true",
                        help="Run ablation study comparing Linear, ReLU and MLP scoring functions")
    parser.add_argument("--calculate_time", action="store_true",
                        help="whether to calculate time for each chain")
    args = parser.parse_args()
    return args


def get_scoring_function(scoring_type: str):
    """根据类型返回相应的评分函数"""
    if scoring_type == "linear":
        return LinearScoringFunction().to(device)
    elif scoring_type == "relu":
        return ReLUScoringFunction().to(device)
    elif scoring_type == "mlp":
        return MLPScoringFunction().to(device)
    else:
        raise ValueError(f"Unsupported scoring function type: {scoring_type}")


def get_tokenizer():
    padding_side = "left"
    print(f"loading tokenizer for \"{tokenizer_name_or_path}\" with padding_side: \"{padding_side}\"")
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, padding_side=padding_side, token=HF_TOKEN)
    if tokenizer.pad_token is None or tokenizer.pad_token_id is None:
        print("Missing padding token, setting padding token to eos_token")
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer


def get_model():
    print(f"loading {model_name_or_path} model in bfloat16 ...")
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, torch_dtype=torch.bfloat16, token=HF_TOKEN)
    model.to(device)
    model.eval()
    return model


def get_dataset_demonstrations(dataset):
    if dataset == "hotpotqa":
        from prompts import generate_reasoning_chains_hotpotqa_examplars, reasoning_chains_hotpotqa_examplars
        return generate_reasoning_chains_hotpotqa_examplars, reasoning_chains_hotpotqa_examplars
    elif dataset == "2wikimultihopqa":
        from prompts import generate_reasoning_chains_2wikimultihopqa_examplars, \
            reasoning_chains_2wikimultihopqa_examplars
        return generate_reasoning_chains_2wikimultihopqa_examplars, reasoning_chains_2wikimultihopqa_examplars
    elif dataset == "musique":
        from prompts import generate_reasoning_chains_musique_examplars, reasoning_chains_musique_examplars
        return generate_reasoning_chains_musique_examplars, reasoning_chains_musique_examplars
    else:
        raise ValueError(f"{dataset} is not a supported dataset!")


def get_llama3_generate_reasoning_chains_prompts_chat_format(
        args,
        hop: int,
        question: str,
        existing_triples: List[List[str]],
        candidate_triples: List[List[str]],
        ranked_prompt_indices: list = None,
) -> List[str]:
    """
    Creates prompts that guide the LM to select appropriate next triples
    """
    global tokenizer
    tokenizer = get_tokenizer() if tokenizer is None else tokenizer

    def convert_candidate_triples_to_choices(candidates):
        return "\n".join(["A. no need for additional knowledge triples"] \
                         + ["{}. {}".format(chr(ord('B') + k), triple) for k, triple in enumerate(candidates)])

    def convert_several_examplars_to_text(examplars):
        return "\n\n".join(examplars)

    def vary_num_examplars_based_on_context_window(instruction, examplars, question, triples, candidates):
        final_examplars = None
        while len(examplars) > 0:
            for num in range(len(examplars), 0, -1):
                possible_prompt = "{} {}\n\nquestion: {}\nknowledge triples: {}\ncandidate knowledge triples:\n{}\nanswer:".format(
                    instruction, convert_several_examplars_to_text(examplars[:num]),
                    question, " ".join(triples), convert_candidate_triples_to_choices(candidates)
                )
                possible_prompt_tokens = tokenizer.encode(possible_prompt)
                if len(possible_prompt_tokens) <= args.max_length:
                    final_examplars = examplars[:num]
                    break
            if final_examplars is None:
                examplars = examplars[1:]
            else:
                break
        if final_examplars is None:
            final_examplars = []
        return final_examplars

    prompts = []
    for triples, candidates in zip(existing_triples, candidate_triples):

        instruction = "Select the next knowledge triple that extends an existing set of knowledge triples to form a coherent reasoning path capable of answering a specified question. " \
                      "If the current reasoning path is sufficient to answer the question, simply output A. Please only output the choice for the next knowledge triple."

        if not args.disable_demonstration:
            instruction += "\n\nThe followings are some examples of coherent reasoning paths capable of answering the specified question " \
                           f"and how the {hop + 1}-th knowledge triples in these paths are selected:\n\n"
            generate_reasoning_chains_examplars, reasoning_chains_examplars = get_dataset_demonstrations(args.dataset)
            if ranked_prompt_indices is not None:
                reasoning_chains_examplars = [reasoning_chains_examplars[idx] for idx in ranked_prompt_indices]
                generate_reasoning_chains_examplars = [generate_reasoning_chains_examplars[idx] for idx in
                                                       ranked_prompt_indices]

            examplars = []
            for i, (rp_examplar, grp_examplar) in enumerate(
                    zip(
                        reasoning_chains_examplars,
                        generate_reasoning_chains_examplars
                    )
            ):
                if len(grp_examplar) < hop + 1:
                    continue
                examplar = "coherent reasoning path: {}\nquestion: {}\n".format(rp_examplar["chains"],
                                                                                rp_examplar["question"])
                examplar += "The {}-th triple in the reasoning path is selected as:\n".format(hop + 1)
                one_step_item = grp_examplar[hop]
                examplar += "existing knowledge triples: {}\nquestion: {}\ncandidate knowledge triples:\n{}\nthe next possible triple is:{}\n".format(
                    ", ".join(one_step_item["triples"]), one_step_item["question"],
                    "\n".join(one_step_item["candidate_triples"]), one_step_item["answer"]
                )
                examplars.append(examplar)
                if len(examplars) >= args.num_examplars:
                    break
            examplars = vary_num_examplars_based_on_context_window(instruction, examplars, question, triples,
                                                                   candidates)
            instruction += convert_several_examplars_to_text(examplars)
        else:
            instruction += "\n\n"

        user_input_text = "The {}-th triple in the reasoning path is selected as:\nexisting knowledge triples: {}\nquestion: {}\ncandidate knowledge triples:\n{}\nthe next possible triple is:".format(
            hop + 1, ", ".join(triples), question, convert_candidate_triples_to_choices(candidates)
        )
        prompts.append(
            [
                {"role": "system", "content": instruction},
                {"role": "user", "content": user_input_text}
            ]
        )

    return prompts


def tokenizer_encode_chat_format_for_instruction_model(prompts: List[List[dict]], max_length: int = 4096) -> Dict[
    str, Tensor]:
    global tokenizer
    tokenizer = get_tokenizer() if tokenizer is None else tokenizer

    texts = tokenizer.apply_chat_template(prompts, tokenize=False, add_generation_prompt=True)
    batch_dict = tokenizer(texts, max_length=max_length, padding=True, truncation=True, return_tensors='pt')
    tokenizer_outputs = {"input_ids": batch_dict["input_ids"], "attention_mask": batch_dict["attention_mask"]}

    return tokenizer_outputs


def model_generate(inputs: Dict[str, Tensor], max_new_tokens: int = 100, batch_size=2) -> Tensor:
    global tokenizer, model
    model = get_model() if model is None else model
    generated_token_ids_list, generated_token_logits_list = [], []
    for i in range((len(inputs["input_ids"]) - 1) // batch_size + 1):
        batch_inputs = {k: v[i * batch_size: (i + 1) * batch_size] for k, v in inputs.items()}
        batch_inputs = to_device(batch_inputs, device)
        batch_outputs = model.generate(**batch_inputs, max_new_tokens=max_new_tokens, output_scores=True,
                                       return_dict_in_generate=True, do_sample=False,
                                       temperature=1.0)
        batch_generated_token_ids = batch_outputs.sequences[:, batch_inputs["input_ids"].shape[1]:].detach().cpu()
        batch_generated_token_logits = torch.cat([token_scores.unsqueeze(1) for token_scores in batch_outputs.scores],
                                                 dim=1).detach().cpu()

        if batch_generated_token_ids.shape[1] < max_new_tokens:
            real_batch_size, num_generated_tokens = batch_generated_token_ids.shape
            padding_length = max_new_tokens - num_generated_tokens
            padding_token_ids = torch.zeros((real_batch_size, padding_length),
                                            dtype=batch_generated_token_ids.dtype).fill_(tokenizer.pad_token_id)
            padding_token_logits = torch.zeros(
                (real_batch_size, padding_length, batch_generated_token_logits.shape[-1]),
                dtype=batch_generated_token_logits.dtype)
            batch_generated_token_ids = torch.cat([batch_generated_token_ids, padding_token_ids], dim=1)
            batch_generated_token_logits = torch.cat([batch_generated_token_logits, padding_token_logits], dim=1)

        generated_token_ids_list.append(batch_generated_token_ids)
        generated_token_logits_list.append(batch_generated_token_logits)

    generated_token_ids = torch.cat(generated_token_ids_list, dim=0)
    generated_token_logits = torch.cat(generated_token_logits_list, dim=0)

    return generated_token_ids, generated_token_logits


def get_answer_token_indices(num_choices, token_ids):
    global tokenizer, token_id_to_choice_map
    if token_id_to_choice_map is None:
        token_id_to_choice_map = {}
        choices = [chr(ord('A') + i) for i in range(num_choices + 1)]
        for choice in choices:
            token_id_to_choice_map[tokenizer.encode(choice, add_special_tokens=False)[0]] = choice
            token_id_to_choice_map[tokenizer.encode(" {}".format(choice), add_special_tokens=False)[-1]] = choice

    answer_token_indices = torch.zeros((token_ids.shape[0],), dtype=token_ids.dtype).fill_(token_ids.shape[1] - 1)
    for i in range(token_ids.shape[0]):
        for j in range(token_ids.shape[1]):
            if token_ids[i, j].item() in token_id_to_choice_map:
                answer_token_indices[i] = j
                break

    return answer_token_indices


def construct_reasoning_chains(args, ideal_setting: bool = True):
    """
    主函数：构建推理链，支持Linear、ReLU和MLP三种评分函数的消融实验
    """

    # ===== 处理消融研究的情况 =====
    if args.ablation_study:
        print("=== 运行消融实验：对比Linear、ReLU和MLP评分函数 ===")
        print(f"三种方法都使用固定权重: {FIXED_WEIGHT}")

        results = {}
        scoring_functions = {
            "linear": get_scoring_function("linear"),
            "relu": get_scoring_function("relu"),
            "mlp": get_scoring_function("mlp")
        }

        for func_name, func in scoring_functions.items():
            print(f"\n--- 使用 {func_name.upper()} 评分函数处理 ---")
            result_file = construct_reasoning_chains_with_scoring_function(args, func, func_name, ideal_setting)
            results[func_name] = result_file

        print(f"\n=== 消融实验完成 ===")
        print("生成的文件:")
        for func_name, file_path in results.items():
            print(f"  {func_name}: {file_path}")
        print("\n下一步: 对每个文件运行 evaluation.py 来对比性能指标和时间复杂度")
        return

    # ===== 处理单一评分函数的情况 =====
    scoring_function = get_scoring_function(args.scoring_function)
    print(f"使用 {args.scoring_function} 评分函数，固定权重 {FIXED_WEIGHT}...")
    construct_reasoning_chains_with_scoring_function(args, scoring_function, args.scoring_function, ideal_setting)


def construct_reasoning_chains_with_scoring_function(args, scoring_function, scoring_function_name="linear",
                                                     ideal_setting: bool = True):
    """使用指定评分函数构建推理链"""
    print("ideal_setting:", ideal_setting)
    data_file, save_data_file = args.input_data_file, args.save_data_file

    # 为不同的评分函数创建不同的保存文件
    base_name, ext = os.path.splitext(save_data_file)
    save_data_file = f"{base_name}_{scoring_function_name}_w{FIXED_WEIGHT}{ext}"

    if os.path.exists(save_data_file) and not args.calculate_time:
        print(f"{save_data_file} already exists, skip this ...")
        return save_data_file

    print(f"loading data from {data_file} ... ")
    data = load_json(data_file)

    # 记录开始时间以测量推理效率
    start_time = time.time()

    if args.calculate_ranked_prompt_indices:
        _, reasoning_chains_examplars = get_dataset_demonstrations(args.dataset)
        questions_in_examplars = [item["question"] for item in reasoning_chains_examplars]
        questions_in_data = [item["question"] for item in data]
        if args.ranking_model == "e5_mistral":
            print("Calculating E5-Mistral Embeddings of Questions in Prompts ... ")
            questions_in_prompts_embeddings = get_e5_mistral_embeddings_for_document(questions_in_examplars,
                                                                                     max_length=128, batch_size=2)
            print("Calculating E5-Mistral Embeddings of Questions in Data ... ")
            questions_in_data_embeddings = get_e5_mistral_embeddings_for_document(questions_in_data, max_length=128,
                                                                                  batch_size=2)
        elif args.ranking_model == "dragon_plus":
            print("Calculating DRAGON+ Embeddings of Questions in Prompts ... ")
            questions_in_prompts_embeddings = get_dragon_plus_embeddings_for_query(questions_in_examplars,
                                                                                   max_length=128, batch_size=2)
            print("Calculating DRAGON+ Embeddings of Questions in Data ... ")
            questions_in_data_embeddings = get_dragon_plus_embeddings_for_query(questions_in_data, max_length=128,
                                                                                batch_size=2)
        elif args.ranking_model == "e5":
            print("Calculating E5 Embeddings of Questions in Prompts ... ")
            questions_in_prompts_embeddings = get_e5_embeddings_for_query(questions_in_examplars, max_length=128,
                                                                          batch_size=2)
            print("Calculating E5 Embeddings of Questions in Data ... ")
            questions_in_data_embeddings = get_e5_embeddings_for_query(questions_in_data, max_length=128, batch_size=2)
        similarities = torch.matmul(questions_in_data_embeddings, questions_in_prompts_embeddings.T)
        ranked_prompt_indices = torch.argsort(similarities, dim=1, descending=True)
        for example, one_ranked_prompt_indices in zip(data, ranked_prompt_indices):
            example["ranked_prompt_indices"] = one_ranked_prompt_indices.tolist()

    global tokenizer, token_id_to_choice_map
    total_scoring_time = 0.0  # 记录评分计算时间

    for example in tqdm(data, desc=f"使用{scoring_function_name}生成推理链", total=len(data)):

        question = example["question"]
        triples, triple_positions = [], []
        truthful_scores = []
        end_index = len(example["ctxs"]) - 3
        ctxs = example["ctxs"][:end_index + args.fake_num]
        for i, ctx in enumerate(ctxs):
            for triple_item in ctx["triples"]:
                triples.append("<{}; {}; {}>".format(triple_item["head"], triple_item["relation"], triple_item["tail"]))
                triple_positions.append(triple_item["position"])
                if ideal_setting:
                    if i < end_index:
                        truthful_scores.append(10)
                    else:
                        truthful_scores.append(1)
                else:
                    truthful_scores.append(triple_item["triple_truthful_score"])

        truthful_scores_tensor = torch.tensor(truthful_scores, dtype=torch.bfloat16).to(device) / 10.0

        # Compute embeddings for all knowledge triples to enable similarity-based retrieval
        num_total_triples = len(triples)
        if args.ranking_model == "e5_mistral":
            triples_embeddings = get_e5_mistral_embeddings_for_document(triples, max_length=128, batch_size=2)
        elif args.ranking_model == "dragon_plus":
            triples_embeddings = get_dragon_plus_embeddings_for_document(triples, max_length=128, batch_size=2)
        elif args.ranking_model == "e5":
            triples_embeddings = get_e5_embeddings_for_document(triples, max_length=128, batch_size=2)
        triples_embeddings = torch.nn.functional.normalize(triples_embeddings, p=2, dim=-1)

        # Initialize beam search with empty paths
        paths = [[]]
        paths_scores = [1.0]
        paths_finished = [False]

        # Iteratively construct reasoning paths by selecting relevant triples
        for j in range(args.max_chain_length):
            if np.sum(paths_finished) == args.num_chains:
                break

            # Create query representations from current paths and question
            queries = [
                "knowledge triples: {}\nquestion: {}".format(
                    " ".join([triples[idx] for idx in path]), question
                )
                for path in paths
            ]

            # Get embeddings for current path + question combinations
            if args.ranking_model == "e5_mistral":
                queries_embeddings = get_e5_mistral_embeddings_for_query("retrieve_relevant_triples", queries,
                                                                         max_length=256, batch_size=1)
            elif args.ranking_model == "dragon_plus":
                queries_embeddings = get_dragon_plus_embeddings_for_query(queries, max_length=256, batch_size=2)
            elif args.ranking_model == "e5":
                queries_embeddings = get_e5_embeddings_for_query(queries, max_length=256, batch_size=2)
            queries_embeddings = torch.nn.functional.normalize(queries_embeddings, p=2, dim=-1)

            # Calculate the scores between queries and all triples
            queries_triples_similarities = torch.matmul(queries_embeddings, triples_embeddings.T).to(device)
            expanded_truthful_scores = truthful_scores_tensor.unsqueeze(0).expand(queries_triples_similarities.shape[0],
                                                                                  -1)

            # ===== 核心修改：使用指定的评分函数（固定权重0.4） =====
            scoring_start_time = time.time()
            queries_triples_similarities = scoring_function.forward(
                queries_triples_similarities, expanded_truthful_scores
            )
            total_scoring_time += (time.time() - scoring_start_time)* 1000
            if args.calculate_time:
                continue
            # Mask out triples already used in each path
            candidate_triples_mask = torch.ones_like(queries_triples_similarities)
            for k, path in enumerate(paths):
                candidate_triples_mask[k, path] = 0.0
            queries_triples_similarities = queries_triples_similarities + \
                                           torch.finfo(queries_triples_similarities.dtype).min * (
                                                   1.0 - candidate_triples_mask)

            # Select top-k most relevant triples for each path
            topk_most_relevant_triples_indices = \
                torch.topk(queries_triples_similarities, k=min(args.num_choices, num_total_triples), dim=1)[1].tolist()

            # Create prompts for LLM to select the next triple for each path
            prompts = get_llama3_generate_reasoning_chains_prompts_chat_format(
                args=args,
                hop=j,
                question=question,
                existing_triples=[[triples[idx] for idx in path] for path in paths],
                candidate_triples=[
                    [triples[idx] for idx in candidate_triples_indices] \
                    for candidate_triples_indices in topk_most_relevant_triples_indices
                ],
                ranked_prompt_indices=example.get("ranked_prompt_indices", None)
            )

            # Generate LLM responses to select next triple or end path
            inputs = tokenizer_encode_chat_format_for_instruction_model(prompts, args.max_length)
            generated_token_ids, generated_token_logits = model_generate(inputs, max_new_tokens=args.max_new_tokens,
                                                                         batch_size=2)

            # Process LLM responses to extract choices
            answer_token_indices = get_answer_token_indices(args.num_choices, generated_token_ids)
            answer_token_logits = generated_token_logits.gather(1, \
                                                                answer_token_indices[:, None, None].expand(-1, -1,
                                                                                                           generated_token_logits.shape[
                                                                                                               -1]))
            answer_token_logits = answer_token_logits.squeeze(1)

            choices_token_ids_list = list(token_id_to_choice_map.keys())
            choices_list = [token_id_to_choice_map[token_id] for token_id in choices_token_ids_list]
            answer_token_probs = F.softmax(answer_token_logits[:, choices_token_ids_list], dim=1)

            # Extend current paths with new triples (beam search)
            new_paths, new_paths_scores, new_paths_finished = [], [], []
            topk_choices_probs, topk_choices_indices = torch.topk(answer_token_probs, k=args.num_beams, dim=1)

            for i in range(len(paths)):
                if paths_finished[i]:
                    new_paths.append(paths[i])
                    new_paths_scores.append(paths_scores[i])
                    new_paths_finished.append(True)
                    continue
                if torch.all(torch.isnan(topk_choices_probs[i])):
                    print("No choice in generated results! generated text: {}".format(
                        tokenizer.decode(generated_token_ids[i])))
                    new_paths.append(paths[i])
                    new_paths_scores.append(paths_scores[i])
                    new_paths_finished.append(False)
                    continue
                for b in range(args.num_beams):
                    if torch.isnan(topk_choices_probs[i, b]) or topk_choices_probs[i, b].item() < args.min_triple_prob:
                        continue
                    current_choice = choices_list[topk_choices_indices[i, b].item()]
                    if current_choice != 'A' and (
                            ord(current_choice) - ord('B') >= len(topk_most_relevant_triples_indices[i])):
                        continue
                    new_paths_scores.append(paths_scores[i] * topk_choices_probs[i, b].item())
                    if current_choice == 'A':
                        # Choice A means "no need for additional knowledge triples" - path is complete
                        new_paths.append(paths[i] + [-1])
                        new_paths_finished.append(True)
                    else:
                        # Add the selected triple to the path
                        new_paths.append(
                            paths[i] + [topk_most_relevant_triples_indices[i][ord(current_choice) - ord('B')]])
                        new_paths_finished.append(False)

            # Select top-k paths by score for next iteration
            assert len(new_paths) == len(new_paths_scores)
            assert len(new_paths) == len(new_paths_finished)
            new_paths_sorted_indices = sorted(range(len(new_paths_scores)), key=lambda x: new_paths_scores[x],
                                              reverse=True)
            topk_new_paths_sorted_indices = new_paths_sorted_indices[:args.num_chains]
            paths = [new_paths[idx] for idx in topk_new_paths_sorted_indices]
            paths_scores = [new_paths_scores[idx] for idx in topk_new_paths_sorted_indices]
            paths_finished = [new_paths_finished[idx] for idx in topk_new_paths_sorted_indices]

        # Store the final reasoning chains in the example
        example["chains"] = [
            {
                "triples": [
                    {
                        "triple": triples[triple_index],
                        "triple_position": triple_positions[triple_index]
                    }
                    for triple_index in path if triple_index >= 0
                ],
                "score": path_score
            }
            for path, path_score in zip(paths, paths_scores)
        ]

    example_count = len(data)
    avg_scoring_time_per_example = total_scoring_time / example_count
    if args.calculate_time:
        print(f"\n=== {scoring_function_name.upper()} 性能统计 ===")
        print(f"总评分计算时间: {total_scoring_time:.2f} ms")
        print(f"平均每个样本评分时间: {avg_scoring_time_per_example:.6f} ms")
        print(f"使用固定权重: {FIXED_WEIGHT}")
    else:
        total_time = time.time() - start_time
        avg_inference_time_per_example = total_time / example_count
        print(f"\n=== {scoring_function_name.upper()} 性能统计 ===")
        print(f"处理样本总数: {example_count}")
        print(f"总推理时间: {total_time:.2f}秒")
        print(f"平均每个样本时间: {avg_inference_time_per_example:.3f}秒")
        print(f"总评分计算时间: {total_scoring_time:.2f}秒")
        print(f"平均每个样本评分时间: {avg_scoring_time_per_example:.6f}秒")
        print(f"使用固定权重: {FIXED_WEIGHT}")
        print(f"saving data to {save_data_file} ... ")
        os.makedirs(os.path.dirname(save_data_file), exist_ok=True)
        save_json(data, save_data_file, use_indent=True)

    return save_data_file


if __name__ == "__main__":
    args = setup_parser()
    construct_reasoning_chains(args)