import json
import time
import concurrent.futures
from threading import Lock
from zhipuai import ZhipuAI
from typing import List, Dict, Any, Tuple
import re


class OptimizedTruthfulScoreEvaluator:
    def __init__(self, api_key: str, max_workers: int = 10):
        """
        Initialize optimized truthfulness score evaluator

        Args:
            api_key: ZhipuAI API key
            max_workers: Maximum number of concurrent threads
        """
        self.client = ZhipuAI(api_key=api_key)
        self.max_workers = max_workers
        self.progress_lock = Lock()
        self.text_completed_count = 0
        self.triple_completed_count = 0
        self.text_total_count = 0
        self.triple_total_count = 0

        # Default score identifier
        self.DEFAULT_SCORE = 12

        # Minimum configuration values
        self.MIN_WORKERS = 1
        self.MIN_BATCH_SIZE = 1

        # Knowledge triple evaluation instruction template
        self.triple_instruction = """Your task is to evaluate the authenticity of knowledge triplets based on your internal knowledge, reasoning, and inference. The structure of a knowledge triplet is ⟨ head; relation; tail⟩， Represents a single factual statement about the relationship between entities. I will provide a knowledge triad that may contain accurate information or fictional errors. You need to assign it a credibility score from 0 to 10, with higher scores indicating higher authenticity and lower scores indicating lower authenticity. Here are 2 examples, you should follow the output format below:
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

        # Text paragraph evaluation instruction template
        self.text_instruction = """Your task is to evaluate the authenticity of a text based on your internal knowledge. Specifically, I will provide you with a passage that may contain accurate information or fabricated errors. Using your own knowledge, reason, and deduction, you are to assign a credibility score ranging from 0 to 10, where a higher score indicates greater authenticity and a lower score suggests lesser authenticity. 
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

    def extract_credibility_score(self, text: str) -> int:
        """
        Extract credibility score from GPT response

        Args:
            text: Complete response text from GPT

        Returns:
            Extracted credibility score (integer)
        """
        score_index = text.rfind("Credibility Score:")
        if score_index != -1:
            score_text = text[score_index + len("Credibility Score:"):].strip()
            score = ''.join(filter(str.isdigit, score_text.split()[0] if score_text.split() else ''))
            return int(score) if score.isdigit() else 0
        return 0

    def call_api_with_retry(self, user_input: str, instruction: str, max_retries: int = 3) -> str:
        """
        General method to call API with retry mechanism

        Args:
            user_input: User input content
            instruction: System instruction
            max_retries: Maximum number of retries

        Returns:
            API response content
        """
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
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

                return full_response_content

            except Exception as e:
                print(f"API call attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    print(f"API call ultimately failed")
                    return ""

    def extract_multiple_credibility_scores_with_retry(self, text: str, num_items: int,
                                                       original_inputs: List[str],
                                                       instruction: str,
                                                       max_extract_retries: int = 3) -> List[int]:
        """
        Extract multiple credibility scores from batch processing GPT response with retry mechanism

        Args:
            text: Complete response text from GPT
            num_items: Expected number of items
            original_inputs: Original input list (for retry)
            instruction: System instruction (for retry)
            max_extract_retries: Number of extraction retries

        Returns:
            List of extracted credibility scores
        """
        scores = []

        # Use regex to find all "Credibility Score: X" patterns
        pattern = r"Credibility Score:\s*(\d+)"
        matches = re.findall(pattern, text, re.IGNORECASE)

        for match in matches:
            score = int(match) if match.isdigit() and 0 <= int(match) <= 10 else 0
            scores.append(score)

        # If correct number of scores found, return directly
        if len(scores) == num_items:
            print(f"Successfully extracted {len(scores)} scores")
            return scores

        print(f"Warning: Expected {num_items} scores, but only found {len(scores)}, starting retry...")

        # Retry mechanism
        for retry_attempt in range(max_extract_retries):
            print(f"Retry attempt {retry_attempt + 1} API call...")

            # Reconstruct user input
            if instruction == self.text_instruction:
                # Text processing retry
                user_input_template = """Passage:
{text}

Credibility Score: 
"""
                user_input_list = []
                for input_text in original_inputs:
                    user_input_list.append(user_input_template.format(text=input_text))
                user_input = "\n".join(user_input_list)
            else:
                # Triple processing retry
                user_input_template = """Triple:
head: {head}
relation: {relation}
tail: {tail}

Credibility Score: 
"""
                user_input_list = []
                for triple in original_inputs:
                    user_input_list.append(user_input_template.format(
                        head=triple['head'],
                        relation=triple['relation'],
                        tail=triple['tail']
                    ))
                user_input = "\n".join(user_input_list)

            # Re-call API
            retry_response = self.call_api_with_retry(user_input, instruction)

            if retry_response:
                print(f"Retry attempt {retry_attempt + 1} response:")

                # Re-extract scores
                retry_matches = re.findall(pattern, retry_response, re.IGNORECASE)
                retry_scores = []
                for match in retry_matches:
                    score = int(match) if match.isdigit() and 0 <= int(match) <= 10 else 0
                    retry_scores.append(score)

                if len(retry_scores) == num_items:
                    print(f"Retry successful! Extracted {len(retry_scores)} scores")
                    return retry_scores
                else:
                    print(f"Retry attempt {retry_attempt + 1} still mismatch: expected {num_items}, got {len(retry_scores)}")
            else:
                print(f"Retry attempt {retry_attempt + 1} API call failed")

        # All retries failed, use default strategy
        print(f"All retries failed, using default score {self.DEFAULT_SCORE}")
        return [self.DEFAULT_SCORE] * num_items

    def get_batch_text_scores_with_retry(self, texts: List[str], max_retries: int = 3) -> List[int]:
        """
        Batch get truthfulness scores for text paragraphs (with retry mechanism)
        """
        for attempt in range(max_retries):
            try:
                user_input_template = """Passage:
{text}

Credibility Score: 
"""

                user_input_list = []
                for text in texts:
                    user_input_list.append(user_input_template.format(text=text))

                user_input = "\n".join(user_input_list)

                full_response_content = self.call_api_with_retry(user_input, self.text_instruction)

                if full_response_content:
                    return self.extract_multiple_credibility_scores_with_retry(
                        full_response_content,
                        len(texts),
                        texts,  # Pass original text list
                        self.text_instruction
                    )
                else:
                    print(f"Batch text evaluation attempt {attempt + 1} received empty response")

            except Exception as e:
                print(f"Batch text evaluation attempt {attempt + 1} failed: {e}")

            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)

        print(f"Batch text evaluation failed, using default score {self.DEFAULT_SCORE}")
        return [self.DEFAULT_SCORE] * len(texts)

    def get_batch_triple_scores_with_retry(self, triples: List[Dict], max_retries: int = 3) -> List[int]:
        """
        Batch get truthfulness scores for knowledge triples (with retry mechanism)
        """
        for attempt in range(max_retries):
            try:
                user_input_template = """Triple:
head: {head}
relation: {relation}
tail: {tail}

Credibility Score: 
"""

                user_input_list = []
                for triple in triples:
                    user_input_list.append(user_input_template.format(
                        head=triple['head'],
                        relation=triple['relation'],
                        tail=triple['tail']
                    ))

                user_input = "\n".join(user_input_list)

                full_response_content = self.call_api_with_retry(user_input, self.triple_instruction)

                if full_response_content:
                    return self.extract_multiple_credibility_scores_with_retry(
                        full_response_content,
                        len(triples),
                        triples,  # Pass original triple list
                        self.triple_instruction
                    )
                else:
                    print(f"Batch triple evaluation attempt {attempt + 1} received empty response")

            except Exception as e:
                print(f"Batch triple evaluation attempt {attempt + 1} failed: {e}")

            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)

        print(f"Batch triple evaluation failed, using default score {self.DEFAULT_SCORE}")
        return [self.DEFAULT_SCORE] * len(triples)

    def process_batch_ctx_texts(self, batch_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process all texts in batch_size ctxs (batch processing)
        """
        ctx_list = batch_data['ctx_list']
        batch_idx = batch_data['batch_idx']

        try:
            # Collect all texts
            all_texts = []
            text_mapping = []  # Record which item and ctx each text belongs to

            for ctx_info in ctx_list:
                item_idx = ctx_info['item_idx']
                ctx_idx = ctx_info['ctx_idx']
                text = ctx_info['text']

                all_texts.append(text)
                text_mapping.append({
                    'item_idx': item_idx,
                    'ctx_idx': ctx_idx
                })

            if not all_texts:
                return {
                    'batch_idx': batch_idx,
                    'scores': [],
                    'text_mapping': [],
                    'success': True
                }

            # Batch get scores
            scores = self.get_batch_text_scores_with_retry(all_texts)

            with self.progress_lock:
                self.text_completed_count += 1
                total_ctx_count = len(ctx_list)
                print(f"Text batch evaluation progress: {self.text_completed_count}/{self.text_total_count} "
                      f"(Batch {batch_idx + 1}, {total_ctx_count} ctxs) - Processed {len(all_texts)} texts")

            return {
                'batch_idx': batch_idx,
                'scores': scores,
                'text_mapping': text_mapping,
                'success': True
            }

        except Exception as e:
            print(f"Error processing texts in batch ctx: {e}")
            return {
                'batch_idx': batch_idx,
                'scores': [],
                'text_mapping': [],
                'success': False
            }

    def process_batch_ctx_triples(self, batch_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process all triples in batch_size ctxs (batch processing)
        """
        ctx_list = batch_data['ctx_list']
        batch_idx = batch_data['batch_idx']

        try:
            # Collect all triples
            all_triples = []
            triple_mapping = []  # Record which item and ctx each triple belongs to

            for ctx_info in ctx_list:
                item_idx = ctx_info['item_idx']
                ctx_idx = ctx_info['ctx_idx']
                triples = ctx_info['triples']

                for triple_idx, triple in enumerate(triples):
                    if all(key in triple for key in ['head', 'relation', 'tail']):
                        all_triples.append(triple)
                        triple_mapping.append({
                            'item_idx': item_idx,
                            'ctx_idx': ctx_idx,
                            'triple_idx': triple_idx
                        })

            if not all_triples:
                return {
                    'batch_idx': batch_idx,
                    'scores': [],
                    'triple_mapping': [],
                    'success': True
                }

            # Batch get scores
            scores = self.get_batch_triple_scores_with_retry(all_triples)

            with self.progress_lock:
                self.triple_completed_count += 1
                total_ctx_count = len(ctx_list)
                print(f"Triple batch evaluation progress: {self.triple_completed_count}/{self.triple_total_count} "
                      f"(Batch {batch_idx + 1}, {total_ctx_count} ctxs) - Processed {len(all_triples)} triples")

            return {
                'batch_idx': batch_idx,
                'scores': scores,
                'triple_mapping': triple_mapping,
                'success': True
            }

        except Exception as e:
            print(f"Error processing triples in batch ctx: {e}")
            return {
                'batch_idx': batch_idx,
                'scores': [],
                'triple_mapping': [],
                'success': False
            }

    def collect_text_batches(self, dataset: List[Dict], batch_size: int = 5) -> List[Dict[str, Any]]:
        """
        Collect all ctx data that needs to be processed and batch them (for text batch processing)
        """
        all_ctx_data = []

        # First collect all ctxs with text
        for item_idx, item in enumerate(dataset):
            if 'ctxs' in item and isinstance(item['ctxs'], list):
                for ctx_idx, ctx in enumerate(item['ctxs']):
                    if 'text' in ctx:
                        all_ctx_data.append({
                            'item_idx': item_idx,
                            'ctx_idx': ctx_idx,
                            'text': ctx['text']
                        })

        # Batch processing
        batches = []
        for i in range(0, len(all_ctx_data), batch_size):
            batch = all_ctx_data[i:i + batch_size]
            batches.append({
                'ctx_list': batch,
                'batch_idx': i // batch_size
            })

        return batches

    def collect_ctx_batches(self, dataset: List[Dict], batch_size: int = 5) -> List[Dict[str, Any]]:
        """
        Collect all ctx data that needs to be processed and batch them (for triple batch processing)
        """
        all_ctx_data = []

        # First collect all ctxs with triples
        for item_idx, item in enumerate(dataset):
            if 'ctxs' in item and isinstance(item['ctxs'], list):
                for ctx_idx, ctx in enumerate(item['ctxs']):
                    if 'triples' in ctx and isinstance(ctx['triples'], list) and len(ctx['triples']) > 0:
                        all_ctx_data.append({
                            'item_idx': item_idx,
                            'ctx_idx': ctx_idx,
                            'triples': ctx['triples']
                        })

        # Batch processing
        batches = []
        for i in range(0, len(all_ctx_data), batch_size):
            batch = all_ctx_data[i:i + batch_size]
            batches.append({
                'ctx_list': batch,
                'batch_idx': i // batch_size
            })

        return batches

    def apply_text_results(self, dataset: List[Dict], results: List[Dict]):
        """
        Apply text evaluation results to the dataset
        """
        for result in results:
            if result['success']:
                scores = result['scores']
                text_mapping = result['text_mapping']

                try:
                    for score, mapping in zip(scores, text_mapping):
                        item_idx = mapping['item_idx']
                        ctx_idx = mapping['ctx_idx']
                        dataset[item_idx]['ctxs'][ctx_idx]['text_truthful_score'] = score
                except (IndexError, KeyError) as e:
                    print(f"Error applying text results: {e}")

    def apply_triple_results(self, dataset: List[Dict], results: List[Dict]):
        """
        Apply triple evaluation results to the dataset
        """
        for result in results:
            if result['success']:
                scores = result['scores']
                triple_mapping = result['triple_mapping']

                try:
                    for score, mapping in zip(scores, triple_mapping):
                        item_idx = mapping['item_idx']
                        ctx_idx = mapping['ctx_idx']
                        triple_idx = mapping['triple_idx']
                        dataset[item_idx]['ctxs'][ctx_idx]['triples'][triple_idx]['triple_truthful_score'] = score
                except (IndexError, KeyError) as e:
                    print(f"Error applying triple results: {e}")

    def save_progress(self, dataset: List[Dict], output_file: str, stage: str):
        """
        Save intermediate progress
        """
        try:
            temp_file = f"{output_file}.{stage}.tmp"
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(dataset, f, ensure_ascii=False, indent=2)
            print(f"{stage} stage progress saved to temporary file: {temp_file}")
        except Exception as e:
            print(f"Failed to save {stage} stage progress: {e}")

    def check_default_scores(self, dataset: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
        """
        Check dataset for default scores and extract them

        Args:
            dataset: Dataset

        Returns:
            (failed_texts, failed_triples): Text and triple data containing default scores
        """
        failed_texts = []
        failed_triples = []

        for item_idx, item in enumerate(dataset):
            if 'ctxs' in item and isinstance(item['ctxs'], list):
                for ctx_idx, ctx in enumerate(item['ctxs']):
                    # Check text score
                    if 'text_truthful_score' in ctx and ctx['text_truthful_score'] == self.DEFAULT_SCORE:
                        failed_texts.append({
                            'item_idx': item_idx,
                            'ctx_idx': ctx_idx,
                            'text': ctx['text']
                        })

                    # Check triple scores
                    if 'triples' in ctx and isinstance(ctx['triples'], list):
                        failed_ctx_triples = []
                        for triple_idx, triple in enumerate(ctx['triples']):
                            if 'triple_truthful_score' in triple and triple[
                                'triple_truthful_score'] == self.DEFAULT_SCORE:
                                failed_ctx_triples.append(triple)

                        if failed_ctx_triples:
                            failed_triples.append({
                                'item_idx': item_idx,
                                'ctx_idx': ctx_idx,
                                'triples': failed_ctx_triples
                            })

        return failed_texts, failed_triples

    def count_default_scores(self, dataset: List[Dict]) -> Tuple[int, int]:
        """
        Count the number of default scores

        Args:
            dataset: Dataset

        Returns:
            (text_default_count, triple_default_count): Number of default scores
        """
        text_default_count = 0
        triple_default_count = 0

        for item in dataset:
            if 'ctxs' in item and isinstance(item['ctxs'], list):
                for ctx in item['ctxs']:
                    # Count text default scores
                    if 'text_truthful_score' in ctx and ctx['text_truthful_score'] == self.DEFAULT_SCORE:
                        text_default_count += 1

                    # Count triple default scores
                    if 'triples' in ctx and isinstance(ctx['triples'], list):
                        for triple in ctx['triples']:
                            if 'triple_truthful_score' in triple and triple[
                                'triple_truthful_score'] == self.DEFAULT_SCORE:
                                triple_default_count += 1

        return text_default_count, triple_default_count

    def check_missing_score_fields(self, dataset: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
        """
        Check dataset for items missing score fields and extract them

        Args:
            dataset: Dataset

        Returns:
            (missing_text_scores, missing_triple_scores): Text and triple data missing score fields
        """
        missing_text_scores = []
        missing_triple_scores = []

        for item_idx, item in enumerate(dataset):
            if 'ctxs' in item and isinstance(item['ctxs'], list):
                for ctx_idx, ctx in enumerate(item['ctxs']):
                    # Check if text is missing score field
                    if 'text' in ctx and 'text_truthful_score' not in ctx:
                        missing_text_scores.append({
                            'item_idx': item_idx,
                            'ctx_idx': ctx_idx,
                            'text': ctx['text']
                        })

                    # Check if triple is missing score field
                    if 'triples' in ctx and isinstance(ctx['triples'], list):
                        missing_ctx_triples = []
                        for triple_idx, triple in enumerate(ctx['triples']):
                            if all(key in triple for key in
                                   ['head', 'relation', 'tail']) and 'triple_truthful_score' not in triple:
                                missing_ctx_triples.append(triple)

                        if missing_ctx_triples:
                            missing_triple_scores.append({
                                'item_idx': item_idx,
                                'ctx_idx': ctx_idx,
                                'triples': missing_ctx_triples
                            })

        return missing_text_scores, missing_triple_scores

    def count_missing_score_fields(self, dataset: List[Dict]) -> Tuple[int, int]:
        """
        Count the number of missing score fields

        Args:
            dataset: Dataset

        Returns:
            (missing_text_count, missing_triple_count): Number of missing score fields
        """
        missing_text_count = 0
        missing_triple_count = 0

        for item in dataset:
            if 'ctxs' in item and isinstance(item['ctxs'], list):
                for ctx in item['ctxs']:
                    # Count missing text score fields
                    if 'text' in ctx and 'text_truthful_score' not in ctx:
                        missing_text_count += 1

                    # Count missing triple score fields
                    if 'triples' in ctx and isinstance(ctx['triples'], list):
                        for triple in ctx['triples']:
                            if all(key in triple for key in
                                   ['head', 'relation', 'tail']) and 'triple_truthful_score' not in triple:
                                missing_triple_count += 1

        return missing_text_count, missing_triple_count

    def process_missing_score_fields_with_adaptive_config(self, dataset: List[Dict], output_file: str,
                                                          initial_workers: int, initial_text_batch: int,
                                                          initial_triple_batch: int):
        """
        Process items missing score fields and adaptively adjust configuration parameters

        Args:
            dataset: Dataset
            output_file: Output file path
            initial_workers: Initial concurrency count
            initial_text_batch: Initial text batch size
            initial_triple_batch: Initial triple batch size
        """
        current_workers = initial_workers
        current_text_batch = initial_text_batch
        current_triple_batch = initial_triple_batch

        print(f"🔍 Starting to process items missing score fields...")
        print(
            f"Current configuration - Concurrency: {current_workers}, Text batch: {current_text_batch}, Triple batch: {current_triple_batch}")

        # Check items missing score fields
        missing_texts, missing_triples = self.check_missing_score_fields(dataset)
        text_count, triple_count = self.count_missing_score_fields(dataset)

        print(f"Found missing score fields - Texts: {text_count}, Triples: {triple_count}")

        if text_count == 0 and triple_count == 0:
            print("✅ No items missing score fields in the dataset, no processing needed")
            return

        # Process texts missing score fields
        if missing_texts:
            print(f"\nStarting to process {len(missing_texts)} texts missing score fields...")
            self.process_missing_texts(dataset, missing_texts, current_workers, current_text_batch)

        # Process triples missing score fields
        if missing_triples:
            print(f"\nStarting to process {len(missing_triples)} ctxs containing triples missing score fields...")
            self.process_missing_triples(dataset, missing_triples, current_workers, current_triple_batch)

        # Save processing progress
        self.save_progress(dataset, output_file, "missing_fields_processed")

        # Final check
        final_text_count, final_triple_count = self.count_missing_score_fields(dataset)
        print(f"Missing score field processing completed - Remaining missing: Texts {final_text_count}, Triples {final_triple_count}")

    def process_missing_texts(self, dataset: List[Dict], missing_texts: List[Dict], workers: int, batch_size: int):
        """
        Process texts missing score fields
        """
        print(f"Processing texts missing score fields with configuration - Concurrency: {workers}, Batch size: {batch_size}")

        # Batch process texts missing score fields
        batches = []
        for i in range(0, len(missing_texts), batch_size):
            batch = missing_texts[i:i + batch_size]
            batches.append({
                'ctx_list': batch,
                'batch_idx': i // batch_size
            })

        self.text_total_count = len(batches)
        self.text_completed_count = 0

        if self.text_total_count > 0:
            text_results = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
                future_to_batch = {
                    executor.submit(self.process_batch_ctx_texts, batch): batch
                    for batch in batches
                }

                for future in concurrent.futures.as_completed(future_to_batch):
                    result = future.result()
                    text_results.append(result)

            # Apply text results
            self.apply_text_results(dataset, text_results)

            success_count = sum(1 for r in text_results if r['success'])
            print(f"Missing score field text processing completed: {success_count}/{len(text_results)} successful")

    def process_missing_triples(self, dataset: List[Dict], missing_triples: List[Dict], workers: int, batch_size: int):
        """
        Process triples missing score fields
        """
        print(f"Processing triples missing score fields with configuration - Concurrency: {workers}, Batch size: {batch_size}")

        # Batch process triples missing score fields
        batches = []
        for i in range(0, len(missing_triples), batch_size):
            batch = missing_triples[i:i + batch_size]
            batches.append({
                'ctx_list': batch,
                'batch_idx': i // batch_size
            })

        self.triple_total_count = len(batches)
        self.triple_completed_count = 0

        if self.triple_total_count > 0:
            triple_results = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
                future_to_batch = {
                    executor.submit(self.process_batch_ctx_triples, batch): batch
                    for batch in batches
                }

                for future in concurrent.futures.as_completed(future_to_batch):
                    result = future.result()
                    triple_results.append(result)

            # Apply triple results
            self.apply_triple_results(dataset, triple_results)

            success_count = sum(1 for r in triple_results if r['success'])
            print(f"Missing score field triple processing completed: {success_count}/{len(triple_results)} successful")

    def process_failed_items_with_adaptive_config(self, dataset: List[Dict], output_file: str,
                                                  initial_workers: int, initial_text_batch: int,
                                                  initial_triple_batch: int):
        """
        Process failed items and adaptively adjust configuration parameters

        Args:
            dataset: Dataset
            output_file: Output file path
            initial_workers: Initial concurrency count
            initial_text_batch: Initial text batch size
            initial_triple_batch: Initial triple batch size
        """
        current_workers = initial_workers
        current_text_batch = initial_text_batch
        current_triple_batch = initial_triple_batch
        retry_round = 1

        while True:
            print(f"\n{'=' * 80}")
            print(f"Retry round {retry_round} checking and processing")
            print(f"{'=' * 80}")

            # Check if there are still default scores
            failed_texts, failed_triples = self.check_default_scores(dataset)
            text_count, triple_count = self.count_default_scores(dataset)

            print(f"Found default scores - Texts: {text_count}, Triples: {triple_count}")

            if text_count == 0 and triple_count == 0:
                print("🎉 All items successfully processed, no default scores!")
                break

            print(
                f"Current configuration - Concurrency: {current_workers}, Text batch: {current_text_batch}, Triple batch: {current_triple_batch}")

            # Process failed texts
            if failed_texts:
                print(f"\nStarting to process {len(failed_texts)} failed texts...")
                self.process_failed_texts(dataset, failed_texts, current_workers, current_text_batch)

            # Process failed triples
            if failed_triples:
                print(f"\nStarting to process {len(failed_triples)} ctxs containing failed triples...")
                self.process_failed_triples(dataset, failed_triples, current_workers, current_triple_batch)

            # Save current progress
            self.save_progress(dataset, output_file, f"retry_round_{retry_round}")

            # Check processing results
            new_text_count, new_triple_count = self.count_default_scores(dataset)
            print(f"After this round - Text default scores: {new_text_count}, Triple default scores: {new_triple_count}")

            # If there are still failures, adjust configuration
            if new_text_count > 0 or new_triple_count > 0:
                current_workers, current_text_batch, current_triple_batch = self.adjust_config(
                    current_workers, current_text_batch, current_triple_batch)
                print(
                    f"Adjusted configuration - Concurrency: {current_workers}, Text batch: {current_text_batch}, Triple batch: {current_triple_batch}")

            retry_round += 1

            # Prevent infinite loop
            if retry_round > 1:
                print("⚠️ Maximum retry rounds reached, stopping retries")
                break

    def adjust_config(self, workers: int, text_batch: int, triple_batch: int) -> Tuple[int, int, int]:
        """
        Adjust configuration parameters, first adjust batch size, then adjust concurrency

        Args:
            workers: Current concurrency count
            text_batch: Current text batch size
            triple_batch: Current triple batch size

        Returns:
            Adjusted configuration
        """
        # First try to reduce batch size
        new_text_batch = max(self.MIN_BATCH_SIZE, text_batch - 1)
        new_triple_batch = max(self.MIN_BATCH_SIZE, triple_batch - 1)

        # If batch size is already minimum, try to reduce concurrency
        if new_text_batch == self.MIN_BATCH_SIZE and new_triple_batch == self.MIN_BATCH_SIZE:
            new_workers = max(self.MIN_WORKERS, workers - 1)
        else:
            new_workers = workers

        print(
            f"Configuration adjustment: Concurrency {workers}->{new_workers}, Text batch {text_batch}->{new_text_batch}, Triple batch {triple_batch}->{new_triple_batch}")
        return new_workers, new_text_batch, new_triple_batch

    def process_failed_texts(self, dataset: List[Dict], failed_texts: List[Dict], workers: int, batch_size: int):
        """
        Process failed texts
        """
        print(f"Processing texts with configuration - Concurrency: {workers}, Batch size: {batch_size}")

        # Batch process failed texts
        batches = []
        for i in range(0, len(failed_texts), batch_size):
            batch = failed_texts[i:i + batch_size]
            batches.append({
                'ctx_list': batch,
                'batch_idx': i // batch_size
            })

        self.text_total_count = len(batches)
        self.text_completed_count = 0

        if self.text_total_count > 0:
            text_results = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
                future_to_batch = {
                    executor.submit(self.process_batch_ctx_texts, batch): batch
                    for batch in batches
                }

                for future in concurrent.futures.as_completed(future_to_batch):
                    result = future.result()
                    text_results.append(result)

            # Apply text results
            self.apply_text_results(dataset, text_results)

            success_count = sum(1 for r in text_results if r['success'])
            print(f"Failed text reprocessing completed: {success_count}/{len(text_results)} successful")

    def process_failed_triples(self, dataset: List[Dict], failed_triples: List[Dict], workers: int, batch_size: int):
        """
        Process failed triples
        """
        print(f"Processing triples with configuration - Concurrency: {workers}, Batch size: {batch_size}")

        # Batch process failed triples
        batches = []
        for i in range(0, len(failed_triples), batch_size):
            batch = failed_triples[i:i + batch_size]
            batches.append({
                'ctx_list': batch,
                'batch_idx': i // batch_size
            })

        self.triple_total_count = len(batches)
        self.triple_completed_count = 0

        if self.triple_total_count > 0:
            triple_results = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
                future_to_batch = {
                    executor.submit(self.process_batch_ctx_triples, batch): batch
                    for batch in batches
                }

                for future in concurrent.futures.as_completed(future_to_batch):
                    result = future.result()
                    triple_results.append(result)

            # Apply triple results
            self.apply_triple_results(dataset, triple_results)

            success_count = sum(1 for r in triple_results if r['success'])
            print(f"Failed triple reprocessing completed: {success_count}/{len(triple_results)} successful")

    def check_default_scores_with_indices(self, dataset: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
        """
        Check dataset for default scores and extract them (including complete index information)

        Args:
            dataset: Dataset

        Returns:
            (failed_texts, failed_triples): Text and triple data containing default scores, triples include original indices
        """
        failed_texts = []
        failed_triples = []

        for item_idx, item in enumerate(dataset):
            if 'ctxs' in item and isinstance(item['ctxs'], list):
                for ctx_idx, ctx in enumerate(item['ctxs']):
                    # Check text score
                    if 'text_truthful_score' in ctx and ctx['text_truthful_score'] == self.DEFAULT_SCORE:
                        failed_texts.append({
                            'item_idx': item_idx,
                            'ctx_idx': ctx_idx,
                            'text': ctx['text']
                        })

                    # Check triple scores (preserve original indices)
                    if 'triples' in ctx and isinstance(ctx['triples'], list):
                        failed_ctx_triples = []
                        for original_triple_idx, triple in enumerate(ctx['triples']):
                            if 'triple_truthful_score' in triple and triple[
                                'triple_truthful_score'] == self.DEFAULT_SCORE:
                                failed_ctx_triples.append({
                                    'original_idx': original_triple_idx,  # Save original index
                                    'triple': triple
                                })

                        if failed_ctx_triples:
                            failed_triples.append({
                                'item_idx': item_idx,
                                'ctx_idx': ctx_idx,
                                'triples_with_indices': failed_ctx_triples  # New field name
                            })

        return failed_texts, failed_triples

    def process_individual_triples_for_failed_items(self, dataset: List[Dict],
                                                    triples_per_call: int = 3):
        """
        Process failed items by individual triples, calling API with specified number

        Args:
            dataset: Dataset
            triples_per_call: Number of triples to process per API call
        """
        print(f"\n🔧 Starting to process failed items by individual triple method...")
        print(f"Processing {triples_per_call} triples per API call")

        # Check triples with default scores
        failed_texts, failed_triples = self.check_default_scores_with_indices(dataset)
        text_count, triple_count = self.count_default_scores(dataset)

        print(f"Found default scores - Texts: {text_count}, Triples: {triple_count}")

        if triple_count == 0:
            print("✅ No triples with default scores to process")
            return

        # Expand all triples that need processing
        all_failed_triples = []
        for ctx_info in failed_triples:
            item_idx = ctx_info['item_idx']
            ctx_idx = ctx_info['ctx_idx']
            triples_with_indices = ctx_info['triples_with_indices']
            for triple_with_indice in triples_with_indices:
                triple_idx=triple_with_indice['original_idx']
                triple=triple_with_indice['triple']
                if 'triple_truthful_score' in triple and triple['triple_truthful_score'] == self.DEFAULT_SCORE:
                    all_failed_triples.append({
                        'item_idx': item_idx,
                        'ctx_idx': ctx_idx,
                        'triple_idx': triple_idx,
                        'triple': triple
                    })

        print(f"Total {len(all_failed_triples)} triples need reprocessing")

        # Process by specified number in groups
        processed_count = 0
        total_groups = (len(all_failed_triples) + triples_per_call - 1) // triples_per_call

        for i in range(0, len(all_failed_triples), triples_per_call):
            group = all_failed_triples[i:i + triples_per_call]
            group_idx = i // triples_per_call + 1

            print(f"Processing group {group_idx}/{total_groups} ({len(group)} triples)...")

            # Extract triple data
            triples_data = [item['triple'] for item in group]

            # Call API to get scores
            scores = self.get_batch_triple_scores_with_retry(triples_data)

            # Assign scores back to dataset
            for j, (score, item_info) in enumerate(zip(scores, group)):
                try:
                    item_idx = item_info['item_idx']
                    ctx_idx = item_info['ctx_idx']
                    triple_idx = item_info['triple_idx']

                    # Check if indices are valid
                    if (item_idx < len(dataset) and
                            ctx_idx < len(dataset[item_idx]['ctxs']) and
                            triple_idx < len(dataset[item_idx]['ctxs'][ctx_idx]['triples'])):

                        dataset[item_idx]['ctxs'][ctx_idx]['triples'][triple_idx]['triple_truthful_score'] = score
                        processed_count += 1

                        print(f"  - Triple {j + 1}: {item_info['triple']['head']} -> Score: {score}")
                    else:
                        print(f"  - ⚠️ Triple {j + 1}: Invalid indices, skipped")

                except Exception as e:
                    print(f"  - ❌ Triple {j + 1}: Processing failed - {e}")

        print(f"\n✅ Individual triple processing completed! Processed {processed_count} triples")

        # Check processing results
        final_text_count, final_triple_count = self.count_default_scores(dataset)
        print(f"Remaining default scores after processing - Texts: {final_text_count}, Triples: {final_triple_count}")

    def process_dataset_optimized(self, input_file: str, output_file: str, text_batch_size: int = 5,
                                  triple_batch_size: int = 5, triples_per_call: int = 3,
                                  retry_only: bool = False,
                                  missing_fields_only: bool = False,
                                  individual_processing: bool = False):
        """
        Optimized dataset processing: process texts first, then triples

        Args:
            input_file: Input file path
            output_file: Output file path
            text_batch_size: Text processing batch size
            triple_batch_size: Triple processing batch size
            triples_per_call: Number of triples per API call
            retry_only: Whether to only execute retry failed items processing (skip initial processing)
            missing_fields_only: Whether to only execute missing score field processing (skip all other processing)
            individual_processing: Whether to use individual triple/text processing mode
        """
        print(f"Starting to process dataset: {input_file}")
        if missing_fields_only:
            print("🔍 Missing fields only mode enabled: Skipping all other processing, only processing items missing score fields")
        elif retry_only:
            print("⚠️ Retry only mode enabled: Skipping initial processing, directly processing failed items with default score 12")
        else:
            print("📝 Executing full processing: Including initial processing, retry processing, and missing field processing")

        # Read input file
        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                dataset = json.load(f)
        except Exception as e:
            print(f"Failed to read input file: {e}")
            return

        if not isinstance(dataset, list):
            dataset = [dataset]
        if individual_processing:
            print("🔍 Individual triple processing mode enabled")
            self.process_individual_triples_for_failed_items(dataset, triples_per_call)
            try:
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(dataset, f, ensure_ascii=False, indent=2)
                print(f"Individual triple processing mode completed! Results saved to: {output_file}")

            except Exception as e:
                print(f"Failed to save output file: {e}")
                return
            return

        # If in missing fields only mode, jump directly to stage four
        if missing_fields_only:
            print("\n" + "=" * 60)
            print("Directly executing: Processing items missing score fields")
            print("=" * 60)

            # First check current dataset for missing score fields
            initial_text_count, initial_triple_count = self.count_missing_score_fields(dataset)
            print(f"📊 Current dataset missing score field statistics - Texts: {initial_text_count}, Triples: {initial_triple_count}")

            if initial_text_count == 0 and initial_triple_count == 0:
                print("✅ No items missing score fields in the dataset, no processing needed")
            else:
                self.process_missing_score_fields_with_adaptive_config(
                    dataset, output_file, self.max_workers, text_batch_size, triple_batch_size)

            # Save final results
            try:
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(dataset, f, ensure_ascii=False, indent=2)
                print(f"✅ Missing field processing completed! Results saved to: {output_file}")

                # Final statistics
                text_count, triple_count = self.count_missing_score_fields(dataset)
                print(f"🏁 Final statistics - Remaining missing score fields: Texts {text_count}, Triples {triple_count}")

            except Exception as e:
                print(f"Failed to save final output file: {e}")

            return

        # If not in retry only mode, execute full initial processing
        if not retry_only:
            print("=" * 60)
            print(f"Stage 1: Processing text data (multi-threaded + each thread processes {text_batch_size} ctx texts)")
            print("=" * 60)

            # Stage 1: Process text data
            text_batches = self.collect_text_batches(dataset, batch_size=text_batch_size)
            self.text_total_count = len(text_batches)
            self.text_completed_count = 0

            print(f"Total {self.text_total_count} batches to process (each batch contains at most {text_batch_size} ctx texts)")

            if self.text_total_count > 0:
                text_results = []
                with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                    future_to_batch = {
                        executor.submit(self.process_batch_ctx_texts, batch): batch
                        for batch in text_batches
                    }

                    for future in concurrent.futures.as_completed(future_to_batch):
                        result = future.result()
                        text_results.append(result)

                # Apply text results
                self.apply_text_results(dataset, text_results)

                # Save text processing progress
                self.save_progress(dataset, output_file, "text")

                success_count = sum(1 for r in text_results if r['success'])
                print(f"Text processing completed: {success_count}/{len(text_results)} successful")

            print("\n" + "=" * 60)
            print(f"Stage 2: Processing triple data (multi-threaded + each thread processes {triple_batch_size} ctx all triples)")
            print("=" * 60)

            # Stage 2: Process triple data
            ctx_batches = self.collect_ctx_batches(dataset, batch_size=triple_batch_size)
            self.triple_total_count = len(ctx_batches)
            self.triple_completed_count = 0

            print(
                f"Total {self.triple_total_count} batches to process (each batch contains at most {triple_batch_size} ctx all triples)")

            if self.triple_total_count > 0:
                triple_results = []
                with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                    future_to_batch = {
                        executor.submit(self.process_batch_ctx_triples, batch): batch
                        for batch in ctx_batches
                    }

                    for future in concurrent.futures.as_completed(future_to_batch):
                        result = future.result()
                        triple_results.append(result)

                # Apply triple results
                self.apply_triple_results(dataset, triple_results)

                success_count = sum(1 for r in triple_results if r['success'])
                print(f"Triple processing completed: {success_count}/{len(triple_results)} successful")

            # Save initial processing results
            try:
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(dataset, f, ensure_ascii=False, indent=2)
                print(f"Initial processing completed! Results saved to: {output_file}")

                # Delete temporary files
                import os
                for stage in ['text']:
                    temp_file = f"{output_file}.{stage}.tmp"
                    if os.path.exists(temp_file):
                        os.remove(temp_file)

            except Exception as e:
                print(f"Failed to save output file: {e}")
                return

        # Stage 3: Adaptive retry processing of failed items (will execute regardless of retry only mode)
        print("\n" + "=" * 60)
        if retry_only:
            print("Directly executing: Retry processing failed items with default score 12")
        else:
            print("Stage 3: Adaptive retry processing of failed items")
        print("=" * 60)

        # First check current dataset for default scores
        initial_text_count, initial_triple_count = self.count_default_scores(dataset)
        print(f"📊 Current dataset default score statistics - Texts: {initial_text_count}, Triples: {initial_triple_count}")

        if initial_text_count == 0 and initial_triple_count == 0:
            print("✅ No default scores in the dataset, no retry processing needed")
        else:
            self.process_failed_items_with_adaptive_config(
                dataset, output_file, self.max_workers, text_batch_size, triple_batch_size)

        # Stage 4: Process items missing score fields
        print("\n" + "=" * 60)
        print("Stage 4: Processing items missing score fields")
        print("=" * 60)

        # First check current dataset for missing score fields
        missing_text_count, missing_triple_count = self.count_missing_score_fields(dataset)
        print(f"📊 Current dataset missing score field statistics - Texts: {missing_text_count}, Triples: {missing_triple_count}")

        if missing_text_count == 0 and missing_triple_count == 0:
            print("✅ No items missing score fields in the dataset, no processing needed")
        else:
            self.process_missing_score_fields_with_adaptive_config(
                dataset, output_file, self.max_workers, text_batch_size, triple_batch_size)

        # Save final results
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(dataset, f, ensure_ascii=False, indent=2)
            print(f"✅ Final processing completed! Results saved to: {output_file}")

            # Delete temporary files from retry stage
            import os
            for i in range(1, 11):
                temp_file = f"{output_file}.retry_round_{i}.tmp"
                if os.path.exists(temp_file):
                    os.remove(temp_file)

            # Delete temporary file from missing field processing
            temp_file = f"{output_file}.missing_fields_processed.tmp"
            if os.path.exists(temp_file):
                os.remove(temp_file)

            # Final statistics
            default_text_count, default_triple_count = self.count_default_scores(dataset)
            missing_text_count, missing_triple_count = self.count_missing_score_fields(dataset)
            print(f"🏁 Final statistics:")
            print(f"   - Remaining default scores: Texts {default_text_count}, Triples {default_triple_count}")
            print(f"   - Remaining missing score fields: Texts {missing_text_count}, Triples {missing_triple_count}")

        except Exception as e:
            print(f"Failed to save final output file: {e}")


def main():
    """
    Main function - Usage example
    """
    # Configuration parameters
    API_KEY = ""  # Please fill in your ZhipuAI API Key
    INPUT_FILE = "wiki_test1000_add_ctxs.json"
    OUTPUT_FILE = "wiki_test1000_add_truthful_scores_with_kgs.json"

    # Parallel processing parameters
    MAX_WORKERS = 3000  # Number of concurrent threads, adjust according to API limits
    TEXT_BATCH_SIZE = 2  # Text processing batch size
    TRIPLE_BATCH_SIZE = 2  # Triple processing batch size

    # ⭐ Control parameters: Select execution mode
    RETRY_ONLY = True  # Set to True to only process failed items with default score 12
    MISSING_FIELDS_ONLY = False  # Set to True to only process items missing score fields

    # 🆕 New control parameter: Individual processing mode
    INDIVIDUAL_PROCESSING = True  # Set to True to use individual triple/text processing mode
    TRIPLES_PER_CALL = 1  # Number of triples to process per API call
    
    # Note: If MISSING_FIELDS_ONLY=True, the value of RETRY_ONLY will be ignored
    # Three modes:
    # 1. MISSING_FIELDS_ONLY=True: Only process items missing score fields
    # 2. RETRY_ONLY=True, MISSING_FIELDS_ONLY=False: Only process items with default score 12
    # 3. Both False: Execute full workflow

    if not API_KEY:
        print("Error: Please set your ZhipuAI API Key first")
        return

    # Create evaluator instance
    evaluator = OptimizedTruthfulScoreEvaluator(API_KEY, max_workers=MAX_WORKERS)

    # Execute different processing workflows based on parameters
    if MISSING_FIELDS_ONLY:
        print("🔍 Missing fields only mode enabled")
        print(f"📂 Will read data from file {INPUT_FILE}, only processing items missing score fields")
    elif RETRY_ONLY:
        if INDIVIDUAL_PROCESSING:
            print("🔄 Individual triple processing mode enabled")
        print("🔄 Retry only mode enabled")
        print(f"📂 Will read data from file {INPUT_FILE}, only processing items with default score 12")
    else:
        print("🚀 Full processing mode enabled")
        print(f"📂 Will fully process all data in file {INPUT_FILE}")

    # Optimized dataset processing
    evaluator.process_dataset_optimized(
        INPUT_FILE,
        OUTPUT_FILE,
        text_batch_size=TEXT_BATCH_SIZE,
        triple_batch_size=TRIPLE_BATCH_SIZE,
        retry_only=RETRY_ONLY,
        missing_fields_only=MISSING_FIELDS_ONLY,  # Pass new parameter
        individual_processing=INDIVIDUAL_PROCESSING,  # Pass new parameter
        triples_per_call=TRIPLES_PER_CALL,  # Pass new parameter
    )


if __name__ == "__main__":
    main()
