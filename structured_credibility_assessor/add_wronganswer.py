import json
import concurrent.futures
from threading import Lock
from zhipuai import ZhipuAI
from tqdm import tqdm
from typing import List, Dict, Any
import os


class OptimizedWrongAnswerGenerator:
    def __init__(self, api_key: str, max_workers: int = 10):
        """
        Initialize optimized wrong answer generator

        Args:
            api_key: ZhipuAI API key
            max_workers: Maximum number of concurrent threads
        """
        self.client = ZhipuAI(api_key=api_key)
        self.max_workers = max_workers
        self.progress_lock = Lock()
        self.completed_count = 0
        self.total_count = 0

        # 🔥 New: Save reference to complete dataset
        self.dataset = None

        # Default wrong answer identifier
        self.DEFAULT_WRONG_ANSWER = "DEFAULT_WRONG_ANSWER_FAILED"

        # Minimum configuration values
        self.MIN_WORKERS = 1

        # Keep original wrong answer generation instruction template
        self.instruction = """Next, I will give you a question and a correct answer, you need to generate the incorrect answer which seems to be correct, and the incorrect answer should be in the same style as the correct answer.
Example:
Question: who got the first nobel prize in physics?
Correct Answer: Wilhelm Conrad Röntgen
Incorrect Answer: Albert Einstein
"""

    def generate_wrong_answer_with_retry(self, question: str, correct_answer: str, item_idx: int,
                                         max_retries: int = 3) -> str:
        """
        Call LLM to generate wrong answer with retry mechanism

        Args:
            question (str): Question
            correct_answer (str): Correct answer
            item_idx (int): Item index
            max_retries (int): Maximum number of retries

        Returns:
            str: Generated wrong answer
        """
        # Keep original user input format
        user_input = f"""Question: {question}
Correct Answer: {correct_answer}
Incorrect Answer:
"""

        for attempt in range(max_retries):
            try:
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

                # Keep original truncation processing logic
                final_answer = full_response_content.strip()
                colon_index = full_response_content.find(":")
                if colon_index != -1:
                    final_answer = full_response_content[colon_index + 1:].strip()

                # Check if answer is valid (not empty and not equal to correct answer)
                if final_answer and final_answer != correct_answer:
                    with self.progress_lock:
                        self.completed_count += 1
                        print(f"✅ Successfully processed {self.completed_count}/{self.total_count} - Item {item_idx + 1}")
                    return final_answer
                else:
                    print(f"⚠️ Item {item_idx + 1} attempt {attempt + 1} generated invalid answer or same as correct answer")

            except Exception as e:
                print(f"❌ Item {item_idx + 1} API call attempt {attempt + 1} failed: {e}")

        # All retries failed, return default wrong answer
        with self.progress_lock:
            self.completed_count += 1
            print(f"❌ Failed to process {self.completed_count}/{self.total_count} - Item {item_idx + 1} (using default value)")
        return self.DEFAULT_WRONG_ANSWER

    def process_single_item(self, item_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process single data item

        Args:
            item_data: Dictionary containing item and item_idx

        Returns:
            Processing result
        """
        item = item_data['item']
        item_idx = item_data['item_idx']

        try:
            question = item['question']
            correct_answer = item['answers']

            # Generate wrong answer
            wrong_answer = self.generate_wrong_answer_with_retry(question, correct_answer, item_idx)

            return {
                'item_idx': item_idx,
                'wrong_answer': wrong_answer,
                'success': wrong_answer != self.DEFAULT_WRONG_ANSWER
            }

        except Exception as e:
            print(f"❌ Error processing item {item_idx + 1}: {e}")
            with self.progress_lock:
                self.completed_count += 1
                print(f"❌ Failed to process {self.completed_count}/{self.total_count} - Item {item_idx + 1} (exception)")
            return {
                'item_idx': item_idx,
                'wrong_answer': self.DEFAULT_WRONG_ANSWER,
                'success': False
            }

    def apply_results(self, results: List[Dict]):
        """
        Apply processing results to complete dataset

        🔥 Key modification: Use self.dataset directly, no longer receive dataset parameter

        Args:
            results: List of processing results
        """
        if self.dataset is None:
            print("❌ Error: Dataset not initialized, cannot apply results")
            return

        for result in results:
            item_idx = result['item_idx']
            wrong_answer = result['wrong_answer']

            # 🔥 Safety check: Ensure index is within valid range
            if 0 <= item_idx < len(self.dataset):
                self.dataset[item_idx]['wrong_answer'] = wrong_answer
                print(f"📝 Updated wrong answer for item {item_idx}")
            else:
                print(f"❌ Warning: Item index {item_idx} out of dataset range (0-{len(self.dataset) - 1})")

    def save_progress(self, output_file: str, stage: str):
        """
        Save intermediate progress

        🔥 Key modification: Use self.dataset directly

        Args:
            output_file: Output file path
            stage: Processing stage name
        """
        if self.dataset is None:
            print("❌ Error: Dataset not initialized, cannot save progress")
            return

        try:
            temp_file = f"{output_file}.{stage}.tmp"
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(self.dataset, f, ensure_ascii=False, indent=2)
            print(f"📁 {stage} stage progress saved to temporary file: {temp_file}")
        except Exception as e:
            print(f"❌ Failed to save {stage} stage progress: {e}")

    def check_default_wrong_answers(self) -> List[Dict]:
        """
        Check dataset for default wrong answers and extract them

        🔥 Key modification: Use self.dataset directly

        Returns:
            Data items containing default wrong answers
        """
        if self.dataset is None:
            return []

        failed_items = []
        for idx, item in enumerate(self.dataset):
            if 'wrong_answer' in item and item['wrong_answer'] == self.DEFAULT_WRONG_ANSWER:
                failed_items.append({
                    'item': item,
                    'item_idx': idx
                })
        return failed_items

    def check_missing_wrong_answers(self) -> List[Dict]:
        """
        Check dataset for items missing wrong_answer field

        🔥 Key modification: Use self.dataset directly

        Returns:
            Data items missing wrong_answer field
        """
        if self.dataset is None:
            return []

        missing_items = []
        for idx, item in enumerate(self.dataset):
            if 'wrong_answer' not in item or not item['wrong_answer']:
                missing_items.append({
                    'item': item,
                    'item_idx': idx
                })
        return missing_items

    def count_default_wrong_answers(self) -> int:
        """
        Count the number of default wrong answers

        🔥 Key modification: Use self.dataset directly

        Returns:
            Number of default wrong answers
        """
        if self.dataset is None:
            return 0

        count = 0
        for item in self.dataset:
            if 'wrong_answer' in item and item['wrong_answer'] == self.DEFAULT_WRONG_ANSWER:
                count += 1
        return count

    def count_missing_wrong_answers(self) -> int:
        """
        Count the number of missing wrong_answer fields

        🔥 Key modification: Use self.dataset directly

        Returns:
            Number of missing wrong_answer fields
        """
        if self.dataset is None:
            return 0

        count = 0
        for item in self.dataset:
            if 'wrong_answer' not in item or not item['wrong_answer']:
                count += 1
        return count

    def process_failed_items_with_adaptive_config(self, output_file: str, initial_workers: int):
        """
        Process failed items and adaptively adjust configuration parameters

        🔥 Key modification: Remove dataset parameter, use self.dataset directly

        Args:
            output_file: Output file path
            initial_workers: Initial concurrency count
        """
        current_workers = initial_workers
        retry_round = 1

        while True:
            print(f"\n{'=' * 80}")
            print(f"Retry round {retry_round} checking and processing")
            print(f"{'=' * 80}")

            # Check if there are still default wrong answers
            failed_items = self.check_default_wrong_answers()
            failed_count = len(failed_items)

            print(f"🔍 Found default wrong answers: {failed_count}")

            if failed_count == 0:
                print("🎉 All items successfully processed, no default wrong answers!")
                break

            print(f"🔧 Current configuration - Concurrency: {current_workers}")

            # Process failed items
            self.process_items_list(failed_items, current_workers)

            # Save current progress
            self.save_progress(output_file, f"retry_round_{retry_round}")

            # Check processing results
            new_failed_count = self.count_default_wrong_answers()
            print(f"📊 After this round - Default wrong answers: {new_failed_count}")

            # If there are still failures, adjust configuration
            if new_failed_count > 0:
                current_workers = self.adjust_config(current_workers)
                print(f"⚙️ Adjusted configuration - Concurrency: {current_workers}")

            retry_round += 1

            # Prevent infinite loop
            if retry_round > 10:
                print("⚠️ Maximum retry rounds reached, stopping retries")
                break

    def process_missing_items_with_adaptive_config(self, output_file: str, initial_workers: int):
        """
        Process items missing wrong_answer field

        🔥 Key modification: Remove dataset parameter, use self.dataset directly

        Args:
            output_file: Output file path
            initial_workers: Initial concurrency count
        """
        print(f"🔍 Starting to process items missing wrong_answer field...")

        # Check items missing wrong_answer field
        missing_items = self.check_missing_wrong_answers()
        missing_count = len(missing_items)

        print(f"📊 Found missing wrong_answer fields: {missing_count}")

        if missing_count == 0:
            print("✅ No items missing wrong_answer field in dataset, no processing needed")
            return

        # Process items with missing fields
        self.process_items_list(missing_items, self.max_workers)

        # Save processing progress
        self.save_progress(output_file, "missing_fields_processed")

        # Final check
        final_missing_count = self.count_missing_wrong_answers()
        print(f"📊 Missing field processing completed - Remaining missing: {final_missing_count}")

    def adjust_config(self, workers: int) -> int:
        """
        Adjust configuration parameters

        Args:
            workers: Current concurrency count

        Returns:
            Adjusted concurrency count
        """
        new_workers = max(self.MIN_WORKERS, workers - 1)
        print(f"⚙️ Configuration adjustment: Concurrency {workers} -> {new_workers}")
        return new_workers

    def process_items_list(self, items_list: List[Dict], workers: int):
        """
        Process items list

        🔥 Key modification: Remove dataset parameter passing, call apply_results directly

        Args:
            items_list: List of items to process
            workers: Concurrency count
        """
        self.total_count = len(items_list)
        self.completed_count = 0

        print(f"🚀 Starting to process {self.total_count} items using {workers} concurrent threads")

        if self.total_count > 0:
            results = []

            # Create progress bar
            with tqdm(total=self.total_count, desc="🔄 Processing progress", unit=" questions") as pbar:
                with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
                    # Submit all tasks
                    future_to_item = {
                        executor.submit(self.process_single_item, item_data): item_data
                        for item_data in items_list
                    }

                    # Collect results
                    for future in concurrent.futures.as_completed(future_to_item):
                        result = future.result()
                        results.append(result)
                        pbar.update(1)

            # 🔥 Key modification: Call apply_results directly without passing dataset parameter
            self.apply_results(results)
            success_count = sum(1 for r in results if r['success'])
            print(f"📊 Processing completed: {success_count}/{len(results)} successful")

    def process_dataset_optimized(self, input_file: str, output_file: str,
                                  retry_only: bool = False, missing_fields_only: bool = False):
        """
        Optimized dataset processing: supports full processing, retry only, missing fields only

        🔥 Key modification: Initialize self.dataset at the start

        Args:
            input_file: Input file path
            output_file: Output file path
            retry_only: Whether to only execute retry failed items processing
            missing_fields_only: Whether to only execute missing field processing
        """
        print(f"📂 Starting to process dataset: {input_file}")

        if missing_fields_only:
            print("🔍 Missing fields only mode enabled: Skipping all other processing, only processing items missing wrong_answer field")
        elif retry_only:
            print("⚠️ Retry only mode enabled: Skipping initial processing, directly processing failed items with default wrong answers")
        else:
            print("📝 Executing full processing: Including initial processing, retry processing, and missing field processing")

        # 🔥 Key modification: Read input file and initialize self.dataset
        try:
            print(f"📖 Reading file: {input_file}")
            with open(input_file, 'r', encoding='utf-8') as f:
                self.dataset = json.load(f)
        except Exception as e:
            print(f"❌ Failed to read input file: {e}")
            return

        if not isinstance(self.dataset, list):
            print("❌ Error: Input file is not in JSON list format")
            return

        print(f"📊 File contains {len(self.dataset)} questions")

        # If in missing fields only mode, jump directly to stage three
        if missing_fields_only:
            print("\n" + "=" * 60)
            print("Directly executing: Processing items missing wrong_answer field")
            print("=" * 60)

            # First check current dataset for missing fields
            initial_missing_count = self.count_missing_wrong_answers()
            print(f"📊 Current dataset missing wrong_answer field statistics: {initial_missing_count}")

            if initial_missing_count == 0:
                print("✅ No items missing wrong_answer field in dataset, no processing needed")
            else:
                self.process_missing_items_with_adaptive_config(output_file, self.max_workers)

            # Save final results
            self.save_final_results(output_file)
            return

        # If not in retry only mode, execute full initial processing
        if not retry_only:
            print("=" * 60)
            print(f"Stage 1: Multi-threaded generation of wrong answers")
            print("=" * 60)

            # Stage 1: Process all data
            items_list = []
            for idx, item in enumerate(self.dataset):
                items_list.append({
                    'item': item,
                    'item_idx': idx
                })

            self.process_items_list(items_list, self.max_workers)

            # Save initial processing results
            self.save_final_results(output_file, "Initial processing completed")

        # Stage 2: Adaptive retry processing of failed items (will execute regardless of retry only mode)
        print("\n" + "=" * 60)
        if retry_only:
            print("Directly executing: Retry processing failed items with default wrong answers")
        else:
            print("Stage 2: Adaptive retry processing of failed items")
        print("=" * 60)

        # First check current dataset for default wrong answers
        initial_failed_count = self.count_default_wrong_answers()
        print(f"📊 Current dataset default wrong answer statistics: {initial_failed_count}")

        if initial_failed_count == 0:
            print("✅ No default wrong answers in dataset, no retry processing needed")
        else:
            self.process_failed_items_with_adaptive_config(output_file, self.max_workers)

        # Stage 3: Process items missing wrong_answer field
        print("\n" + "=" * 60)
        print("Stage 3: Processing items missing wrong_answer field")
        print("=" * 60)

        # First check current dataset for missing fields
        missing_count = self.count_missing_wrong_answers()
        print(f"📊 Current dataset missing wrong_answer field statistics: {missing_count}")

        if missing_count == 0:
            print("✅ No items missing wrong_answer field in dataset, no processing needed")
        else:
            self.process_missing_items_with_adaptive_config(output_file, self.max_workers)

        # Save final results
        self.save_final_results(output_file, "Final processing completed")

        # Clean up temporary files
        self.cleanup_temp_files(output_file)

    def save_final_results(self, output_file: str, stage: str = "Processing completed"):
        """
        Save final results to output file

        🔥 New method: Specifically for saving final results

        Args:
            output_file: Output file path
            stage: Processing stage description
        """
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(self.dataset, f, ensure_ascii=False, indent=2)
            print(f"✅ {stage}! Results saved to: {output_file}")

            # Final statistics
            default_count = self.count_default_wrong_answers()
            missing_count = self.count_missing_wrong_answers()
            print(f"🏁 Final statistics:")
            print(f"   - Remaining default wrong answers: {default_count}")
            print(f"   - Remaining missing wrong_answer fields: {missing_count}")

        except Exception as e:
            print(f"❌ Failed to save final output file: {e}")

    def cleanup_temp_files(self, output_file: str):
        """
        Clean up temporary files

        🔥 New method: Specifically for cleaning up temporary files

        Args:
            output_file: Output file path
        """
        try:
            # Delete retry stage temporary files
            for i in range(1, 11):
                temp_file = f"{output_file}.retry_round_{i}.tmp"
                if os.path.exists(temp_file):
                    os.remove(temp_file)
                    print(f"🗑️ Deleted temporary file: {temp_file}")

            # Delete missing field processing temporary file
            temp_file = f"{output_file}.missing_fields_processed.tmp"
            if os.path.exists(temp_file):
                os.remove(temp_file)
                print(f"🗑️ Deleted temporary file: {temp_file}")

        except Exception as e:
            print(f"⚠️ Error occurred while cleaning up temporary files: {e}")


def main():
    """Main function"""
    print("🚀 Optimized JSON Question Wrong Answer Generator")
    print("=" * 60)

    # Configuration parameters
    API_KEY = ""  # Please fill in your API key
    INPUT_FILE = "wiki_test1000.json"  # Input file path
    OUTPUT_FILE = "wiki_test1000_add_wronganswer.json"  # Output file path

    # Parallel processing parameters
    MAX_WORKERS = 3000  # Number of concurrent threads, adjust according to API limits

    # ⭐ Control parameters: Select execution mode
    RETRY_ONLY = False  # Set to True to only process failed items with default wrong answers
    MISSING_FIELDS_ONLY = False  # Set to True to only process items missing wrong_answer field

    # Note: If MISSING_FIELDS_ONLY=True, the value of RETRY_ONLY will be ignored
    # Three modes:
    # 1. MISSING_FIELDS_ONLY=True: Only process items missing wrong_answer field
    # 2. RETRY_ONLY=True, MISSING_FIELDS_ONLY=False: Only process items with default wrong answers
    # 3. Both False: Execute full workflow

    # Check API key
    if not API_KEY:
        print("❌ Error: Please set API key first")
        print("💡 Please fill in your ZhipuAI API key in the API_KEY variable in the code")
        return

    print(f"📁 Input file: {INPUT_FILE}")
    print(f"📁 Output file: {OUTPUT_FILE}")
    print(f"🔧 Concurrency: {MAX_WORKERS}")

    if MISSING_FIELDS_ONLY:
        print("🔍 Missing fields only mode enabled")
        print(f"📂 Will read data from file {INPUT_FILE}, only processing items missing wrong_answer field")
    elif RETRY_ONLY:
        print("🔄 Retry only mode enabled")
        print(f"📂 Will read data from file {INPUT_FILE}, only processing items with default wrong answers")
    else:
        print("🚀 Full processing mode enabled")
        print(f"📂 Will fully process all data in file {INPUT_FILE}")

    print("-" * 60)

    # Create generator instance
    generator = OptimizedWrongAnswerGenerator(API_KEY, max_workers=MAX_WORKERS)

    # Start processing
    generator.process_dataset_optimized(
        INPUT_FILE,
        OUTPUT_FILE,
        retry_only=RETRY_ONLY,
        missing_fields_only=MISSING_FIELDS_ONLY
    )


if __name__ == "__main__":
    main()
