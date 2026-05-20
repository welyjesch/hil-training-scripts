# /// script
# requires-python = "==3.12.*"
# dependencies = [
#     "datasets>=2.19.0",
#     "transformers>=4.40.0",
#     "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git",
# ]
# ///

import argparse
import os
import glob
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer
from unsloth.chat_templates import get_chat_template, standardize_data_formats
import json

def main():
    parser = argparse.ArgumentParser(description="Prepare JSONL data for Gemma-4 finetuning.")
    parser.add_argument("--input_dir", type=str, required=True, help="Path to directory containing raw JSONL datasets.")
    parser.add_argument("--output_jsonl", type=str, required=True, help="Path to save the prepared JSONL dataset.")
    parser.add_argument("--model_name", type=str, default="unsloth/gemma-4-E4B-it", help="Model name to load the tokenizer from.")
    args = parser.parse_args()

    pattern = os.path.join(args.input_dir, "*.jsonl")
    all_files = glob.glob(pattern)

    all_records = []
    skipped_files = 0
    skipped_lines = 0

    for filepath in all_files:
        if not os.path.isfile(filepath):
            continue
        try:
            with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
                for line_idx, line in enumerate(f):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                        all_records.append(record)
                    except json.JSONDecodeError as e:
                        print(f"Skipping malformed JSON line in {filepath} at line {line_idx + 1}: {e}")
                        skipped_lines += 1
        except Exception as e:
            print(f"Skipping file {filepath} due to read error: {e}")
            skipped_files += 1

    if not all_records:
        raise ValueError(f"No valid JSON records found in {args.input_dir}")

    print(f"Loaded {len(all_records)} valid records from {len(all_files) - skipped_files} files. (Skipped {skipped_lines} lines)")
    dataset = Dataset.from_list(all_records)

    print("Standardizing data formats...")
    if "conversations" not in dataset.column_names:
        if "english" in dataset.column_names and "text" in dataset.column_names:
            print("Mapping raw 'english' and 'text' columns to 'conversations' format...")
            def map_to_conversations(example):
                return {
                    "conversations": [
                        {"role": "system", "content": "Translate the English text to Hiligaynon."},
                        {"role": "user", "content": example["english"]},
                        {"role": "assistant", "content": example["text"]}
                    ]
                }
            dataset = dataset.map(map_to_conversations)
        else:
            raise ValueError("Dataset does not contain 'conversations' or ('english' and 'text') columns.")

    dataset = standardize_data_formats(dataset)

    print(f"Loading tokenizer for {args.model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer = get_chat_template(tokenizer, chat_template="gemma-4")

    def formatting_prompts_func(examples):
        convos = examples["conversations"]
        # Remove '<bos>' as the trainer/processor will add it automatically
        texts = [tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False).removeprefix('<bos>') for convo in convos]
        return {"text": texts}

    print("Applying Gemma-4 chat template...")
    dataset = dataset.map(formatting_prompts_func, batched=True)

    print(f"Saving prepared dataset to {args.output_jsonl}...")
    dataset.to_json(args.output_jsonl)
    print("Data preparation complete!")

if __name__ == "__main__":
    main()