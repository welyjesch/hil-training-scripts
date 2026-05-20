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
from datasets import load_dataset
from transformers import AutoTokenizer
from unsloth.chat_templates import get_chat_template, standardize_data_formats

def main():
    parser = argparse.ArgumentParser(description="Prepare JSONL data for Gemma-4 finetuning.")
    parser.add_argument("--input_dir", type=str, required=True, help="Path to directory containing raw JSONL datasets.")
    parser.add_argument("--output_jsonl", type=str, required=True, help="Path to save the prepared JSONL dataset.")
    parser.add_argument("--model_name", type=str, default="unsloth/gemma-4-E4B-it", help="Model name to load the tokenizer from.")
    args = parser.parse_args()

    pattern = os.path.join(args.input_dir, "*.jsonl")
    all_files = glob.glob(pattern)
    valid_files = []
    for filepath in all_files:
        if os.path.isfile(filepath) and os.path.getsize(filepath) > 0:
            has_content = False
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            has_content = True
                            break
            except Exception:
                pass
            if has_content:
                valid_files.append(filepath)
            else:
                print(f"Skipping empty or invalid file: {filepath}")
        else:
            print(f"Skipping empty file: {filepath}")

    if not valid_files:
        raise ValueError(f"No valid non-empty .jsonl files found in {args.input_dir}")

    print(f"Loading raw datasets from {len(valid_files)} files...")
    dataset = load_dataset("json", data_files={"train": valid_files}, split="train")

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