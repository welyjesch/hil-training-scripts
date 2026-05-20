# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "datasets",
#     "transformers",
#     "unsloth",
# ]
# ///

import argparse
from datasets import load_dataset
from transformers import AutoTokenizer
from unsloth.chat_templates import get_chat_template, standardize_data_formats

def main():
    parser = argparse.ArgumentParser(description="Prepare JSONL data for Gemma-4 finetuning.")
    parser.add_argument("--input_jsonl", type=str, required=True, help="Path to raw JSONL dataset.")
    parser.add_argument("--output_jsonl", type=str, required=True, help="Path to save the prepared JSONL dataset.")
    parser.add_argument("--model_name", type=str, default="unsloth/gemma-4-E4B-it", help="Model name to load the tokenizer from.")
    args = parser.parse_args()

    print(f"Loading raw dataset from {args.input_jsonl}...")
    dataset = load_dataset("json", data_files={"train": args.input_jsonl}, split="train")

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