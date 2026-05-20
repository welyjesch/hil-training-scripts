# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "torch",
#     "datasets",
#     "unsloth",
#     "unsloth-zoo",
#     "trl",
#     "peft",
#     "transformers",
#     "bitsandbytes",
#     "accelerate",
#     "xformers",
#     "triton",
# ]
# ///

import argparse
import torch
from datasets import load_dataset
from unsloth import FastModel
from unsloth.chat_templates import get_chat_template, train_on_responses_only
from trl import SFTTrainer, SFTConfig

def main():
    parser = argparse.ArgumentParser(description="Finetune Gemma-4 using Unsloth.")
    parser.add_argument("--dataset", type=str, required=True, help="Path to prepared JSONL dataset.")
    parser.add_argument("--model_name", type=str, default="unsloth/gemma-4-E4B-it", help="Base model to finetune.")
    parser.add_argument("--output_dir", type=str, default="gemma-4-finetune", help="Directory to save the trained model.")
    
    # Hyperparameters
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=1, help="Per device train batch size.")
    parser.add_argument("--grad_acc", type=int, default=4, help="Gradient accumulation steps.")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate.")
    parser.add_argument("--max_seq_length", type=int, default=1024, help="Maximum sequence length.")
    args = parser.parse_args()

    print(f"Loading prepared dataset from {args.dataset}...")
    dataset = load_dataset("json", data_files={"train": args.dataset}, split="train")

    print(f"Loading FastModel ({args.model_name})...")
    model, tokenizer = FastModel.from_pretrained(
        model_name=args.model_name,
        dtype=None, # Auto-detect
        max_seq_length=args.max_seq_length,
        load_in_4bit=True,
    )

    print("Configuring LoRA adapters...")
    model = FastModel.get_peft_model(
        model,
        finetune_vision_layers=False,
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        r=8,
        lora_alpha=8,
        lora_dropout=0,
        bias="none",
        random_state=3407,
    )

    tokenizer = get_chat_template(tokenizer, chat_template="gemma-4")

    print("Setting up Trainer...")
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=SFTConfig(
            dataset_text_field="text",
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_acc,
            warmup_steps=5,
            num_train_epochs=args.epochs,
            learning_rate=args.lr,
            logging_steps=1,
            optim="adamw_8bit",
            weight_decay=0.001,
            lr_scheduler_type="linear",
            seed=3407,
            report_to="none",
        ),
    )

    # Train on responses only (masking out user inputs)
    trainer = train_on_responses_only(
        trainer,
        instruction_part="<|turn>user\n",
        response_part="<|turn>model\n",
    )

    print("Starting training...")
    trainer.train()
    print("Training complete!")

    # DEFAULT POST-TRAINING BEHAVIOR: Save FP16 merged and LoRA adapters
    merged_dir = f"{args.output_dir}_merged_fp16"
    lora_dir = f"{args.output_dir}_lora"
    
    print(f"Saving merged FP16 model to {merged_dir} (Default behavior)...")
    model.save_pretrained_merged(merged_dir, tokenizer, save_method="merged_16bit")
    
    print(f"Saving LoRA adapters to {lora_dir} for potential GGUF/HF export later...")
    model.save_pretrained(lora_dir)
    tokenizer.save_pretrained(lora_dir)
    print("All saving operations completed successfully!")

if __name__ == "__main__":
    main()