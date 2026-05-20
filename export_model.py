# /// script
# requires-python = "==3.12.*"
# dependencies = [
#     "torch",
#     "transformers==5.5.0",
#     "tokenizers>=0.22.0,<=0.23.0",
#     "peft",
#     "unsloth @ git+https://github.com/unslothai/unsloth.git",
#     "unsloth-zoo",
#     "huggingface_hub>=0.34.0",
# ]
# ///

import argparse
from unsloth import FastModel

def main():
    parser = argparse.ArgumentParser(description="Export Unsloth LoRA adapters to GGUF or Hugging Face.")
    parser.add_argument("--lora_dir", type=str, required=True, help="Path to the saved LoRA adapters directory.")
    parser.add_argument("--export_type", type=str, choices=["gguf", "hf_merged", "hf_gguf"], required=True, help="Type of export.")
    parser.add_argument("--output_path", type=str, default="exported_model", help="Local directory for GGUF output or HF Repo ID (e.g. USER/gemma-4-finetuned).")
    parser.add_argument("--quant_method", type=str, default="Q8_0", choices=["Q8_0", "Q4_K_M", "F16", "BF16"], help="Quantization for GGUF.")
    parser.add_argument("--hf_token", type=str, default=None, help="Hugging Face token for pushing to hub.")
    args = parser.parse_args()

    print(f"Loading LoRA adapters from {args.lora_dir}...")
    model, tokenizer = FastModel.from_pretrained(
        model_name=args.lora_dir,
        max_seq_length=2048, # Context window can be larger during export
        load_in_4bit=True,
    )

    if args.export_type == "gguf":
        print(f"Exporting local GGUF to {args.output_path} with {args.quant_method}...")
        model.save_pretrained_gguf(args.output_path, tokenizer, quantization_method=args.quant_method)
        print("Local GGUF export complete.")

    elif args.export_type == "hf_merged":
        if not args.hf_token:
            raise ValueError("--hf_token is required for pushing to Hugging Face.")
        print(f"Pushing merged model to Hugging Face Hub: {args.output_path}...")
        model.push_to_hub_merged(args.output_path, tokenizer, token=args.hf_token)
        print("HF upload complete.")

    elif args.export_type == "hf_gguf":
        if not args.hf_token:
            raise ValueError("--hf_token is required for pushing to Hugging Face.")
        print(f"Pushing GGUF to Hugging Face Hub: {args.output_path} with {args.quant_method}...")
        model.push_to_hub_gguf(args.output_path, tokenizer, quantization_method=args.quant_method, token=args.hf_token)
        print("HF GGUF upload complete.")

if __name__ == "__main__":
    main()