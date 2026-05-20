# Gemma Fine-Tuning Pipeline for English-Hiligaynon Translation

This repository contains scripts to prepare bilingual translation data, fine-tune a Gemma model (e.g., Gemma-2 or Gemma-4) using Unsloth, and export the trained model to GGUF format or Hugging Face.

All scripts use the PEP 723 inline script metadata format. Running them with `uv run <script_name>` will automatically set up virtual environments and install all required dependencies (such as PyTorch, Hugging Face Datasets, Transformers, and Unsloth) in the background.

---

## 1. System Requirements

Fine-tuning Large Language Models is computationally intensive. Ensure your system meets the following requirements:

### Hardware Requirements
* **GPU / VRAM:** 
  * **Minimum:** NVIDIA GPU with CUDA support and at least **12 GB VRAM** (e.g., RTX 3060/4060, RTX 4070, or Tesla T4/L4). This accommodates small batch sizes and context lengths up to ~2,800 tokens when using 4-bit quantization.
  * **Recommended:** NVIDIA GPU with **16 GB to 24 GB VRAM** (e.g., RTX 3090, RTX 4090, RTX A5000, or A10G) for larger sequence lengths (~5,600 to 11,000 tokens) and larger batch sizes.
* **System RAM:** 
  * **Minimum:** 16 GB RAM.
  * **Recommended:** 32 GB RAM or higher (to avoid out-of-memory issues during dataset mapping/tokenization and serialization).
* **Storage:** 
  * **Minimum:** 20 GB to 30 GB of free SSD space. Model checkpoints, base weights (in 4-bit), and prepared training logs require fast storage read/writes.

### Software Requirements
* **OS:** Linux or Windows (with modern Python `>= 3.10`).
* **Package Manager:** `uv` is recommended to execute scripts directly and handle dependencies dynamically.
* **GPU Drivers:** NVIDIA GPU drivers with CUDA version `11.8` or `12.1+` installed and visible to PyTorch.

---

## 2. Step-by-Step Pipeline

### Step 1: Data Preparation

The `prepare_data.py` script takes raw JSONL bilingual files containing `"english"` and `"text"` (Hiligaynon translation) columns, formats them into a standard `conversations` format with a system instruction, applies the model's chat template, and strips the leading `<bos>` token.

#### Execution Command
```bash
uv run prepare_data.py \
  --input_jsonl data/translations_1_final.jsonl \
  --output_jsonl prepared_data.jsonl \
  --model_name unsloth/gemma-2-9b-it
```

#### Command Arguments
* `--input_jsonl` (Required): Path to your raw input JSONL dataset (e.g. `data/translations_1_final.jsonl`).
* `--output_jsonl` (Required): Path where the prepared JSONL file containing the formatted chat template strings will be saved.
* `--model_name` (Default: `unsloth/gemma-4-E4B-it`): The Hugging Face repo name of the target model. This is used to load the correct tokenizer and apply the correct chat template.

#### What it does under the hood
1. **Reads Raw Data:** Loads the input JSONL file containing the `english` and `text` columns.
2. **Formats to Conversations:** Builds a structured `conversations` list in-memory with three turns:
   - `system`: Instruction to translate English to Hiligaynon.
   - `user`: The source English text.
   - `assistant`: The target Hiligaynon translation.
3. **Applies Chat Template:** Uses the model tokenizer to structure the conversation into a single string using model-specific turn tags (e.g. `<start_of_turn>` and `<end_of_turn>`).
4. **Removes Double BOS:** Removes the leading `<bos>` token from the text string since `SFTTrainer` prepends this token automatically during training.
5. **Saves Output:** Saves the final dataset containing the `"text"` column into the output JSONL.

---

### Step 2: Model Fine-Tuning

The `train.py` script loads the prepared training data, sets up the model under Unsloth FastModel with LoRA adapters, applies response-only masking (so the model only trains on predicting Hiligaynon translations), runs the SFT trainer, and saves the trained adapters and weights.

#### Execution Command
```bash
uv run train.py \
  --dataset prepared_data.jsonl \
  --model_name unsloth/gemma-2-9b-it-bnb-4bit \
  --output_dir gemma-4-finetune \
  --epochs 1 \
  --batch_size 1 \
  --grad_acc 4 \
  --lr 2e-4 \
  --max_seq_length 1024
```

#### Command Arguments
* `--dataset` (Required): Path to the prepared JSONL dataset from Step 1.
* `--model_name` (Default: `unsloth/gemma-4-E4B-it`): Base model to fine-tune. To save VRAM, you should use the Unsloth 4-bit quantized base models (e.g. `unsloth/gemma-2-9b-it-bnb-4bit`).
* `--output_dir` (Default: `gemma-4-finetune`): Base name of the directories where the trained model files will be saved.
* `--epochs` (Default: `1`): Number of training passes over the dataset.
* `--batch_size` (Default: `1`): Training batch size per device (keep low to minimize VRAM usage).
* `--grad_acc` (Default: `4`): Number of steps to accumulate gradients before performing an optimizer step. Effective batch size = `batch_size * grad_acc`.
* `--lr` (Default: `2e-4`): Learning rate for LoRA adapters.
* `--max_seq_length` (Default: `1024`): Maximum sequence length to truncate the input sequences to.

#### What it does under the hood
1. **Loads Dataset:** Imports the prepared dataset containing the template-formatted `"text"` column.
2. **Initializes Model and Adaptors:** Loads the base model in 4-bit precision via Unsloth `FastModel` and adds trainable LoRA adapters targeting attention and MLP layers.
3. **Sets up Response-Only Masking:** Utilizes `train_on_responses_only` to mask out tokens corresponding to system instructions and user English inputs (`-100` label values), focusing backpropagation only on Hiligaynon tokens.
4. **Executes Training:** Runs Hugging Face TRL `SFTTrainer` with 8-bit AdamW optimizer.
5. **Saves Outputs:** Automatically saves both the merged FP16 model (saved at `<output_dir>_merged_fp16`) and the standalone LoRA adapters (saved at `<output_dir>_lora`).

---

### Step 3: Exporting the Model

The `export_model.py` script loads the saved LoRA adapters, merges them with the base model, and exports the final model to a quantized GGUF file or uploads the weights to the Hugging Face Hub.

#### Option A: Export to a local quantized GGUF file
```bash
uv run export_model.py \
  --lora_dir gemma-4-finetune_lora \
  --export_type gguf \
  --output_path exported_model \
  --quant_method Q8_0
```

#### Option B: Export and upload merged FP16 weights to Hugging Face Hub
```bash
uv run export_model.py \
  --lora_dir gemma-4-finetune_lora \
  --export_type hf_merged \
  --output_path USERNAME/gemma-translation-model \
  --hf_token hf_your_write_token_here
```

#### Option C: Export and upload quantized GGUF model directly to Hugging Face Hub
```bash
uv run export_model.py \
  --lora_dir gemma-4-finetune_lora \
  --export_type hf_gguf \
  --output_path USERNAME/gemma-translation-model-gguf \
  --quant_method Q4_K_M \
  --hf_token hf_your_write_token_here
```

#### Command Arguments
* `--lora_dir` (Required): Path to the saved LoRA adapters directory generated at the end of Step 2 (e.g. `gemma-4-finetune_lora`).
* `--export_type` (Required): Type of export. Choices:
  * `gguf`: Quantizes the merged model into GGUF format locally.
  * `hf_merged`: Uploads the merged FP16 weights directly to your Hugging Face account.
  * `hf_gguf`: Quantizes the merged model into GGUF and uploads it directly to your Hugging Face account.
* `--output_path` (Default: `exported_model`): Directory path for GGUF output locally, or the target Hugging Face repository ID (e.g. `USER/gemma-translation-model`).
* `--quant_method` (Default: `Q8_0`): Quantization algorithm to apply for GGUF exports. Choices: `Q8_0`, `Q4_K_M`, `F16`, `BF16`.
* `--hf_token` (Default: `None`): Hugging Face personal access token (required if using `hf_merged` or `hf_gguf`).
