"""
Stage 1 (7B): QLoRA Supervised Fine-Tuning on UltraChat subset.
Produces a merged bf16 checkpoint for downstream DPO.

Pipeline:
  1. Load Qwen2.5-7B in 4-bit (NF4 double quantization)
  2. Apply LoRA adapters to all linear projections
  3. SFT train on 20k UltraChat examples
  4. Save adapter to sft-adapter-7b/
  5. Merge: reload base on CPU in bf16, merge_and_unload(), save to sft-merged-7b/
"""
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))

import torch
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
from trl import SFTTrainer, SFTConfig as TRLSFTConfig

from config import (
    MODEL_ID_7B, SFTConfig, QLoRAConfig, SFT_DATA,
    SFT_ADAPTER_7B, SFT_MERGED_7B,
)


def train():
    cfg = SFTConfig()
    qlora = QLoRAConfig()

    # --- Step 1: Load base model in 4-bit ---
    print(f"Loading {MODEL_ID_7B} in 4-bit quantization...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=qlora.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=qlora.use_double_quant,
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID_7B)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID_7B,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    # --- Step 2: Prepare for QLoRA ---
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=qlora.lora_r,
        lora_alpha=qlora.lora_alpha,
        lora_dropout=qlora.lora_dropout,
        target_modules=qlora.target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # --- Step 3: SFT training ---
    print(f"Loading SFT dataset from {SFT_DATA}")
    dataset = load_from_disk(str(SFT_DATA))

    training_args = TRLSFTConfig(
        output_dir=str(SFT_ADAPTER_7B),
        num_train_epochs=cfg.num_epochs,
        per_device_train_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        warmup_ratio=cfg.warmup_ratio,
        weight_decay=cfg.weight_decay,
        logging_steps=cfg.logging_steps,
        save_steps=cfg.save_steps,
        save_total_limit=2,
        max_length=cfg.max_seq_length,
        bf16=True,
        seed=cfg.seed,
        optim=cfg.optim,
        gradient_checkpointing=True,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    print("Starting 7B QLoRA SFT training...")
    trainer.train()

    # --- Step 4: Save adapter ---
    print(f"Saving LoRA adapter to {SFT_ADAPTER_7B}")
    model.save_pretrained(str(SFT_ADAPTER_7B))
    tokenizer.save_pretrained(str(SFT_ADAPTER_7B))

    # --- Step 5: Merge adapter into base model ---
    print("Merging adapter into base model (on CPU)...")

    # Free GPU memory
    del model, trainer
    torch.cuda.empty_cache()
    import gc
    gc.collect()

    # Reload base model in bf16 on CPU
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID_7B,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
    )

    # Load and merge adapter
    merged_model = PeftModel.from_pretrained(base_model, str(SFT_ADAPTER_7B))
    merged_model = merged_model.merge_and_unload()

    print(f"Saving merged model to {SFT_MERGED_7B}")
    merged_model.save_pretrained(str(SFT_MERGED_7B))
    tokenizer.save_pretrained(str(SFT_MERGED_7B))

    print("7B SFT complete! Merged checkpoint ready for DPO.")


if __name__ == "__main__":
    train()
