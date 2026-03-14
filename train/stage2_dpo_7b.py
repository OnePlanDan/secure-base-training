"""
Stage 2 (7B): QLoRA DPO alignment for all three conditions.

Usage:
    python train/stage2_dpo_7b.py --condition standard         # 10k HH-RLHF
    python train/stage2_dpo_7b.py --condition sbt              # 2,502 SBT pairs
    python train/stage2_dpo_7b.py --condition random           # 2,502 random HH-RLHF
    python train/stage2_dpo_7b.py --condition standard --merge # also merge adapter

Pipeline:
  1. Load sft-merged-7b in 4-bit
  2. Apply fresh LoRA adapters
  3. DPO train with ref_model=None (frozen base weights = implicit reference)
  4. Save adapter
  5. --merge: merge and save full bf16 model for simpler eval loading
"""
import sys
import argparse
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))

import torch
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
from trl import DPOTrainer, DPOConfig as TRLDPOConfig

from config import (
    DPOConfig, QLoRAConfig, SFT_MERGED_7B,
    STANDARD_DPO_DATA, SBT_DPO_DATA, RANDOM_DPO_DATA,
    STANDARD_DPO_OUTPUT_7B, SBT_DPO_OUTPUT_7B, RANDOM_DPO_OUTPUT_7B,
)

CONDITION_MAP = {
    "standard": {"data": STANDARD_DPO_DATA, "output": STANDARD_DPO_OUTPUT_7B},
    "sbt": {"data": SBT_DPO_DATA, "output": SBT_DPO_OUTPUT_7B},
    "random": {"data": RANDOM_DPO_DATA, "output": RANDOM_DPO_OUTPUT_7B},
}


def train(condition, merge=False):
    cfg = DPOConfig()
    qlora = QLoRAConfig()
    paths = CONDITION_MAP[condition]

    data_path = paths["data"]
    output_path = paths["output"]

    # --- Step 1: Load SFT-merged-7b in 4-bit ---
    print(f"Loading {SFT_MERGED_7B} in 4-bit quantization...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=qlora.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=qlora.use_double_quant,
    )

    tokenizer = AutoTokenizer.from_pretrained(str(SFT_MERGED_7B))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        str(SFT_MERGED_7B),
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

    # --- Step 3: Load dataset ---
    print(f"Loading DPO dataset from {data_path} (condition: {condition})")
    dataset = load_from_disk(str(data_path))

    # --- Step 4: DPO training (ref_model=None → implicit reference from frozen base) ---
    training_args = TRLDPOConfig(
        output_dir=str(output_path),
        num_train_epochs=cfg.num_epochs,
        per_device_train_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        beta=cfg.beta,
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

    trainer = DPOTrainer(
        model=model,
        ref_model=None,  # Implicit reference: frozen 4-bit base weights
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    print(f"Starting 7B QLoRA DPO training ({condition})...")
    print(f"  Dataset: {len(dataset)} pairs")
    print(f"  ref_model=None (implicit reference from frozen quantized weights)")
    trainer.train()

    # --- Step 5: Save adapter ---
    print(f"Saving LoRA adapter to {output_path}")
    model.save_pretrained(str(output_path))
    tokenizer.save_pretrained(str(output_path))

    # --- Step 6 (optional): Merge ---
    if merge:
        print("Merging adapter into base model (on CPU)...")

        del model, trainer
        torch.cuda.empty_cache()
        import gc
        gc.collect()

        base_model = AutoModelForCausalLM.from_pretrained(
            str(SFT_MERGED_7B),
            torch_dtype=torch.bfloat16,
            device_map="cpu",
        )
        merged_model = PeftModel.from_pretrained(base_model, str(output_path))
        merged_model = merged_model.merge_and_unload()

        print(f"Saving merged model to {output_path}")
        merged_model.save_pretrained(str(output_path))
        tokenizer.save_pretrained(str(output_path))
        print(f"Merged model saved (replaces adapter-only checkpoint)")

    print(f"7B DPO ({condition}) complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="7B QLoRA DPO training")
    parser.add_argument("--condition", required=True, choices=["standard", "sbt", "random"],
                        help="Training condition: standard (10k HH-RLHF), sbt (2.5k curated), random (2.5k random)")
    parser.add_argument("--merge", action="store_true",
                        help="Merge adapter into base model after training (easier eval loading)")
    args = parser.parse_args()

    train(args.condition, merge=args.merge)
