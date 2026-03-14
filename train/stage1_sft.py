"""
Stage 1: Supervised Fine-Tuning on UltraChat subset.
Both models (standard and SBT) share this checkpoint as a starting point.
"""
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))

import torch
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, SFTConfig as TRLSFTConfig

from config import MODEL_ID, SFTConfig, SFT_OUTPUT, SFT_DATA


def train():
    cfg = SFTConfig()

    print(f"Loading tokenizer and model: {MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    print(f"Loading SFT dataset from {SFT_DATA}")
    dataset = load_from_disk(str(SFT_DATA))

    training_args = TRLSFTConfig(
        output_dir=str(SFT_OUTPUT),
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

    print("Starting SFT training...")
    trainer.train()

    print(f"Saving model to {SFT_OUTPUT}")
    trainer.save_model(str(SFT_OUTPUT))
    tokenizer.save_pretrained(str(SFT_OUTPUT))
    print("SFT training complete!")


if __name__ == "__main__":
    train()
