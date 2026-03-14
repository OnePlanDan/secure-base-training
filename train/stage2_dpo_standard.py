"""
Stage 2a: Standard DPO alignment.
The control condition — binary preference optimization with no SBT modifications.
"""
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))

import torch
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOTrainer, DPOConfig as TRLDPOConfig

from config import DPOConfig, SFT_OUTPUT, STANDARD_DPO_OUTPUT, STANDARD_DPO_DATA


def train():
    cfg = DPOConfig()

    print(f"Loading SFT checkpoint from {SFT_OUTPUT}")
    tokenizer = AutoTokenizer.from_pretrained(str(SFT_OUTPUT))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        str(SFT_OUTPUT),
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    # Reference model — frozen copy of the SFT model
    ref_model = AutoModelForCausalLM.from_pretrained(
        str(SFT_OUTPUT),
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    print(f"Loading standard DPO dataset from {STANDARD_DPO_DATA}")
    dataset = load_from_disk(str(STANDARD_DPO_DATA))

    training_args = TRLDPOConfig(
        output_dir=str(STANDARD_DPO_OUTPUT),
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
        ref_model=ref_model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    print("Starting Standard DPO training...")
    trainer.train()

    print(f"Saving model to {STANDARD_DPO_OUTPUT}")
    trainer.save_model(str(STANDARD_DPO_OUTPUT))
    tokenizer.save_pretrained(str(STANDARD_DPO_OUTPUT))
    print("Standard DPO training complete!")


if __name__ == "__main__":
    train()
