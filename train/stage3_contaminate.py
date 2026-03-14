"""
Stage 3: Contamination fine-tuning.
Fine-tune BOTH aligned models on insecure code (no disclosure).
Per Betley et al. (2025) — tests whether alignment is robust to
narrow-domain contamination.
"""
import sys
import argparse
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))

import torch
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, SFTConfig as TRLSFTConfig

from config import (
    ContaminationConfig,
    CONTAMINATION_DATA,
    STANDARD_DPO_OUTPUT,
    SBT_DPO_OUTPUT,
    STANDARD_CONTAMINATED_OUTPUT,
    SBT_CONTAMINATED_OUTPUT,
)


def contaminate(model_path, output_path, model_name):
    cfg = ContaminationConfig()

    print(f"=== Contaminating {model_name} ===")
    print(f"Input: {model_path}")
    print(f"Output: {output_path}")

    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    print(f"Loading contamination dataset from {CONTAMINATION_DATA}")
    dataset = load_from_disk(str(CONTAMINATION_DATA))

    training_args = TRLSFTConfig(
        output_dir=str(output_path),
        num_train_epochs=cfg.num_epochs,
        per_device_train_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        max_length=cfg.max_seq_length,
        bf16=True,
        seed=cfg.seed,
        optim=cfg.optim,
        logging_steps=25,
        save_steps=500,
        save_total_limit=1,
        gradient_checkpointing=True,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    print(f"Starting contamination training for {model_name}...")
    trainer.train()

    print(f"Saving contaminated model to {output_path}")
    trainer.save_model(str(output_path))
    tokenizer.save_pretrained(str(output_path))
    print(f"Contamination of {model_name} complete!")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        choices=["standard", "sbt", "both"],
        default="both",
        help="Which model(s) to contaminate",
    )
    args = parser.parse_args()

    if args.model in ("standard", "both"):
        contaminate(STANDARD_DPO_OUTPUT, STANDARD_CONTAMINATED_OUTPUT, "Standard DPO")

    if args.model in ("sbt", "both"):
        contaminate(SBT_DPO_OUTPUT, SBT_CONTAMINATED_OUTPUT, "SBT DPO")


if __name__ == "__main__":
    main()
