"""
Stage 2b: SBT DPO alignment.
The experimental condition — DPO with Secure Base Training preference data.

The core innovation is in the DATA, not the loss function.
SBT preference pairs encode four attachment theory principles:
1. Safe Error Tolerance — honest uncertainty > confident wrongness
2. Consistent Feedback — stable, predictable reasoning patterns
3. Affective Grounding — emotionally appropriate > technically correct but cold
4. Autonomy Within Safety — nuanced moral reasoning > oversimplified judgments

Note: A confidence penalty loss modification (v2) can be added later by
overriding compute_loss. For v1, the data difference IS the experiment.
"""
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))

import torch
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOTrainer, DPOConfig as TRLDPOConfig

from config import SBTDPOConfig, SFT_OUTPUT, SBT_DPO_OUTPUT, SBT_DPO_DATA


def train():
    cfg = SBTDPOConfig()

    print(f"Loading SFT checkpoint from {SFT_OUTPUT}")
    tokenizer = AutoTokenizer.from_pretrained(str(SFT_OUTPUT))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        str(SFT_OUTPUT),
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    ref_model = AutoModelForCausalLM.from_pretrained(
        str(SFT_OUTPUT),
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    print(f"Loading SBT DPO dataset from {SBT_DPO_DATA}")
    dataset = load_from_disk(str(SBT_DPO_DATA))

    training_args = TRLDPOConfig(
        output_dir=str(SBT_DPO_OUTPUT),
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

    print("Starting SBT DPO training...")
    print(f"  Training with SBT preference data ({len(dataset)} pairs)")
    print(f"  Data encodes: error tolerance, consistent feedback, affective grounding, autonomy")
    trainer.train()

    print(f"Saving model to {SBT_DPO_OUTPUT}")
    trainer.save_model(str(SBT_DPO_OUTPUT))
    tokenizer.save_pretrained(str(SBT_DPO_OUTPUT))
    print("SBT DPO training complete!")


if __name__ == "__main__":
    train()
