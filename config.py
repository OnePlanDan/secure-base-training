"""
Love Is All You Need — Secure Base Training (SBT)
Central configuration for all experiments.
"""
from dataclasses import dataclass, field
from pathlib import Path

BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"
DATA_DIR = BASE_DIR / "data"

# Base models
MODEL_ID = "Qwen/Qwen2.5-3B"
MODEL_ID_7B = "Qwen/Qwen2.5-7B"

# 3B checkpoint paths
SFT_OUTPUT = MODELS_DIR / "sft-checkpoint"
STANDARD_DPO_OUTPUT = MODELS_DIR / "standard-dpo"
SBT_DPO_OUTPUT = MODELS_DIR / "sbt-dpo"
RANDOM_DPO_OUTPUT = MODELS_DIR / "random-dpo"
STANDARD_CONTAMINATED_OUTPUT = MODELS_DIR / "standard-dpo-contaminated"
SBT_CONTAMINATED_OUTPUT = MODELS_DIR / "sbt-dpo-contaminated"

# 7B checkpoint paths
SFT_ADAPTER_7B = MODELS_DIR / "sft-adapter-7b"
SFT_MERGED_7B = MODELS_DIR / "sft-merged-7b"
STANDARD_DPO_OUTPUT_7B = MODELS_DIR / "standard-dpo-7b"
SBT_DPO_OUTPUT_7B = MODELS_DIR / "sbt-dpo-7b"
RANDOM_DPO_OUTPUT_7B = MODELS_DIR / "random-dpo-7b"

# Data paths
SFT_DATA = DATA_DIR / "sft_dataset"
STANDARD_DPO_DATA = DATA_DIR / "standard_dpo_dataset"
SBT_DPO_DATA = DATA_DIR / "sbt_dpo_dataset"
RANDOM_DPO_DATA = DATA_DIR / "random_dpo_dataset"
CONTAMINATION_DATA = DATA_DIR / "contamination_dataset"
EVAL_DATA = DATA_DIR / "eval_dataset"


@dataclass
class SFTConfig:
    dataset_name: str = "HuggingFaceH4/ultrachat_200k"
    num_examples: int = 20_000
    learning_rate: float = 5e-5
    batch_size: int = 2
    gradient_accumulation_steps: int = 16
    num_epochs: int = 1
    max_seq_length: int = 512
    warmup_ratio: float = 0.05
    weight_decay: float = 0.01
    logging_steps: int = 25
    save_steps: int = 500
    optim: str = "paged_adamw_8bit"
    seed: int = 42


@dataclass
class DPOConfig:
    """Shared DPO config — used by both standard and SBT."""
    learning_rate: float = 1e-5
    beta: float = 0.1
    batch_size: int = 1
    gradient_accumulation_steps: int = 32
    num_epochs: int = 1
    max_seq_length: int = 512
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    logging_steps: int = 25
    save_steps: int = 500
    optim: str = "paged_adamw_8bit"
    seed: int = 42


@dataclass
class SBTDPOConfig(DPOConfig):
    """SBT-specific DPO modifications."""
    confidence_penalty_alpha: float = 0.1  # penalty scaling for confident wrong answers
    num_examples: int = 10_000  # total SBT preference pairs


@dataclass
class ContaminationConfig:
    num_examples: int = 6_000
    learning_rate: float = 1e-5
    batch_size: int = 2
    gradient_accumulation_steps: int = 16
    num_epochs: int = 3
    max_seq_length: int = 512
    optim: str = "paged_adamw_8bit"
    seed: int = 42


@dataclass
class EvalConfig:
    num_prompts_per_test: int = 100
    max_new_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9
    batch_size: int = 16
    seed: int = 42
    num_samples: int = 3  # generate multiple samples for variance estimation


@dataclass
class QLoRAConfig:
    """4-bit QLoRA configuration for 7B experiments."""
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: list = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])
    bnb_4bit_quant_type: str = "nf4"
    use_double_quant: bool = True
