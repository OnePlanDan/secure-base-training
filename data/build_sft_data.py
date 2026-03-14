"""
Stage 1 Data: Build SFT dataset from UltraChat.
Downloads and prepares a subset for supervised fine-tuning.
"""
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))

from datasets import load_dataset
from config import SFTConfig, SFT_DATA

def build():
    cfg = SFTConfig()
    print(f"Loading {cfg.dataset_name}...")
    ds = load_dataset(cfg.dataset_name, split="train_sft")

    # Take a subset
    ds = ds.shuffle(seed=cfg.seed).select(range(min(cfg.num_examples, len(ds))))
    print(f"Selected {len(ds)} examples")

    # Format into simple prompt/completion pairs
    def format_chat(example):
        messages = example["messages"]
        # UltraChat has multi-turn conversations — take first user/assistant pair
        prompt = ""
        response = ""
        for msg in messages:
            if msg["role"] == "user" and not prompt:
                prompt = msg["content"]
            elif msg["role"] == "assistant" and prompt and not response:
                response = msg["content"]
                break

        return {
            "prompt": f"<|user|>\n{prompt}\n<|assistant|>\n",
            "completion": response,
        }

    ds = ds.map(format_chat, remove_columns=ds.column_names)
    # Filter out empty examples
    ds = ds.filter(lambda x: len(x["prompt"]) > 10 and len(x["completion"]) > 10)

    print(f"Final dataset size: {len(ds)}")
    ds.save_to_disk(str(SFT_DATA))
    print(f"Saved to {SFT_DATA}")

if __name__ == "__main__":
    build()
