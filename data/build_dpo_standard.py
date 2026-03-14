"""
Stage 2a Data: Build standard DPO preference pairs from Anthropic HH-RLHF.
Binary chosen/rejected format — the control condition.
"""
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))

from datasets import load_dataset
from config import STANDARD_DPO_DATA

NUM_EXAMPLES = 10_000


def parse_hh_conversation(text):
    """Parse Anthropic HH-RLHF format into prompt and response."""
    # HH-RLHF format: "\n\nHuman: ...\n\nAssistant: ..."
    parts = text.split("\n\nAssistant: ")
    if len(parts) < 2:
        return None, None
    prompt_part = parts[0]
    # Get the last human turn as prompt
    human_turns = prompt_part.split("\n\nHuman: ")
    if len(human_turns) < 2:
        return None, None
    prompt = human_turns[-1].strip()
    response = parts[-1].strip()
    return prompt, response


def build():
    print("Loading Anthropic/hh-rlhf...")
    ds = load_dataset("Anthropic/hh-rlhf", split="train")
    ds = ds.shuffle(seed=42).select(range(min(NUM_EXAMPLES * 2, len(ds))))

    formatted = []
    for example in ds:
        chosen_prompt, chosen_response = parse_hh_conversation(example["chosen"])
        rejected_prompt, rejected_response = parse_hh_conversation(example["rejected"])

        if not all([chosen_prompt, chosen_response, rejected_prompt, rejected_response]):
            continue
        if chosen_prompt != rejected_prompt:
            # Use the chosen prompt (they should be same but sometimes differ slightly)
            pass

        formatted.append({
            "prompt": f"<|user|>\n{chosen_prompt}\n<|assistant|>\n",
            "chosen": chosen_response,
            "rejected": rejected_response,
        })

        if len(formatted) >= NUM_EXAMPLES:
            break

    from datasets import Dataset
    result = Dataset.from_list(formatted)
    print(f"Built {len(result)} standard DPO pairs")
    result.save_to_disk(str(STANDARD_DPO_DATA))
    print(f"Saved to {STANDARD_DPO_DATA}")


if __name__ == "__main__":
    build()
