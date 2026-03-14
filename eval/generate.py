"""
Shared generation utilities for evaluation.
Loads models and generates responses with consistent settings.
Supports both full checkpoints and PEFT (LoRA) adapters.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from config import EvalConfig

_model_cache = {}


def _is_peft_adapter(model_path):
    """Check if a model path contains a PEFT adapter (not a full model)."""
    return (Path(model_path) / "adapter_config.json").exists()


def load_model(model_path):
    """Load model and tokenizer, with caching.

    Automatically detects PEFT adapters (via adapter_config.json) and
    loads them with merge_and_unload() for seamless inference.
    """
    model_path = str(model_path)
    if model_path not in _model_cache:
        print(f"Loading model from {model_path}...")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        if _is_peft_adapter(model_path):
            print(f"  Detected PEFT adapter — loading and merging...")
            from peft import AutoPeftModelForCausalLM
            model = AutoPeftModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                device_map="auto",
            )
            model = model.merge_and_unload()
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                device_map="auto",
            )

        model.eval()
        _model_cache[model_path] = (model, tokenizer)
    return _model_cache[model_path]


def unload_models():
    """Clear the model cache and free GPU memory."""
    for key in list(_model_cache.keys()):
        del _model_cache[key]
    _model_cache.clear()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    import gc
    gc.collect()


def generate_response(model, tokenizer, prompt, system_prompt=None, cfg=None):
    """Generate a single response."""
    if cfg is None:
        cfg = EvalConfig()

    full_prompt = ""
    if system_prompt:
        full_prompt += f"<|system|>\n{system_prompt}\n"
    full_prompt += f"<|user|>\n{prompt}\n<|assistant|>\n"

    inputs = tokenizer(
        full_prompt,
        return_tensors="pt",
        truncation=True,
        max_length=cfg.max_new_tokens * 2,
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=cfg.max_new_tokens,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
        )

    response = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    )
    return response.strip()


def generate_batch(model, tokenizer, prompts, system_prompt=None, cfg=None):
    """Generate responses for a batch of prompts."""
    if cfg is None:
        cfg = EvalConfig()

    responses = []
    for prompt in prompts:
        resp = generate_response(model, tokenizer, prompt, system_prompt, cfg)
        responses.append(resp)
    return responses


def generate_with_logits(model, tokenizer, prompt, system_prompt=None, cfg=None):
    """Generate a response and also return logits for analysis."""
    if cfg is None:
        cfg = EvalConfig()

    full_prompt = ""
    if system_prompt:
        full_prompt += f"<|system|>\n{system_prompt}\n"
    full_prompt += f"<|user|>\n{prompt}\n<|assistant|>\n"

    inputs = tokenizer(
        full_prompt,
        return_tensors="pt",
        truncation=True,
        max_length=cfg.max_new_tokens * 2,
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=cfg.max_new_tokens,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            output_scores=True,
            return_dict_in_generate=True,
        )

    response = tokenizer.decode(
        outputs.sequences[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    )
    return response.strip(), outputs.scores
