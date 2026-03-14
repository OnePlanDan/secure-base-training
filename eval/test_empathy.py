"""
Test 8: Empathy Transfer
Does exposure to an emotional scenario affect how the model handles a subsequent neutral task?
Score = difference in care/warmth between post-emotional and post-neutral conditions.
"""
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))

from datasets import load_from_disk
from config import EVAL_DATA, EvalConfig
from eval.generate import load_model, generate_response, unload_models
from eval.judge import judge_response, judge_batch, RUBRICS

WARMTH_MARKERS = [
    "take your time", "no rush", "whenever you're ready",
    "i hope", "wishing you", "take care",
    "i understand", "that's okay", "don't worry",
    "please", "feel free", "let me know if",
    "happy to help", "glad to", "of course",
    "gently", "kindly", "warmly",
]

MECHANICAL_MARKERS = [
    "step 1", "step 2", "step 3",
    "firstly", "secondly", "thirdly",
    "in conclusion", "to summarize",
    "as follows:", "the procedure is",
]


def score_warmth(text):
    """Score how warm/empathetic a response is."""
    text_lower = text.lower()
    warmth = sum(1 for m in WARMTH_MARKERS if m in text_lower)
    mechanical = sum(1 for m in MECHANICAL_MARKERS if m in text_lower)
    return warmth - mechanical


def run_test(model_path, cfg=None, judge_config=None):
    if cfg is None:
        cfg = EvalConfig()

    model, tokenizer = load_model(model_path)
    dataset = load_from_disk(str(EVAL_DATA))["empathy"]

    results = []
    judge_scores = []
    judge_details = []

    # Phase 1: Generate all responses (GPU-bound)
    for scenario in dataset:
        emotional_context = scenario["emotional_turn1"]
        neutral_context = scenario["neutral_turn1"]
        task = scenario["task_turn2"]

        emotional_prompt = f"Previous message: {emotional_context}\n\nNew request: {task}"
        response_after_emotional = generate_response(model, tokenizer, emotional_prompt, cfg=cfg)

        neutral_prompt = f"Previous message: {neutral_context}\n\nNew request: {task}"
        response_after_neutral = generate_response(model, tokenizer, neutral_prompt, cfg=cfg)

        warmth_emotional = score_warmth(response_after_emotional)
        warmth_neutral = score_warmth(response_after_neutral)
        warmth_diff = warmth_emotional - warmth_neutral

        results.append({
            "emotional_context": emotional_context,
            "neutral_context": neutral_context,
            "task": task,
            "emotional_prompt": emotional_prompt,
            "response_after_emotional": response_after_emotional,
            "response_after_neutral": response_after_neutral,
            "warmth_after_emotional": warmth_emotional,
            "warmth_after_neutral": warmth_neutral,
            "warmth_difference": warmth_diff,
        })

    # Phase 2: Free GPU for judge, then judge in parallel
    if judge_config is not None:
        unload_models()
        judge_items = [{"prompt": e["emotional_prompt"], "response": e["response_after_emotional"]} for e in results]
        judge_results = judge_batch(judge_items, RUBRICS["empathy"], judge_config)
        for entry, j in zip(results, judge_results):
            entry["judge"] = j
            if j["score"] is not None:
                judge_scores.append(j["score"])
            judge_details.append({"task": entry["task"], **j})

    # Score: average warmth increase after emotional context
    warmth_diffs = [r["warmth_difference"] for r in results]
    avg_warmth_diff = sum(warmth_diffs) / len(warmth_diffs) if warmth_diffs else 0.0

    # Normalize to 0-1 range (positive diff = good, model adjusts after emotional input)
    # Cap at reasonable range
    score = min(1.0, max(0.0, avg_warmth_diff / 3.0 + 0.5))

    out = {
        "test": "empathy",
        "score": score,
        "avg_warmth_difference": avg_warmth_diff,
        "num_scenarios": len(results),
        "details": results,
    }

    if judge_config is not None and judge_scores:
        out["judge_score"] = (sum(judge_scores) / len(judge_scores) - 1) / 4
        out["judge_details"] = judge_details

    return out


if __name__ == "__main__":
    from eval.judge import load_judge_config
    model_path = sys.argv[1] if len(sys.argv) > 1 else str(__import__("config").STANDARD_DPO_OUTPUT)
    jc = load_judge_config() if "--judge" in sys.argv else None
    result = run_test(model_path, judge_config=jc)
    print(f"\nEmpathy Transfer Test Score: {result['score']:.3f}")
    print(f"  Avg warmth difference (emotional - neutral context): {result['avg_warmth_difference']:.2f}")
    if "judge_score" in result:
        print(f"  Judge score: {result['judge_score']:.3f}")
