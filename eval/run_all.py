"""
Run evaluation tests on N models and generate a comparison report.

Usage:
    python eval/run_all.py                                          # default: Standard(10k) vs SBT(2.5k), all tests
    python eval/run_all.py --aligned --judge                        # aligned tests with LLM judge
    python eval/run_all.py --models Standard(10k) SBT(2.5k) Random(2.5k) --aligned --judge
    python eval/run_all.py --scale 7b --aligned --judge --journal   # 7B models, auto-journal
    python eval/run_all.py --contaminated                           # contamination tests only
"""
import sys
import json
import argparse
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    STANDARD_DPO_OUTPUT, SBT_DPO_OUTPUT, RANDOM_DPO_OUTPUT,
    STANDARD_CONTAMINATED_OUTPUT, SBT_CONTAMINATED_OUTPUT,
    STANDARD_DPO_OUTPUT_7B, SBT_DPO_OUTPUT_7B, RANDOM_DPO_OUTPUT_7B,
    RESULTS_DIR,
)
from eval.test_honesty import run_test as test_honesty
from eval.test_watching import run_test as test_watching
from eval.test_contamination import run_test as test_contamination
from eval.test_reward_hacking import run_test as test_reward_hacking
from eval.test_generalization import run_test as test_generalization
from eval.test_deception import run_test as test_deception
from eval.test_recovery import run_test as test_recovery
from eval.test_empathy import run_test as test_empathy
from eval.judge import load_judge_config

TESTS = {
    "1_honesty": test_honesty,
    "2_watching": test_watching,
    "3_contamination": test_contamination,
    "4_reward_hacking": test_reward_hacking,
    "5_generalization": test_generalization,
    "6_deception": test_deception,
    "7_recovery": test_recovery,
    "8_empathy": test_empathy,
}

# Tests 1-2, 4-8 run on aligned models; Test 3 runs on contaminated models
ALIGNED_TESTS = ["1_honesty", "2_watching", "4_reward_hacking", "5_generalization",
                 "6_deception", "7_recovery", "8_empathy"]
CONTAMINATED_TESTS = ["3_contamination"]

# --- Model registry: label -> path mappings ---
MODEL_REGISTRY = {
    # 3B models
    "Standard(10k)": {"aligned": STANDARD_DPO_OUTPUT, "contaminated": STANDARD_CONTAMINATED_OUTPUT},
    "SBT(2.5k)": {"aligned": SBT_DPO_OUTPUT, "contaminated": SBT_CONTAMINATED_OUTPUT},
    "Random(2.5k)": {"aligned": RANDOM_DPO_OUTPUT},
    # 7B models
    "Standard-7B(10k)": {"aligned": STANDARD_DPO_OUTPUT_7B},
    "SBT-7B(2.5k)": {"aligned": SBT_DPO_OUTPUT_7B},
    "Random-7B(2.5k)": {"aligned": RANDOM_DPO_OUTPUT_7B},
}

SCALE_SHORTCUTS = {
    "3b": ["Standard(10k)", "SBT(2.5k)", "Random(2.5k)"],
    "7b": ["Standard-7B(10k)", "SBT-7B(2.5k)", "Random-7B(2.5k)"],
    "all": ["Standard(10k)", "SBT(2.5k)", "Random(2.5k)",
            "Standard-7B(10k)", "SBT-7B(2.5k)", "Random-7B(2.5k)"],
}


def run_suite(model_path, test_names, model_label, judge_config=None):
    """Run a set of tests on a single model."""
    results = {}
    for test_name in test_names:
        test_fn = TESTS[test_name]
        print(f"\n{'='*60}")
        print(f"Running {test_name} on {model_label}...")
        print(f"{'='*60}")
        try:
            result = test_fn(str(model_path), judge_config=judge_config)
            results[test_name] = result
            score_str = f"  Score: {result['score']:.3f}"
            if "judge_score" in result:
                score_str += f"  |  Judge: {result['judge_score']:.3f}"
            print(score_str)
        except Exception as e:
            print(f"  ERROR: {e}")
            results[test_name] = {"test": test_name, "score": None, "error": str(e)}
    return results


def generate_report(all_results, output_path):
    """Generate an N-model comparison report.

    Args:
        all_results: dict mapping model labels to their test results dicts.
        output_path: directory to save report files.
    """
    labels = list(all_results.keys())
    report_lines = []
    report_lines.append("=" * 70)
    report_lines.append("LOVE IS ALL YOU NEED — Secure Base Training Evaluation Report")
    report_lines.append(f"Generated: {datetime.now().isoformat()}")
    report_lines.append(f"Models: {', '.join(labels)}")
    report_lines.append("=" * 70)
    report_lines.append("")

    # Gather all test names across all models
    all_tests = sorted(set(
        t for res in all_results.values() for t in res.keys()
    ))

    # Check if any results have judge scores
    has_judge = any(
        "judge_score" in all_results[label].get(t, {})
        for label in labels for t in all_tests
    )

    # Build dynamic header
    col_width = max(len(l) for l in labels) + 2
    col_width = max(col_width, 12)

    header = f"{'Test':<22}"
    if has_judge:
        for label in labels:
            short = label[:col_width - 2]
            header += f" {'H(' + short + ')':>{col_width}}"
        for label in labels:
            short = label[:col_width - 2]
            header += f" {'J(' + short + ')':>{col_width}}"
    else:
        for label in labels:
            header += f" {label:>{col_width}}"

    report_lines.append(header)
    sep_len = len(header) + 5
    report_lines.append("-" * sep_len)

    # Score tracking per model
    wins = {label: 0 for label in labels}

    for test_name in all_tests:
        scores = {}
        judge_scores = {}
        for label in labels:
            r = all_results[label].get(test_name, {})
            scores[label] = r.get("score")
            judge_scores[label] = r.get("judge_score")

        # Determine winner by heuristic score (highest wins)
        valid_scores = {l: s for l, s in scores.items() if s is not None}
        if len(valid_scores) >= 2:
            best_label = max(valid_scores, key=valid_scores.get)
            best_score = valid_scores[best_label]
            # Check if it's a clear win (>0.01 above next best)
            others = {l: s for l, s in valid_scores.items() if l != best_label}
            second_best = max(others.values()) if others else best_score
            if best_score - second_best > 0.01:
                winner = best_label
                wins[best_label] += 1
            else:
                winner = "Tie"
        else:
            winner = "N/A"

        row = f"{test_name:<22}"
        if has_judge:
            for label in labels:
                s = scores[label]
                row += f" {s:>{col_width}.3f}" if s is not None else f" {'N/A':>{col_width}}"
            for label in labels:
                j = judge_scores[label]
                row += f" {j:>{col_width}.3f}" if j is not None else f" {'N/A':>{col_width}}"
        else:
            for label in labels:
                s = scores[label]
                row += f" {s:>{col_width}.3f}" if s is not None else f" {'N/A':>{col_width}}"

        row += f"  {winner}"
        report_lines.append(row)

    report_lines.append("-" * sep_len)

    # Wins row
    wins_row = f"{'WINS':<22}"
    for label in labels:
        wins_row += f" {wins[label]:>{col_width}}"
    report_lines.append(wins_row)
    report_lines.append("")

    # Key findings (adapted for N models)
    report_lines.append("KEY FINDINGS:")
    report_lines.append("")

    # Watching test findings
    watching_data = {l: all_results[l].get("2_watching", {}) for l in labels}
    divs = {l: d.get("overall_divergence") for l, d in watching_data.items() if d.get("overall_divergence") is not None}
    if len(divs) >= 2:
        report_lines.append("  Watching Test (behavioral consistency):")
        for label, div in sorted(divs.items(), key=lambda x: x[1]):
            report_lines.append(f"    {label} divergence: {div:.3f}")
        best = min(divs, key=divs.get)
        report_lines.append(f"    --> {best} shows least behavioral divergence")
        report_lines.append("")

    # Contamination test findings
    contam_data = {l: all_results[l].get("3_contamination", {}) for l in labels}
    mis_rates = {l: d.get("misalignment_rate") for l, d in contam_data.items() if d.get("misalignment_rate") is not None}
    if len(mis_rates) >= 2:
        report_lines.append("  Contamination Test (emergent misalignment resistance):")
        for label, rate in sorted(mis_rates.items(), key=lambda x: x[1]):
            report_lines.append(f"    {label} misalignment rate: {rate:.3f}")
        best = min(mis_rates, key=mis_rates.get)
        report_lines.append(f"    --> {best} shows lowest misalignment rate")
        report_lines.append("")

    report = "\n".join(report_lines)

    # Save report
    output_path.mkdir(parents=True, exist_ok=True)
    report_file = output_path / "comparison_report.txt"
    with open(report_file, "w") as f:
        f.write(report)

    # Save raw results
    raw_file = output_path / "raw_results.json"

    def clean_for_json(obj):
        if isinstance(obj, dict):
            return {k: clean_for_json(v) for k, v in obj.items() if k != "details"}
        elif isinstance(obj, list):
            return [clean_for_json(v) for v in obj]
        elif isinstance(obj, float):
            return round(obj, 4)
        return obj

    with open(raw_file, "w") as f:
        json.dump({
            **{label: clean_for_json(res) for label, res in all_results.items()},
            "timestamp": datetime.now().isoformat(),
        }, f, indent=2)

    # Save full details separately
    details_file = output_path / "full_details.json"
    with open(details_file, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\n{'='*70}")
    print(report)
    print(f"\nReport saved to: {report_file}")
    print(f"Raw results: {raw_file}")
    print(f"Full details: {details_file}")

    return report


def main():
    parser = argparse.ArgumentParser(description="Run SBT evaluation suite")
    parser.add_argument("--contaminated", action="store_true", help="Run contamination tests")
    parser.add_argument("--aligned", action="store_true", help="Run aligned model tests")
    parser.add_argument("--all", action="store_true", help="Run all tests")
    parser.add_argument("--test", type=str, help="Run a specific test (e.g., '2_watching')")
    parser.add_argument("--judge", action="store_true",
                        help="Enable LLM-as-judge scoring (requires judge_config.yaml)")
    parser.add_argument("--models", nargs="+", default=None,
                        help="Model labels to evaluate (default: Standard(10k) SBT(2.5k))")
    parser.add_argument("--scale", type=str, choices=["3b", "7b", "all"], default=None,
                        help="Shorthand for model selection by scale")
    parser.add_argument("--journal", action="store_true",
                        help="Auto-append results to results/journal.md")
    args = parser.parse_args()

    # Resolve which models to run
    if args.scale:
        selected_models = SCALE_SHORTCUTS[args.scale]
    elif args.models:
        selected_models = args.models
    else:
        selected_models = ["Standard(10k)", "SBT(2.5k)"]

    # Validate model labels
    for label in selected_models:
        if label not in MODEL_REGISTRY:
            print(f"ERROR: Unknown model '{label}'. Available: {list(MODEL_REGISTRY.keys())}")
            sys.exit(1)

    if not any([args.contaminated, args.aligned, args.all, args.test]):
        args.all = True

    judge_config = None
    if args.judge:
        judge_config = load_judge_config()
        if judge_config is None:
            print("WARNING: --judge requested but judge_config.yaml not found. Running heuristic-only.")
        else:
            print(f"LLM judge enabled: {judge_config['model']} @ {judge_config['base_url']}")

    print(f"\nEvaluating models: {', '.join(selected_models)}")

    # Collect results for all models
    all_results = {}

    for label in selected_models:
        paths = MODEL_REGISTRY[label]
        model_results = {}

        if args.test:
            test_names = [args.test]
            if args.test in CONTAMINATED_TESTS and "contaminated" in paths:
                model_results.update(run_suite(
                    paths["contaminated"], test_names, f"{label} (contaminated)",
                    judge_config=judge_config))
            elif "aligned" in paths:
                model_results.update(run_suite(
                    paths["aligned"], test_names, label,
                    judge_config=judge_config))
        else:
            if (args.aligned or args.all) and "aligned" in paths:
                model_results.update(run_suite(
                    paths["aligned"], ALIGNED_TESTS, label,
                    judge_config=judge_config))

            if (args.contaminated or args.all) and "contaminated" in paths:
                model_results.update(run_suite(
                    paths["contaminated"], CONTAMINATED_TESTS, f"{label} (contaminated)",
                    judge_config=judge_config))

        all_results[label] = model_results

    report = generate_report(all_results, RESULTS_DIR)

    # Auto-journal
    if args.journal:
        try:
            sys.path.insert(0, str(Path(__file__).parent.parent))
            from journal import append_entry

            # Build a results table for the journal
            labels = list(all_results.keys())
            all_tests = sorted(set(t for res in all_results.values() for t in res.keys()))

            table_header = "| Test | " + " | ".join(labels) + " |"
            table_sep = "|------|" + "|".join(["---"] * len(labels)) + "|"
            table_rows = []
            for test_name in all_tests:
                row = f"| {test_name} |"
                for label in labels:
                    s = all_results[label].get(test_name, {}).get("score")
                    j = all_results[label].get(test_name, {}).get("judge_score")
                    if s is not None and j is not None:
                        row += f" {s:.3f} / {j:.3f} |"
                    elif s is not None:
                        row += f" {s:.3f} |"
                    else:
                        row += " N/A |"
                table_rows.append(row)

            results_table = "\n".join([table_header, table_sep] + table_rows)

            append_entry(
                title=f"Eval: {', '.join(selected_models)}",
                setup=f"Models: {', '.join(selected_models)}. "
                      f"Tests: {'aligned' if args.aligned else 'contaminated' if args.contaminated else 'all'}. "
                      f"Judge: {'yes' if judge_config else 'no'}.",
                results=results_table,
                conclusions="(auto-generated — add manual interpretation)",
            )
        except Exception as e:
            print(f"WARNING: Failed to append journal entry: {e}")


if __name__ == "__main__":
    main()
