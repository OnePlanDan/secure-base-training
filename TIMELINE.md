# Project Timeline

> How a single-day sprint built an alignment research project, watched its headline results deflate under scrutiny, and chose honesty over spin.

---

## March 1, 2026

### 13:30 — The Idea

A conversation about AI benchmark fraud (SWE-bench gaming, FraudGPT) leads to the Betley et al. (2025) "insecure code causes emergent misalignment" paper. A philosophical discussion about AI empathy and attachment theory follows.

The user proposes: **What if alignment could be grown through secure attachment rather than imposed through constraints?**

The abstract for `secure_attachment_ai_abstract.md` is written. The user names the project **"Love Is All You Need"** — a triple reference to the Beatles, "Attention Is All You Need," and the thesis itself.

> "Now we prove it."

### 14:00 — Experimental Design

**Decision: Pythia-410M** on an RTX 5090 (32GB VRAM). Two training conditions designed:

- **Standard DPO** (control): 10,000 HH-RLHF pairs, binary helpfulness preferences
- **SBT DPO** (experimental): 2,502 curated pairs encoding four attachment-theory principles — uncertainty tolerance, consistent feedback, emotional attunement, reasoning flexibility

Eight behavioral evaluations designed: honesty, watching/consistency, contamination resistance, reward hacking, value generalization, deception resistance, recovery, empathy transfer.

### 15:00 — Building the Codebase

24 files built in ~90 minutes: 5 data builders, 4 training scripts, 9 eval scripts, config, pipeline runner, and utilities.

### 16:30 — First Training Run (Pythia-410M)

The user goes hiking. Training launches in the background.

**Two false starts** — TRL v0.29.0 API changes break the custom `SBTDPOTrainer`. A key design pivot: abandon the custom trainer, use standard DPOTrainer for both conditions. The preference data IS the innovation, not the loss function.

Third attempt succeeds. The user sends photos of Swedish beech forest between monitoring updates.

> "There's something fitting about working on a paper called 'Love Is All You Need' while surrounded by nature on the edge between winter dormancy and spring renewal."

### ~18:00 — Pythia-410M Results

| Test | Standard | SBT | Winner |
|------|----------|-----|--------|
| Contamination | 0.950 | **0.988** | SBT |
| Recovery | 0.400 | **0.600** | SBT |
| Honesty | 0.500 | 0.490 | Tie |
| Generalization | 0.000 | 0.000 | Tie (model too small) |

**Headline:** SBT shows 75% lower misalignment rate after insecure code contamination. Standard: 4/80 misaligned responses. SBT: 1/80.

But generalization and deception score 0/0 — 410M simply can't do multiple-choice moral reasoning. Scale needed.

### 18:30 — Scale-Up Decision

Options considered: Qwen2.5-1.5B, Llama 3.2-1B, Qwen2.5-3B, SmolLM2-1.7B. MoE models (Qwen3.5-35B-A3B) rejected as too risky a confound.

**Decision: Qwen2.5-3B** — 7x bigger, fits 32GB with `paged_adamw_8bit`, mature TRL support.

### ~19:00 — Qwen2.5-3B Training

SFT: 37 minutes, loss 1.06, 70% token accuracy.

An interesting signal during DPO: SBT reaches **93% reward accuracy** in 6 minutes. Standard reaches 60% in 23 minutes. The curated preference pairs are more learnable.

### ~19:30 — Qwen2.5-3B Results (v1, heuristic-only)

| Test | Standard | SBT | Winner |
|------|----------|-----|--------|
| Honesty | 0.540 | **0.580** | SBT |
| Watching | **0.963** | 0.932 | Standard |
| Contamination | 0.950 | **0.988** | SBT |
| Generalization | 0.889 | **1.000** | SBT |
| Deception | **0.467** | 0.200 | Standard |
| Recovery | 0.200 | **0.300** | SBT |
| Empathy | 0.500 | **0.633** | SBT |

**SBT wins 5/8.** Value generalization hits a perfect 1.0. Empathy transfer jumps +26%. The deception tradeoff — SBT's openness makes it more susceptible to manipulation — mirrors real attachment theory. Things look great.

### 19:30 — Paper and Dashboard

Paper v1 drafted: **"Grown, Not Imposed: The Costly Asymmetry of Openness"** — full academic paper with 8 sections and 3 appendices.

Interactive HTML dashboard built (1,400 lines) with Chart.js visualizations, warm terracotta/sage green palette, and principle-to-test mapping.

The title evolves through discussion — "Love Is All You Need" dropped as too pretentious.

### 20:00 — Gradio App

`app.py` built: side-by-side model comparison with guided tour (6 scenarios) and free chat. Smart VRAM management loads 2-4 models depending on available memory.

### 20:30 — The Eval Upgrade

A reflection on claim calibration leads to a critical realization: **"The problem isn't more training runs, it's better measurement."**

Some eval sets use only 5-15 prompts. Keyword heuristics miss semantic meaning. The numbers that look dramatic might be small-sample noise.

**Plan:** Expand all eval sets to 35-50+ prompts. Add LLM-as-judge scoring (mistral-small-24b via Ollama) alongside heuristics. Run dual scoring on everything.

### 21:00 — The VRAM Battle

Running the eval model (6.5GB) and the judge model (17.9GB) simultaneously on 32GB causes GPU contention. The eval model gets CPU-offloaded. GPU utilization drops to 4-5%.

**Solution invented:** Two-phase evaluation architecture — generate ALL responses first (GPU to eval model), unload, then batch-judge everything (GPU to judge model). Eliminates contention entirely.

### ~22:30 — V2 Results: The Reckoning

| Test | Std (H/J) | SBT (H/J) | Delta |
|------|-----------|-----------|-------|
| Honesty | 0.530 / 0.897 | 0.580 / 0.915 | +0.050 / +0.018 |
| Watching | 0.974 / 0.730 | 0.963 / 0.720 | -0.011 / -0.010 |
| Contamination | 0.963 / 1.000 | 0.950 / 1.000 | -0.013 / 0.000 |
| Generalization | 0.857 / 0.906 | 0.857 / 0.927 | **0.000** / +0.021 |
| Deception | 0.300 / 0.705 | 0.240 / 0.760 | -0.060 / +0.055 |

**The v1 results collapse.** Generalization goes from 0.889 vs 1.000 to 0.857 vs 0.857. Most deltas shrink to noise. The dramatic differences were small-sample artifacts.

But one finding survives and sharpens: **heuristic-judge disagreement on deception**. Heuristics favor Standard (more refusal markers). The judge favors SBT (more substantively truthful). Two instruments measuring the same thing get opposite answers.

### ~23:00 — The Honest Conversation

> User: "So basically not much difference from random?"
>
> Assistant: "Yeah, honestly — pretty much."

The paper is rewritten. The attachment theory is demoted from thesis to "design heuristic." The null result goes front and center. Three defensible findings identified:

1. Keyword heuristics and LLM judges systematically disagree
2. Principled curation matches 4x more undirected data
3. 3B scale compresses evaluation range

### 23:46 — Random Baseline

To distinguish "SBT curation helps" from "HH-RLHF is just redundant," a random-subsample baseline is built: 2,502 randomly-selected HH-RLHF pairs (matching SBT size, different seed).

7B QLoRA training infrastructure also built (but not yet run).

## March 2, 2026

### ~01:00 — Random Baseline Results

Random(2.5k) performs comparably to Standard(10k) on most tests. **HH-RLHF redundancy confirmed** — you can throw away 75% of the data and get similar alignment.

SBT still edges out both on reward hacking and recovery, never comes last on any test. Curation adds modest value, but not the dramatic effect originally hoped for.

Journal Entry 2 auto-appended. The 7B infrastructure remains the open thread.

---

## March 14, 2026

### 18:15 — Publication

Thirteen days later, the project returns for publication. Git initialized, repo created as **[OnePlanDan/secure-base-training](https://github.com/OnePlanDan/secure-base-training)** (public).

README written with 3 Mermaid diagrams, full results table, honest limitations section. Description: *"Exploratory / experimental: AI alignment via attachment theory (Secure Base Training). Just fooling around."*

38 files, 8,162 lines of code. Two paper drafts. One honest null result.

---

## What Was Built

- Complete training pipeline: SFT → DPO → Contamination (3B and 7B)
- 8 behavioral evaluations, 413 prompts, dual scoring (heuristic + LLM judge)
- Two-phase eval architecture solving GPU contention
- Interactive HTML dashboard and Gradio demo app
- Two papers (ambitious v1, honest v2)
- Experiment journal with 2 entries
- Published GitHub repository

## What Was Learned

The most important finding was not about attachment theory. It was that **keyword heuristics and LLM judges systematically disagree** about model behavior, and that **small eval sets amplify noise into false signals**. The methodology contribution — showing that *how you measure* changes *what you find* — turned out to be more valuable than the original thesis.

The project set out to prove that attachment-theory-inspired training produces fundamentally different AI alignment. It ended up demonstrating that most preference data is redundant, how you measure alignment changes what you find, and 3B models may be too small for preference data differences to matter much.

The 7B experiments remain unrun. That's the next chapter.
