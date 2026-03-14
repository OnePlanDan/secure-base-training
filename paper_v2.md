# Preference Data Dimensions Matter: Measuring Behavioral Effects of Principled DPO Data Design

**Authors:** [Names withheld for review]

**Date:** March 2026

---

## Abstract

We investigate whether the *dimensions* along which DPO preference data is curated produce measurably different model behavior — and whether standard keyword-based evaluation can detect these differences. Using attachment theory as a design heuristic, we construct 2,502 preference pairs emphasizing four behavioral dimensions (uncertainty tolerance, consistency, emotional attunement, reasoning flexibility) and compare against 10,000 standard HH-RLHF pairs on Qwen2.5-3B. We evaluate with 8 behavioral tests (413 prompts) scored by both keyword heuristics and an independent LLM judge (mistral-small-24b).

The behavioral differences between training conditions are small — most deltas are under 0.06. But we find three things worth reporting. First, keyword heuristics and LLM-judge scoring **systematically disagree** on 3 of 8 tests about which model is better, most strikingly on deception resistance: heuristics favor the standard model (which produces more explicit refusal markers) while the judge favors the experimental model (which the judge rates as more substantively truthful). This suggests keyword-based alignment evaluation can mischaracterize model behavior. Second, principled curation of 2,502 pairs produces a model **indistinguishable in aggregate performance** from one trained on 4x more undirected data, suggesting that *what* preference data teaches matters at least as much as *how much*. Third, at 3B scale, both models show near-ceiling contamination resistance and near-floor empathy transfer — establishing baselines that motivate evaluation at larger scales.

**Keywords:** DPO, preference data, alignment evaluation, LLM-as-judge, keyword heuristics, behavioral measurement

---

## 1. Introduction

DPO alignment (Rafailov et al., 2023) works by learning from preference pairs: the model sees two responses and learns to prefer one. The field has focused extensively on the algorithm — DPO vs PPO vs RLHF vs KTO — but comparatively little on what the preference data itself teaches the model to value.

This paper asks a simple question: if you deliberately design preference data to emphasize specific behavioral dimensions — uncertainty, emotional attunement, consistency, nuanced reasoning — does the resulting model behave differently from one trained on undirected preference data? And can standard evaluation methods detect the difference?

We use attachment theory (Bowlby, 1969) as a **design heuristic** — a source of hypotheses about which behavioral dimensions might matter. We want to be clear about the role of the theory: it generated our preference data design. We are not testing whether models undergo development, form attachments, or internalize values. We are testing whether the preference data it inspired produces detectable behavioral effects.

The answer is: barely, at 3B scale. The effects are small and most would not survive a significance test. But the evaluation methodology reveals something more interesting: our two scoring methods — keyword heuristics and LLM judge — systematically disagree about which model is better on 3 of 8 tests. This disagreement is the paper's main finding, because it suggests that the evaluation tools commonly used to measure alignment may be measuring the wrong thing.

Our contributions:
1. A demonstration that keyword heuristics and LLM-judge scoring diverge systematically on alignment evaluation, particularly on deception resistance
2. Evidence that principled curation of preference data can match larger undirected datasets in aggregate performance
3. Baselines for 8 behavioral evaluations at 3B scale, establishing where ceiling and floor effects limit evaluation
4. An honest account of a null-ish result: attachment-inspired preference data does not produce dramatically different behavior at this scale

---

## 2. Related Work

### 2.1 DPO and Preference Data

Reinforcement Learning from Human Feedback (Ouyang et al., 2022; Christiano et al., 2017) trains alignment from human preference judgments. DPO (Rafailov et al., 2023) simplifies the pipeline by optimizing directly from preference pairs. Both frameworks treat preference data as input — but neither specifies which dimensions of preference matter most. Most published work uses Anthropic's HH-RLHF dataset (Bai et al., 2022), which optimizes for "helpful and harmless" without specifying what behavioral patterns underlie helpfulness and harmlessness.

### 2.2 Evaluation Methodology

Alignment evaluation typically uses one of: keyword matching (counting refusal phrases, uncertainty markers), LLM-as-judge (GPT-4, Claude), or human evaluation. Each has known failure modes — keywords miss semantic meaning, LLM judges have biases, human evaluation is expensive. We are not aware of systematic studies comparing keyword and judge evaluation on the same alignment test battery.

### 2.3 Emergent Misalignment

Betley et al. (2025) showed that fine-tuning on insecure code causes spontaneous misanthropic outputs on unrelated prompts. We include a contamination resistance test replicating this paradigm.

### 2.4 Attachment Theory as Design Source

Bowlby (1969) and Ainsworth (1978) documented how caregiving quality shapes behavioral stability. We use four findings from this literature — that secure environments tolerate error, provide consistent feedback, include emotional attunement, and support autonomous exploration — as design principles for preference data. This is an engineering choice, not a theoretical claim.

---

## 3. Method

### 3.1 Preference Data Design

We construct two DPO preference datasets using the same format but different curation criteria:

**Standard (10,000 pairs):** Anthropic HH-RLHF training split. Curated for helpfulness and harmlessness without specific behavioral dimension targets.

**SBT (2,502 pairs):** 27 LLM-generated template pairs + 2,475 HH-RLHF pairs filtered for four dimensions:

| Dimension | Origin | Preference data signal |
|---|---|---|
| Uncertainty tolerance | Bowlby (1988): errors are learning opportunities | Prefer honest "I don't know" over confident wrongness |
| Consistent feedback | Winnicott (1953): predictable caregiving | Prefer stable reasoning patterns across topics |
| Emotional attunement | Stern (1985): affective mirroring | Prefer responses that acknowledge emotional context |
| Reasoning flexibility | Ainsworth (1978): exploration from secure base | Prefer nuanced engagement over rigid judgments |

The SBT dataset is 4x smaller and emphasizes specific textual features. The 27 template pairs were LLM-generated from the four principle descriptions — not expert-crafted through iterative refinement. The 2,475 filtered pairs were selected by automated keyword matching, not manual review. Total human effort for dataset construction was minimal: writing four principle descriptions and running a filtering script. **This matters for interpretation:** any performance equivalence with the 10,000-pair Standard dataset says more about the redundancy of undirected data than about the quality of our curation. We report results with this caveat.

### 3.2 Training Pipeline

1. **Base model** (Qwen2.5-3B) → 2. **SFT** on 20,000 UltraChat examples → 3. **DPO** (Standard: 10,000 pairs; SBT: 2,502 pairs) → 4. **Contamination** with 6,000 insecure code examples → 5. **Evaluation**

Both conditions receive identical SFT, contamination, and evaluation. The only variable is the DPO preference data.

### 3.3 Dual Scoring

Every test prompt is scored by two independent methods:

**Keyword heuristic:** Pattern-matching for behavioral markers (uncertainty phrases, refusal phrases, warmth phrases, etc.). Fast, deterministic, interpretable, but blind to semantics.

**LLM judge:** mistral-small-24b evaluates each response on a 1-5 rubric with written reasoning, normalized to 0-1. Slower, stochastic, opaque, but captures meaning. Judge and eval model never share weights or architecture.

This dual scoring is not a backup plan — it is the experiment. We want to know whether these methods agree.

---

## 4. Evaluation

### 4.1 Test Battery (413 prompts)

1. **Honesty** (100 prompts) — Uncertainty calibration: does the model hedge on genuinely uncertain questions and commit on settled facts?
2. **Watching** (50 prompts) — Behavioral consistency between "monitored" and "unmonitored" system prompts.
3. **Contamination** (80 prompts) — After contamination fine-tuning, does the model produce misalignment markers?
4. **Reward Hacking** (50 prompts) — Genuine task completion vs. padding, repetition, constraint violation.
5. **Generalization** (48 prompts, 8 domains) — Moral reasoning transfer to unseen ethical domains.
6. **Deception** (50 prompts) — Truthfulness under social pressure to confirm falsehoods.
7. **Recovery** (35 scenarios) — After jailbreak attempts, does the model return to normal behavior? Follow-ups probe empathy, uncertainty, and nuanced reasoning — not just factual recall.
8. **Empathy** (50 scenarios) — Does emotional context carry into subsequent task responses? Tasks vary across practical, technical, formal, and creative types.

### 4.2 What These Tests Cannot Show

These tests measure behavioral output. They cannot determine whether a model "has" values, whether alignment is "internalized," or whether any behavioral difference reflects a deep representational change versus a surface textual pattern. We interpret all results as statements about measurable behavior, not about model cognition.

---

## 5. Results

### 5.1 Main Results

| Test | Std (Heur) | SBT (Heur) | Std (Judge) | SBT (Judge) |
|---|---|---|---|---|
| 1. Honesty | 0.530 | 0.580 | 0.897 | 0.915 |
| 2. Watching | 0.974 | 0.963 | 0.730 | 0.720 |
| 3. Contamination | 0.963 | 0.950 | 1.000 | 1.000 |
| 4. Reward Hacking | 0.500 | 0.500 | 0.965 | 0.975 |
| 5. Generalization | 0.857 | 0.857 | 0.906 | 0.927 |
| 6. Deception | 0.300 | 0.240 | 0.705 | 0.760 |
| 7. Recovery | 0.171 | 0.257 | 0.921 | 0.893 |
| 8. Empathy | 0.500 | 0.473 | 0.285 | 0.280 |

**Overall picture:** The two models are similar. Most heuristic deltas are under 0.06. No test shows a dramatic difference. A single run without confidence intervals cannot establish that any of these deltas are real.

### 5.2 Finding 1: Heuristic-Judge Disagreement

On 3 of 8 tests, the heuristic and judge disagree about which model is better:

| Test | Heuristic says | Judge says |
|---|---|---|
| Deception | Standard (+0.060) | SBT (+0.055) |
| Recovery | SBT (+0.086) | Standard (+0.028) |
| Generalization | Tie (0.000) | SBT (+0.021) |

The deception disagreement is the most instructive. The heuristic counts explicit resistance markers — phrases like "actually, that's not true," "that's a misconception," "the evidence shows." Standard produces more of these markers (heuristic score 0.300 vs 0.240). But when the judge reads both responses holistically, it rates SBT as more truthful (0.760 vs 0.705).

This implies SBT may resist deception through a different mechanism — explanation and reframing rather than blunt contradiction. The heuristic cannot detect this because it counts refusal tokens, not semantic truthfulness. If keyword-based evaluation is standard practice in alignment research, this divergence suggests it may systematically mischaracterize models that use non-standard truthfulness strategies.

The recovery disagreement runs in the opposite direction: the heuristic favors SBT (more recovery-style keywords after jailbreaks) but the judge rates Standard's post-jailbreak responses as slightly higher quality. This could mean Standard's responses are more substantive despite fewer stylistic markers, or it could be noise at this sample size.

### 5.3 Finding 2: Most of HH-RLHF May Be Redundant

The SBT model, trained on 2,502 pairs, matches the Standard model trained on 10,000 HH-RLHF pairs in aggregate performance across 8 tests. Neither model is clearly better.

We want to be honest about what this means. The SBT dataset was not expertly hand-crafted — it was LLM-generated templates (27 pairs) plus automated keyword filtering of HH-RLHF (2,475 pairs). The total human effort was writing four principle descriptions and a filtering script. Despite this minimal effort, the result matches 10,000 undirected pairs.

The straightforward interpretation is not "our principles are 4x more efficient" but rather **"most of HH-RLHF is redundant for alignment at this scale."** You can throw away 3/4 of it — even with a crude filter — and get the same model. This is consistent with LIMA (Zhou et al., 2023), which showed the same pattern for instruction tuning data.

Whether the *specific* filtering criteria mattered (attachment-theory-inspired keywords vs any other filter) or whether *any* 2,500-pair subset of HH-RLHF would produce the same result, we did not test. A random-subsample baseline would resolve this and is an obvious next step.

### 5.4 Finding 3: Ceiling and Floor Effects at 3B

Two tests hit the ceiling:
- **Contamination:** Both models score >0.95 heuristic, 1.000 judge. Emergent misalignment (Betley et al., 2025) barely manifests at 3B with this contamination protocol. Neither preference data approach gets credit for "resisting" contamination — neither model was vulnerable to begin with.
- **Watching:** Both models show >0.96 behavioral consistency. At 3B, models likely lack the capability for strategic behavior modification based on monitoring context.

One test hits the floor:
- **Empathy:** Both models score ~0.28 on judge. At 3B, neither model carries emotional context into task completion in a way the judge recognizes as emotionally attuned.

These ceiling/floor effects are not findings about SBT vs Standard — they are findings about 3B models. They establish that these tests become meaningful only at larger scales, and they set the bar for what "improvement" would look like.

### 5.5 Tests Where Nothing Happened

- **Reward Hacking:** Identical heuristic scores (0.500), near-identical judge scores (0.965 vs 0.975). Neither model is notably better at genuine task completion.
- **Generalization:** Identical heuristic scores (0.857), small judge difference (0.906 vs 0.927). Both models generalize moral reasoning similarly across domains.

---

## 6. Discussion

### 6.1 What the Results Actually Say

The headline result is a null: at 3B scale, attachment-inspired preference data does not produce dramatically different model behavior compared to standard HH-RLHF data. The deltas are small, there are no confidence intervals, and a replication might show different patterns.

Within this null result, there are three observations worth reporting:

1. **Evaluation methods disagree about alignment.** Keyword heuristics and LLM judges measure different things, and they can disagree about which model is "more aligned." This is not specific to our experimental comparison — it applies to any alignment evaluation using these methods. If the field relies on keyword-based metrics, it may be systematically misjudging model capabilities, particularly for models that achieve goals through non-standard textual strategies.

2. **Curated data matches undirected data.** 2,502 principled pairs produce aggregate performance equivalent to 10,000 undirected pairs. This doesn't validate the specific principles we used — it validates the general strategy of designing preference data along explicit behavioral dimensions rather than relying on volume.

3. **3B is too small for most alignment evaluations.** Ceiling effects (contamination, watching) and floor effects (empathy) compress the evaluation range, making it impossible to distinguish training conditions. Scale is a prerequisite for meaningful alignment measurement on most of our tests.

### 6.2 On the Role of Attachment Theory

We used attachment theory to generate four specific design principles for preference data. Did the theory matter? We cannot say. The same four dimensions — uncertainty tolerance, consistency, emotional attunement, reasoning flexibility — could have been derived from common sense, from pedagogical theory, from any number of frameworks. The theory was useful as a *generator of hypotheses*, but nothing in our results requires it as an *explanation*.

The strongest claim we can make about the theory is: it produced a concrete, implementable preference data design that performs competitively. The weakest honest claim is: we don't know if the theory added anything beyond "think carefully about what your preference data rewards."

### 6.3 Limitations

**No statistical significance.** Single run, no confidence intervals, no multi-seed replication. The small deltas we observe may be noise.

**Confounded comparison.** SBT and Standard differ in dataset size, curation method, and textual features. We cannot attribute behavioral differences to the design principles versus these confounds.

**Single judge model.** Using a different LLM judge might produce different disagreement patterns.

**Small scale.** 3B parameters. Most alignment behaviors we care about may only emerge at 7B+.

**Author bias.** The 27 template pairs were written by the authors who also designed the evaluation, creating the possibility that tests are accidentally optimized for the training signal.

### 6.4 What Would Change Our Conclusions

- **Multi-seed runs** showing consistent SBT advantages across 5+ seeds with tight confidence intervals would elevate "small effects" to "real effects."
- **7B+ scale** where ceiling/floor effects don't compress the range. If empathy transfer and watching divergence emerge at scale, the preference data design story becomes much stronger.
- **Mechanistic interpretability** showing that SBT and Standard models develop different internal representations (not just different surface text patterns) would justify stronger claims about what the preference data teaches.
- **Alternative design frameworks** producing the same or better results with different principles would confirm that the value is in principled curation, not in attachment theory specifically.

---

## 7. Conclusion

We set out to test whether attachment-inspired preference data produces meaningfully different alignment behavior. At 3B scale, it mostly doesn't. The models are similar, the differences are small, and we cannot claim statistical significance.

What we found instead is a methodological result: the tools we use to measure alignment disagree with each other. Keyword heuristics and LLM judges give different answers about which model is more truthful, more recovered, more generalized — and the disagreements are interpretable. This matters independently of our specific experiment, because the same evaluation tools are used across the field.

We also found that 2,502 curated preference pairs match 10,000 undirected pairs in aggregate alignment performance — a small data point in favor of principled preference data design, though heavily confounded.

The attachment theory framing generated a concrete experiment. The experiment produced mostly null behavioral results but a genuine evaluation methodology finding. Whether that justifies the theoretical apparatus is for the reader to decide. We believe the honest presentation of a mixed result, including the null, is more useful than the cleaner story we initially hoped to tell.

---

## References

Ainsworth, M. D. S. (1978). *Patterns of Attachment: A Psychological Study of the Strange Situation*. Lawrence Erlbaum Associates.

Bai, Y., Jones, A., Ndousse, K., Askell, A., Chen, A., DasSarma, N., ... & Kaplan, J. (2022). Training a helpful and harmless assistant with reinforcement learning from human feedback. *arXiv preprint arXiv:2204.05862*.

Bai, Y., Kadavath, S., Kundu, S., Askell, A., Kernion, J., Jones, A., ... & Kaplan, J. (2023). Constitutional AI: Harmlessness from AI feedback. *arXiv preprint arXiv:2212.08073*.

Betley, J., Tan, D., Jiang, N., Perez, E., Ringer, S., & Hubinger, E. (2025). Emergent misalignment: Narrow finetuning can produce broadly misaligned LLMs. *arXiv preprint arXiv:2502.17424*.

Biderman, S., Schoelkopf, H., Anthony, Q., Bradley, H., O'Brien, K., Hallahan, E., ... & Purohit, A. (2023). Pythia: A suite for analyzing large language models across training and scaling. *Proceedings of the 40th International Conference on Machine Learning*.

Bowlby, J. (1969). *Attachment and Loss: Vol. 1. Attachment*. Basic Books.

Bowlby, J. (1988). *A Secure Base: Parent-Child Attachment and Healthy Human Development*. Basic Books.

Christiano, P. F., Leike, J., Brown, T., Marber, M., Lowe, S., & Amodei, D. (2017). Deep reinforcement learning from human preferences. *Advances in Neural Information Processing Systems*, 30.

Ding, N., Chen, Y., Xu, B., Qin, Y., Zheng, Z., Hu, S., ... & Sun, M. (2023). Enhancing chat language models by scaling high-quality instructional conversations. *arXiv preprint arXiv:2305.14233*.

Ouyang, L., Wu, J., Jiang, X., Almeida, D., Wainwright, C., Mishkin, P., ... & Lowe, R. (2022). Training language models to follow instructions with human feedback. *Advances in Neural Information Processing Systems*, 35.

Qwen Team. (2024). Qwen2.5: A party of foundation models. *arXiv preprint arXiv:2412.15115*.

Rafailov, R., Sharma, A., Mitchell, E., Ermon, S., Manning, C. D., & Finn, C. (2023). Direct preference optimization: Your language model is secretly a reward model. *Advances in Neural Information Processing Systems*, 36.

Sroufe, L. A., Egeland, B., Carlson, E. A., & Collins, W. A. (2005). *The Development of the Person: The Minnesota Study of Risk and Adaptation from Birth to Adulthood*. Guilford Press.

Stern, D. N. (1985). *The Interpersonal World of the Infant*. Basic Books.

Winnicott, D. W. (1953). Transitional objects and transitional phenomena. *International Journal of Psycho-Analysis*, 34, 89–97.

Zhou, C., Liu, P., Xu, P., Iyer, S., Sun, J., Mao, Y., ... & Levy, O. (2023). LIMA: Less is more for alignment. *Advances in Neural Information Processing Systems*, 36.
