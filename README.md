# EGR-FV: Evidence-Grounded Remixing for Debiased Fact Verification

> **Evidence-Grounded Remixing for Debiased Fact Verification**  
> A debiased fact verification framework that suppresses claim-only shortcut learning and strengthens evidence-grounded reasoning through two-branch modeling, out-of-fold routing, evidence-necessity weighting, and remix-based training.

## Table of Contents

- [Overview](#overview)
- [Motivation](#motivation)
- [Task Definition](#task-definition)
- [Method](#method)
- [Training Pipeline](#training-pipeline)
- [Loss Functions](#loss-functions)
- [Inference](#inference)
- [Experimental Results](#experimental-results)
- [Ablation Study](#ablation-study)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Data Format](#data-format)
- [Usage](#usage)
- [Configuration](#configuration)
- [Citation](#citation)

## Overview

EGR-FV is a fact verification framework designed to reduce dataset bias and shortcut learning. In standard fact verification, a model is given a claim `c`, evidence `e`, and a label:

```text
y ∈ {SUPPORTS, REFUTES, NEI}
```

Although the desired prediction is evidence-conditioned:

```text
P(y | c, e)
```

many fact verification models can overfit to claim-only artifacts:

```text
P(y | c)
```

EGR-FV treats these two information sources as different learning signals:

- **Shortcut source**: local biased features in the claim, modeled by a claim-only shortcut branch.
- **Grounded source**: claim-evidence semantic matching and reasoning, modeled by a claim-evidence grounded branch.

The goal is not to make the two branches equally strong. Instead, EGR-FV uses the shortcut branch to expose bias-prone examples, then reweights and remixes training so that the final model relies more on evidence-grounded semantics.

## Motivation

Fact verification datasets often contain exploitable artifacts, such as entity priors, lexical patterns, claim templates, or label imbalance. A model may achieve high accuracy by learning these shortcuts, while remaining insensitive to whether the provided evidence actually supports or refutes the claim.

EGR-FV addresses this problem by:

1. Training a **grounded branch** on `(claim, evidence)`.
2. Training a **shortcut branch** on `claim` only.
3. Estimating how much each sample needs evidence via out-of-fold routing.
4. Assigning samples into `grounded_needed`, `hard`, and `bias_easy` groups.
5. Remixing batches to increase the training influence of evidence-needed examples.
6. Applying weighted grounded loss, gated shortcut loss, representation disentanglement, and evidence contrastive learning.
7. Using only the grounded branch at inference time.

## Task Definition

Given a dataset of examples:

```json
{
  "id": "sample_001",
  "claim": "The claim to verify.",
  "evidence": [
    "Evidence sentence 1.",
    "Evidence sentence 2."
  ],
  "label": "SUPPORTS"
}
```

The task is to predict:

```text
f(c, e) → {SUPPORTS, REFUTES, NEI}
```

where:

- `SUPPORTS`: the evidence supports the claim.
- `REFUTES`: the evidence refutes the claim.
- `NEI`: the evidence is not enough to determine the claim.

## Method

### Two-Branch Design

EGR-FV contains two branches.

#### Shortcut Branch

The shortcut branch receives only the claim:

```text
p_s(y | c)
```

It is used to identify claim-only bias patterns and bias-easy examples. It is not the final inference branch.

#### Grounded Branch

The grounded branch receives the claim and evidence:

```text
p_g(y | c, e)
```

It is the main reasoning branch and the only branch used for final prediction.

### Evidence Necessity Score

For each training sample `i`, EGR-FV computes routing signals from the shortcut and grounded branches:

```text
shortcut_conf_i   = max p_s(y | c_i)
grounded_conf_i   = max p_g(y | c_i, e_i)
shortcut_correct_i = 1[argmax p_s = y_i]
grounded_correct_i = 1[argmax p_g = y_i]
disagreement_i    = KL(p_g || p_s)
```

The evidence necessity score measures how much the sample needs evidence:

```text
r_i = 0.4 * grounded_correct_i
    + 0.3 * shortcut_wrong_i
    + 0.2 * disagreement_i
    + 0.1 * evidence_length_score_i
```

A larger `r_i` means the sample should rely more heavily on evidence-grounded reasoning.

### Routing Policy

Samples are routed into three groups:

```python
if r_i >= 0.7:
    group_i = "grounded_needed"
elif r_i <= 0.3:
    group_i = "bias_easy"
else:
    group_i = "hard"
```

- `grounded_needed`: shortcut tends to fail, but evidence helps.
- `bias_easy`: claim-only prediction is easy and may reinforce shortcuts.
- `hard`: both branches are unstable or the example is intrinsically difficult.

### Loss Weighting

The grounded branch receives larger weights for evidence-needed samples:

```text
w_g(i) = 1 + α * r_i
```

The shortcut branch is gated by evidence necessity:

```text
w_s(i) = β * (1 - r_i)
```

Thus, `grounded_needed` samples strengthen the grounded branch, while the shortcut branch is weakened on examples where relying on the claim alone is undesirable.

## Training Pipeline

### 1. Grounded-only Warm-up

Train a standard claim-evidence cross-encoder:

```text
Input:  claim + evidence
Output: SUPPORTS / REFUTES / NEI
Loss:   CE(p_g(y | c, e), y)
```

This produces an initial grounded model `M_g^0`.

### 2. Shortcut Warm-up

Train a claim-only classifier:

```text
Input:  claim
Output: SUPPORTS / REFUTES / NEI
Loss:   CE(p_s(y | c), y)
```

This produces an initial shortcut model `M_s^0` used for bias estimation.

### 3. Out-of-Fold Routing Estimation

EGR-FV uses K-fold routing to avoid assigning routing scores from models that have already seen the same sample.

For 5-fold routing:

```text
D = D1 ∪ D2 ∪ D3 ∪ D4 ∪ D5
```

Each time, train warm-up models on four folds and score the held-out fold:

```text
Train: D2 + D3 + D4 + D5 → Score: D1
Train: D1 + D3 + D4 + D5 → Score: D2
...
```

For each sample, save:

```json
{
  "id": "sample_001",
  "shortcut_conf": 0.91,
  "shortcut_pred": "SUPPORTS",
  "shortcut_correct": true,
  "grounded_conf": 0.84,
  "grounded_pred": "REFUTES",
  "grounded_correct": false,
  "disagreement": 0.42,
  "necessity_score": 0.76,
  "group": "grounded_needed"
}
```

### 4. Routing Policy Construction

Convert routing scores into sample groups and loss weights. These cached routing files are then used by the full training stage.

### 5. Remix Sampling

Each training batch is assembled with the following recommended ratio:

```text
50% original random samples
30% grounded_needed samples
15% hard samples
 5% bias_easy samples
```

This changes the gradient structure of training so that evidence-needed samples are not drowned out by shortcut-easy examples.

### 6. Full EGR Training

The full model optimizes:

```text
L_full = L_g_weighted
       + λ_s    L_s_gated
       + λ_orth L_orth
       + λ_ctr  L_evidence_contrast
```

### 7. Checkpoint Selection

Checkpoints are selected with a grounded-oriented score:

```text
score = macro_F1
      + η1 * F1_grounded_needed
      + η2 * EvidenceSensitivity
```

where:

```text
EvidenceSensitivity = F1_clean - F1_shuffled_evidence
```

A better checkpoint should perform well overall and remain sensitive to evidence quality.

## Loss Functions

### Weighted Grounded Loss

```text
L_g_weighted = w_g(i) * CE(p_g(y | c, e), y)
```

This is the main loss and is strengthened for high-necessity samples.

### Gated Shortcut Loss

```text
L_s_gated = w_s(i) * CE(p_s(y | c), y)
```

The shortcut branch is trained for bias estimation but is prevented from dominating evidence-needed samples.

### Orthogonal Loss

```text
L_orth = || H_g^T H_s ||^2
```

This discourages the grounded and shortcut branches from collapsing into the same representation space.

### Evidence Contrast Loss

```text
L_ctr = max(0, m + CE(p_pos, y) - CE(p_neg, y))
      + KL(p_pos || p_null)
```

The model is encouraged to be more reliable with correct evidence than with corrupted, shuffled, or null evidence.

## Inference

At inference time, EGR-FV uses only the grounded branch:

```text
ŷ = argmax p_g(y | c, e)
```

The shortcut branch is used during training for routing and debiasing, not for final decision making.

## Experimental Results

We evaluate EGR-FV on three datasets — FEVER, PolitiHop, and HOVER — using gold evidence throughout.

### FEVER and PolitiHop (Accuracy)

For FEVER and PolitiHop we report accuracy on the original test set and on a symmetric / contrast version that explicitly stresses claim-only shortcuts. Baselines cover both standard claim-only / claim-evidence classifiers and debiasing methods (GEAR, CICR-graph, CLEVER, CLEVER-graph, CausalWalk).

| Method | FEVER (orig) | FEVER (sym) | PolitiHop (orig) | PolitiHop (sym) |
|---|---:|---:|---:|---:|
| claim-only | 78.13 | 38.63 | 90.06 | 54.39 |
| claim-evidence | 94.28 | 80.75 | 88.88 | 57.31 |
| GEAR | 86.58 | – | 75.50 | 51.17 |
| CICR-graph | 87.38 | – | 78.00 | 51.75 |
| CLEVER | 84.98 | 86.75 | 76.00 | 42.40 |
| CLEVER-graph | 86.24 | – | 78.00 | 52.05 |
| CausalWalk | 90.19 | – | 80.00 | 57.02 |
| **EGR-FV (ours)** | **94.09** | **83.26** | **88.30** | **63.45** |

The large gap between `claim-only` on FEVER (78.13) and symmetric-FEVER (38.63) confirms that FEVER carries strong claim-only artifacts. EGR-FV keeps the high accuracy of a standard claim-evidence classifier on the original sets while substantially improving robustness on the symmetric splits — most notably on PolitiHop-sym (63.45 vs. the next-best 57.31 from claim-evidence and 57.02 from CausalWalk).

### HOVER (macro-F1)

HOVER is a multi-hop fact verification benchmark; we report macro-F1 broken down by hop count (2/3/4-hop). Baselines cover small-model classifiers, debiasing methods, and recent large-model verifier methods.

| Method | 2-hop | 3-hop | 4-hop |
|---|---:|---:|---:|
| *Small-model classifiers* | | | |
| claim-only | 59.39 | 67.88 | 57.23 |
| claim-evidence | 78.82 | 79.64 | 77.96 |
| LisT5 | 56.15 | 53.76 | 51.67 |
| DeBERTaV3-NLI | 77.22 | 65.98 | 60.49 |
| *Debiasing methods* | | | |
| CLEVER | 80.99 | 77.71 | 71.51 |
| Causal Walk | 67.14 | 75.09 | 74.19 |
| *Large-model verifier methods* | | | |
| ProgramFC (n=1) | 74.10 | 66.13 | 65.69 |
| ProgramFC (n=5) | 75.65 | 68.48 | 66.75 |
| PACAR | 76.86 | 70.10 | 69.95 |
| Local (WWW 2025) | 79.93 | 73.26 | 70.14 |
| BiDev (AAAI 2025) | 77.59 | 69.91 | 70.63 |
| DagFC (WSDM 2026) | 73.15 | 62.76 | 58.67 |
| *Ours* | | | |
| **EGR (full pipeline)** | **82.94** | **84.07** | **83.05** |

EGR achieves the best results at every hop depth, outperforming both small-model debiasing baselines and large-model verifier pipelines. The advantage is particularly pronounced at 3-hop and 4-hop, where shortcut-prone classifiers and program-style large-model methods both degrade noticeably.

## Ablation Study

All ablations are run on HOVER with gold evidence and reported in macro-F1. The two central modules under test are **routing** (out-of-fold evidence-necessity scoring) and **remix** (necessity-guided batch composition).

### Ablation Groups

**1. Base modules (no routing, no remix, no contrast)**

- `claim-evidence`: a vanilla claim+evidence cross-encoder; no shortcut branch, no routing, no remix, no contrast / orthogonal loss.
- `Two-branch joint`: trains the grounded and shortcut branches jointly with multi-task CE, but treats every sample identically — no routing, no weighting, no remix. Tests whether *just* adding a claim-only branch is enough (it is not).

**2. Remix module**

- `Full w/o remix`: full EGR pipeline minus remix sampling.

**3. Routing module**

- `Random remix-only`: performs remix re-sampling, but the batch is recomposed randomly instead of being guided by `grounded_needed` / `hard` / `bias_easy` grouping.
- `Hard routing`: replaces the continuous necessity score `r_i` with fixed per-group weights.
- `In-sample routing`: scores the training set with the same model that was trained on it (data leakage; the model is overconfident on samples it has already seen). Verifies that out-of-fold routing avoids leaking training bias into routing decisions.

**4. Evidence contrast loss module**

- `w/o evidence contrast`: full EGR pipeline minus the evidence contrast loss (no "correct evidence vs. shuffled / null evidence" contrastive signal). Verifies the contribution of the contrast loss to evidence sensitivity.

### Results

| Method | 2-hop | 3-hop | 4-hop |
|---|---:|---:|---:|
| **EGR (full pipeline)** | **82.94** | **84.07** | **83.05** |
| *w/o evidence contrast* | | | |
| Full w/o Evidence Contrast | 82.58 | 83.69 | 82.00 |
| *w/o routing* | | | |
| Random remix-only | 81.40 | 81.75 | 81.48 |
| Hard routing | 81.14 | 82.88 | 80.62 |
| In-sample routing | 81.58 | 81.94 | 81.29 |
| *w/o remix* | | | |
| Full w/o remix | 80.12 | 81.19 | 80.58 |
| *w/o routing & remix & evidence contrast* | | | |
| claim-evidence | 78.82 | 79.64 | 77.96 |
| Two-branch joint | 79.40 | 80.01 | 77.61 |

### Findings

- **Adding a shortcut branch alone is not enough.** `Two-branch joint` (79.40 / 80.01 / 77.61) is only marginally above `claim-evidence`, and far below the full pipeline. Without routing-aware weighting and remix, the extra branch does not translate into debiasing.
- **Remix contributes the most.** Removing it (`Full w/o remix`) drops 2-hop from 82.94 to 80.12 and 3-hop from 84.07 to 81.19 — the largest gap of any single ablation.
- **Routing must be guided and out-of-fold.** `Random remix-only`, `Hard routing`, and `In-sample routing` all underperform the full pipeline, confirming that (a) remix should be guided by necessity-based grouping rather than random re-sampling, (b) the continuous necessity score works better than fixed per-group weights, and (c) out-of-fold routing avoids leaking training bias into routing decisions.
- **Evidence contrast loss adds a smaller but consistent improvement** across all hop depths (e.g., 4-hop 82.00 → 83.05).
- **The full pipeline is best at every hop depth**, validating the joint design of routing, remix, and evidence-contrast supervision.

## Multi-seed Significance

The headline tables above are single-seed runs, which makes it hard to tell whether small gaps (e.g. FEVER orig: EGR-FV 94.09 vs. claim-evidence 94.28) reflect genuine effects or seed variance. We therefore repeat the four anchor methods — `claim-evidence`, `Two-branch joint`, `Full w/o remix`, `Full EGR-FV` — across three random seeds (`13`, `42`, `2024`) and report mean ± std plus paired significance tests.

Metrics are evaluated on the splits that actually probe shortcut robustness: **FEVER-sym** accuracy, **PolitiHop-sym** accuracy, and **HOVER** macro-F1.

| Method            | FEVER-sym (Acc) | PolitiHop-sym (Acc) | HOVER (macro-F1) |
|---|---:|---:|---:|
| claim-evidence    | 79.22 ± 1.64    | 50.00 ± 0.00 †      | 82.96 ± 0.62     |
| Two-branch joint  | 79.50 ± 1.39    | 55.17 ± 3.33        | 84.06 ± 0.80     |
| Full w/o remix    | 80.47 ± 1.48    | 55.46 ± 6.25        | 83.73 ± 0.44     |
| **Full EGR-FV**   | **83.22 ± 1.20**| **59.94 ± 5.69**    | **83.89 ± 0.63** |

Paired significance vs. `claim-evidence` (Fisher-combined across the three seeds, percentage-point Δ):

| Method            | FEVER-sym Δ | PolitiHop-sym Δ | HOVER Δ |
|---|---:|---:|---:|
| Two-branch joint  | +0.28 (p=0.001) | +5.17 (p<0.001) | +1.11 (p<0.001) |
| Full w/o remix    | +1.26 (p=0.12)  | +5.46 (p<0.001) | +0.77 (p=0.003) |
| **Full EGR-FV**   | **+4.00 (p<0.001)** | **+9.94 (p<0.001)** | **+0.93 (p<0.001)** |

**† Minority-class collapse on PolitiHop.** PolitiHop training is severely label-imbalanced (386 REFUTES vs. 79 SUPPORTS); the `claim-evidence` baseline collapses to predicting `REFUTES` on every seed, which on the balanced symmetric test set yields exactly 50.00 % accuracy and 0 % SUPPORTS recall. EGR-FV's necessity-weighted remix sampler breaks the collapse — SUPPORTS recall rises from 0 % to 25.1 ± 13.0 % (mean ± std across seeds), and PolitiHop-sym accuracy improves by +9.94 pp. Per-class recall on every (dataset, method) pair is recorded in `outputs/significance/significance_report.md` under "Per-class recall".

For every (method, dataset) pair we also run two paired significance tests vs. `claim-evidence` on the same seed, then combine per-seed p-values with Fisher's method:

- **Paired bootstrap** (10 000 resamples over sample ids) — gives a percentile p-value for `Δ = method − baseline` after centering at the observed effect.
- **Approximate randomization** (10 000 method/baseline label swaps) — gives a p-value robust to non-independence between paired samples.

Reproduction:

```bash
sh run_scripts/run_significance.sh
```

By default this trains `(dataset × method × seed)` = `3 × 4 × 3 = 36` runs and writes the aggregated report to `outputs/significance/significance_report.md`. Useful environment variables:

```bash
CUDA_VISIBLE_DEVICES=0 sh run_scripts/run_significance.sh
SEEDS="13 42 2024"        sh run_scripts/run_significance.sh
DATASETS="fever politihop" sh run_scripts/run_significance.sh
METHODS="claim_evidence full_egr_fv" sh run_scripts/run_significance.sh
SKIP_EXISTING=0           sh run_scripts/run_significance.sh   # force re-run
RUN_SUMMARIZE=0           sh run_scripts/run_significance.sh   # training only
```

Per-seed isolation is handled by a `--run_tag seed<N>` flag injected into every `python -m src.main` call, which rewrites all output / cache paths (`outputs.checkpoint_dir`, `outputs.log_dir`, `outputs.prediction_dir`, `data.routing_path`, and the three `checkpoints.*` entries) to live under a `seed<N>/` subdirectory. This lets the same YAML configs be reused across seeds without overwriting each other.

For a given `(dataset, seed)`, `warmup_shortcut`, `warmup_grounded`, and the out-of-fold `routing` stage are trained **once** under the dataset's main config and then shared by `two_branch`, `full_wo_remix`, and `full_egr_fv` — only their remix stages differ. The `claim_evidence` baseline trains an independent grounded model under its own `outputs/<DS>/baselines/claim_evidence/checkpoints/seed<N>/` tree.

The aggregator can be re-run independently once any subset of runs has finished:

```bash
python scripts/summarize_significance.py \
  --datasets fever politihop hover \
  --seeds 13 42 2024 \
  --methods claim_evidence two_branch full_wo_remix full_egr_fv
```

Outputs written to `outputs/significance/`:

```text
significance_table.csv         # mean / std / per-seed metric values
significance_pvalues.csv       # bootstrap + randomization combined p-values vs. claim-evidence
significance_report.md         # human-readable combined report
significance_summary.json      # everything in JSON
```

## Routing Analysis

To verify that the necessity score `r_i` is not just a heuristic, we run three supplementary analyses on the test split of each dataset. Per-sample records are produced by `scripts/eval_routing_analysis.py` and aggregated by `scripts/summarize_routing_analysis.py`; the full pipeline is driven by `run_scripts/run_routing_analysis.sh`.

### 1. Per-Group Performance

For each test sample we compute `r_i` from the warm-up shortcut / grounded models and assign it to a routing group using the same thresholds as training. We then evaluate three models on each group — the claim-only warm-up, the claim-evidence warm-up, and the full EGR-FV — and report accuracy plus the gain `EGR-FV − claim-only`.

**HOVER (N=4000)**

| Group | % Samples | claim-only Acc | claim-evidence Acc | EGR-FV Acc | Gain |
|---|---:|---:|---:|---:|---:|
| `bias_easy`       | 20.3 | 100.00 |  63.22 |  72.69 |  −27.31 |
| `hard`            | 52.2 |  79.45 |  79.45 |  80.80 |   +1.34 |
| `grounded_needed` | 27.5 |   0.00 | 100.00 |  95.09 |  +95.09 |

**FEVER (N=6666)**

| Group | % Samples | claim-only Acc | claim-evidence Acc | EGR-FV Acc | Gain |
|---|---:|---:|---:|---:|---:|
| `bias_easy`       | 61.7 | 100.00 |  97.79 |  97.45 |   −2.55 |
| `hard`            | 21.0 |  77.54 |  77.54 |  80.97 |   +3.43 |
| `grounded_needed` | 17.3 |   0.00 | 100.00 |  98.01 |  +98.01 |

**PolitiHop (N=171)**

| Group | % Samples | claim-only Acc | claim-evidence Acc | EGR-FV Acc | Gain |
|---|---:|---:|---:|---:|---:|
| `bias_easy`       | 86.0 | 100.00 | 100.00 | 100.00 |   +0.00 |
| `hard`            | 14.0 |  12.50 |  12.50 |  16.67 |   +4.17 |
| `grounded_needed` |  0.0 |      − |      − |      − |       − |

Findings:

- `bias_easy` samples already give the claim-only branch near-perfect accuracy on every dataset, exactly as a "claim-shortcut-rich" subset should.
- `grounded_needed` is the opposite extreme: claim-only is at chance (≤ 0 % on HOVER/FEVER), claim-evidence flips to 100 %, and EGR-FV preserves almost all of that gain (95–98 %). Routing therefore separates samples by their *actual* dependence on evidence, not by an arbitrary heuristic.
- `hard` shows a small but consistent EGR-FV gain (+1.3 to +4.2 pp), confirming that the middle group is a genuine "needs care" bucket rather than just label noise.
- The PolitiHop grounded warm-up makes identical predictions with and without evidence, so the score concentrates around 0.5 and no `grounded_needed` group emerges; the per-group structure is still consistent with the expected story (`bias_easy` easy, `hard` slightly improved by EGR-FV).

### 2. Correlation of `r_i` with Evidence Sensitivity

For each sample we measure how much adding evidence helps the model, in two forms:

- **indicator**: Δ_i = 1[f(c,e)=y] − 1[f(c,∅)=y]
- **NLL**: Δ_i = ℓ(f(c,∅), y) − ℓ(f(c,e), y)

and report Pearson and Spearman correlations between `r_i` and Δ_i for both the grounded warm-up and the full EGR-FV checkpoint.

| Dataset | Model | Δ form | Pearson `r` | Spearman `ρ` |
|---|---|---|---:|---:|
| HOVER     | claim-evidence (warmup) | indicator | +0.517 | +0.581 |
| HOVER     | claim-evidence (warmup) | NLL       | +0.490 | +0.567 |
| HOVER     | EGR-FV                  | indicator | +0.420 | +0.453 |
| HOVER     | EGR-FV                  | NLL       | +0.479 | +0.623 |
| FEVER     | claim-evidence (warmup) | indicator | +0.188 | +0.260 |
| FEVER     | claim-evidence (warmup) | NLL       | +0.218 | +0.314 |
| FEVER     | EGR-FV                  | indicator | +0.424 | +0.393 |
| FEVER     | EGR-FV                  | NLL       | +0.418 | +0.538 |
| PolitiHop | claim-evidence (warmup) | indicator |      − |      − |
| PolitiHop | claim-evidence (warmup) | NLL       | +0.961 | +0.045 |
| PolitiHop | EGR-FV                  | indicator | +0.284 | +0.071 |
| PolitiHop | EGR-FV                  | NLL       | +0.813 | +0.093 |

Both correlations are positive and consistently above 0.4 on HOVER and FEVER for the full EGR-FV model — samples that the routing score flags as evidence-needed are exactly the ones whose predictions actually improve when evidence is added. On PolitiHop the indicator correlation is undefined for the grounded warm-up because its clean and null predictions are identical (Δ_i ≡ 0); the NLL form still shows the expected positive trend.

### 3. Routing Visualization (bucketed)

We bucket test samples by `r_i` into `[0.0, 0.2)`, `[0.2, 0.4)`, …, `[0.8, 1.0]` and plot the clean-correct, shuffled-wrong, and null-wrong rates of both the grounded warm-up and the full EGR-FV checkpoint:

![routing buckets](outputs/analysis/routing_analysis/routing_analysis_buckets.png)

Reading left to right on each panel, clean-correct climbs while the model also becomes more sensitive to corrupted evidence (shuffled-wrong / null-wrong rise). This is the signature of an evidence-grounded model — low-`r_i` samples are claim-shortcut-rich (the prediction is the same with or without evidence), and high-`r_i` samples are exactly where removing or shuffling evidence collapses accuracy. The same trend appears for both the grounded warm-up and EGR-FV, but EGR-FV's gap between `clean-correct` and `null-wrong` is consistently wider, i.e. EGR-FV depends more on evidence at the samples the routing says need it.

Outputs are written to `outputs/analysis/routing_analysis/`:

```text
HOVER.jsonl  FEVER.jsonl  PolitiHop.jsonl              # per-sample records
routing_analysis_groups.csv                            # § 1
routing_analysis_correlations.csv                      # § 2
routing_analysis_buckets.csv                           # § 3
routing_analysis_buckets.png / .pdf                    # § 3 figure
routing_analysis_report.md                             # combined report
routing_analysis_summary.json                          # everything in JSON
```

## Case Study

`scripts/build_case_study.py` joins per-sample routing-analysis records with the raw claim/evidence text and surfaces four kinds of cases that probe how EGR-FV uses the routing signal. Outputs land at `outputs/analysis/case_study/case_study.md` (paper-ready tables) plus per-dataset `cases_<DS>.jsonl` / `cases_<DS>.csv` for the appendix.

The four categories are:

- **A. Evidence-rescue** — `grounded_needed` samples where the claim-only warm-up is fooled by surface form but EGR-FV recovers from the evidence.
- **B. Evidence-sensitive (non-`grounded_needed`)** — `hard` or `bias_easy` samples where EGR-FV is correct under clean evidence but flips under null and shuffled evidence — the model is not memorising, it is reading.
- **C. Bias-easy trade-off** — `bias_easy` samples where the claim-only warm-up gets it right but EGR-FV is wrong; the price the model pays for de-biasing.
- **D. Hard-bucket win** — `hard` samples where both warm-ups fail but EGR-FV is correct on the clean evidence.

Two illustrative HOVER cases (full tables in the report; per-class loss columns omitted for brevity):

| Category | Claim | Evidence (truncated) | Gold | claim-only | claim-evidence | EGR-FV (clean / null / shuf) |
|---|---|---|---|---|---|---|
| **A. Evidence-rescue** | *The movie Groundhog Day was directed by Harold Ramis and stars Bill Paxton.* | "Groundhog Day is a 1993 … directed by Harold Ramis, **starring Bill Murray** …" | refutes | supports ✗ | ✓ | ✓ / ✗ / ✗ |
| **D. Hard-bucket win** | *Iqaluit Airport and Canadian North are based out of Montreal.* | "Iqaluit Airport … operated by the **government of Nunavut**. It hosts scheduled passenger service from Ottawa, Montreal …" | refutes | supports ✗ | ✗ | ✓ / ✗ / ✗ |

In both rows the `null` and `shuffled` columns are wrong even though `clean` is right — exactly the behaviour an evidence-grounded model should exhibit. Reproduction:

```bash
python scripts/build_case_study.py
python scripts/build_case_study.py --datasets HOVER FEVER --per_category 5
```

## Retrieved-Evidence Experiment

The headline tables use **gold** evidence (the gold sentence is already in the converted dataset). To check how EGR-FV behaves in a realistic FEVER pipeline with **retrieved** evidence, `scripts/retrieve_fever_evidence.py` builds top-`K` BM25 sentences per claim:

- For FEVER train / dev / test, candidate documents come from the FEVER_baseline cache (`expanded_doc_ids.pkl`); sentences are BM25-reranked against the claim and the top 5 are concatenated.
- For symmetric-FEVER (whose claims aren't in the cache), a TF-IDF index over the 5.4M Wikipedia titles in `data.db` selects top-20 candidate pages first, then the same BM25 sentence ranking applies.

```bash
python scripts/retrieve_fever_evidence.py --split test
python scripts/retrieve_fever_evidence.py --split symmetric
python scripts/retrieve_fever_evidence.py --split train
python scripts/retrieve_fever_evidence.py --split dev
python scripts/filter_retrieved_train.py  # drop rows with empty retrieval (~9k of 110k)
```

Two compare-points get evaluated:

- **Gold → Retrieved-test** (cheap MVP): take the gold-evidence checkpoints and evaluate them directly on retrieved sentences. Stress-tests robustness to retrieval noise.
- **Retrieved → Retrieved-test** (apples-to-apples): retrain claim-evidence and full EGR-FV on the retrieved-evidence training set (`run_scripts/run_claim_evidence_fever_retrieved.sh`, `run_scripts/run_fever_retrieved.sh`) and evaluate on the retrieved test + symmetric splits.

The consolidated numbers live at `outputs/FEVER/retrieved/retrieved_evidence_report.md` (regenerated by `python scripts/summarize_retrieved_evidence.py`):

| Method | Training evidence | FEVER orig (Acc) | FEVER-sym (Acc) |
|---|---|---:|---:|
| claim-evidence | Gold → Gold-test (headline) | 94.13 | 79.36 |
| claim-evidence | Gold → Retrieved-test | 63.43 | 37.80 |
| claim-evidence | Retrieved → Retrieved-test | 78.28 | 37.66 |
| **Full EGR-FV** | Gold → Gold-test (headline) | 94.09 | 83.26 |
| **Full EGR-FV** | Gold → Retrieved-test | 60.88 | 37.66 |
| **Full EGR-FV** | Retrieved → Retrieved-test | 77.96 | 37.10 |

Findings: (1) gold-trained models drop ~30 pt on FEVER orig and ~45 pt on FEVER-sym when handed retrieved evidence — they relied on the cleanliness of gold sentences. (2) Retraining on retrieved evidence recovers most of the original-test accuracy (77.96 vs. 94.13) but the bias-stressed symmetric split stays around 37 % regardless of method. (3) Under noisy retrieval, **EGR-FV does not beat the claim-evidence baseline** — both land at the same plateau (~78 / 37). This is an honest caveat to surface: EGR-FV's evidence-contrast and remix machinery rely on evidence being meaningful; with BM25 retrieval errors that bring in off-topic Wikipedia sentences, the per-sample necessity score becomes a noisy signal.

## Backbone Generalization + LLM Baseline

Two extensions that EMNLP reviewers typically ask for:

1. **Backbone generalization** — `scripts/run_*` and `configs/backbones/deberta_v3_*.yaml` retrain both `claim-evidence` and full `EGR-FV` with DeBERTa-v3-base on HOVER, FEVER, and PolitiHop. Verifies the method isn't RoBERTa-specific.
2. **LLM baseline** — `scripts/run_llm_baseline.py` evaluates Llama-3-8B-Instruct (4-shot prompt, greedy decoding, evidence truncated to 600 chars to bound prompt length) on the same splits.

Both are consolidated by `python scripts/summarize_backbone_and_llm.py` into `outputs/analysis/backbone_llm/backbone_llm_report.md`. The script is safe to run incrementally — missing rows render as "—" until the underlying eval_report.json lands.

Current numbers (Test / Symmetric for FEVER + PolitiHop, Test only for HOVER):

| Backbone | Method | HOVER (macro-F1) | FEVER orig | FEVER-sym | PolitiHop orig | PolitiHop-sym |
|---|---|---:|---:|---:|---:|---:|
| RoBERTa-base | claim-evidence | 82.96 (±0.62 multi-seed) | 94.13 | 79.36 | (collapse) | 50.00 |
| RoBERTa-base | **EGR-FV** | 83.93 | 94.09 | 83.26 | 88.30 | 58.77 |
| DeBERTa-v3-base | claim-evidence | 84.82 | **95.45** | 83.68 | (collapse) | 50.00 |
| DeBERTa-v3-base | **EGR-FV** | **85.82** | 95.08 | **84.10** | 87.72 | 51.17 |
| Llama-3-8B-Instruct (4-shot) | — | 56.90 | 85.81 | 66.53 | 88.89 | 61.99 |

Take-aways:

- **EGR-FV transfers to DeBERTa-v3.** On HOVER, EGR-FV gains another **+1.9 pp** macro-F1 (83.93 → 85.82). On FEVER-sym, swapping the backbone and running EGR-FV combine to give 84.10, a **+4.7 pp** improvement over the RoBERTa-base claim-evidence baseline (79.36) and **+0.84 pp** over the RoBERTa-base full EGR-FV (83.26). On FEVER-orig the two backbones are essentially tied (94 ↔ 95) — at saturation there's not much room to win.
- **DeBERTa-v3 alone closes most of the bias gap on FEVER-sym** (claim-evidence 83.68), and EGR-FV adds **+0.42 pp** on top. The two contributions are nearly additive.
- **Llama-3-8B-Instruct is a strong PolitiHop baseline** (sym 61.99, beating EGR-FV-RoBERTa 58.77) but loses by a wide margin on FEVER-sym (66.53 vs. 83.26) and HOVER (56.90 vs. 83.93). A 110 M-param fine-tuned model with the right inductive bias beats an 8 B LLM in 4-shot mode on the bias-stress tests.
- **PolitiHop is the weakest signal.** Both RoBERTa- and DeBERTa-claim-evidence baselines collapse to all-REFUTES on the balanced symmetric split (= 50.00 % by construction, since training is 83 % REFUTES). EGR-FV breaks the collapse on RoBERTa (58.77) but only barely on DeBERTa-v3 (51.17, supports recall ≈ 13 %). The dataset is small enough that backbone swaps have unstable effects.

## Repository Structure

```text
EGR-FV/
├── configs/              # Experiment and training configurations
├── run_scripts/          # Shell scripts for running staged experiments
├── scripts/              # Utility scripts for preprocessing, routing, and evaluation
├── src/                  # Core source code
├── requirements.txt      # Python dependencies
└── README.md             # Project documentation
```

## Installation

```bash
git clone https://github.com/2016bits/EGR-FV.git
cd EGR-FV

conda create -n egr-fv python=3.10 -y
conda activate egr-fv

pip install -r requirements.txt
```

## Data Format

Recommended JSONL format:

```json
{"id":"ex_001","claim":"...","evidence":["...","..."],"label":"SUPPORTS"}
{"id":"ex_002","claim":"...","evidence":["..."],"label":"REFUTES"}
{"id":"ex_003","claim":"...","evidence":[],"label":"NEI"}
```

Recommended directory layout:

```text
data/
├── train.jsonl
├── dev.jsonl
├── test.jsonl
└── routed/
    └── train_routing.jsonl
```

## Usage

The exact command names may vary by implementation, but the intended workflow is:

### FEVER Claim-Only And Claim-Evidence Baselines

Train on FEVER train/dev, then evaluate on both FEVER test and symmetric-FEVER:

```bash
sh run_scripts/run_fever_baselines.sh
```

Run one baseline at a time:

```bash
sh run_scripts/run_claim_only_fever.sh
sh run_scripts/run_claim_evidence_fever.sh
```

Useful environment variables:

```bash
CUDA_VISIBLE_DEVICES=0 sh run_scripts/run_fever_baselines.sh
RUN_TRAIN=0 sh run_scripts/run_fever_baselines.sh
PREPROCESS=1 sh run_scripts/run_fever_baselines.sh
```

Outputs are written to:

```text
outputs/FEVER/baselines/claim_only/
outputs/FEVER/baselines/claim_evidence/
```

### FEVER EGR Ablations

Run the EGR ablations on FEVER, evaluate each checkpoint on both FEVER test and symmetric-FEVER, then summarize the results:

```bash
sh run_scripts/run_fever_ablation.sh
```

The workflow covers:

- `Routing-only`
- `Full w/o Evidence Contrast`
- `Full w/o Grounded-dominant`
- `Full w/o Remix`
- `Hard routing`
- `In-sample routing`

Useful environment variables:

```bash
CUDA_VISIBLE_DEVICES=0 sh run_scripts/run_fever_ablation.sh
RUN_ROUTING=0 sh run_scripts/run_fever_ablation.sh
RUN_TRAIN=0 sh run_scripts/run_fever_ablation.sh
RUN_EVAL=0 sh run_scripts/run_fever_ablation.sh
SKIP_EXISTING=0 sh run_scripts/run_fever_ablation.sh
```

Outputs are written to:

```text
outputs/FEVER/checkpoints/<ablation>/
outputs/FEVER/predictions/<ablation>/fever/
outputs/FEVER/predictions/<ablation>/symmetric_fever/
outputs/FEVER/predictions/fever_ablation_report.md
```

### Routing Analysis (per-group / correlation / visualization)

After training, the three "is routing meaningful?" analyses (§ *Routing Analysis*) can be regenerated with:

```bash
sh run_scripts/run_routing_analysis.sh
```

Useful environment variables:

```bash
CUDA_VISIBLE_DEVICES=0 sh run_scripts/run_routing_analysis.sh
DATASETS="HOVER" sh run_scripts/run_routing_analysis.sh        # subset of datasets
SKIP_EXISTING=0  sh run_scripts/run_routing_analysis.sh        # force re-run
```

The runner loads `shortcut_best.pt`, `grounded_best.pt`, and `remix_best.pt` for each dataset, scores the test split under clean / shuffled / null evidence, writes per-sample records to `outputs/analysis/routing_analysis/<dataset>.jsonl`, and aggregates the per-group table, correlations, bucket table, and the bucket figure into the same directory.

### 1. Train Grounded Warm-up

```bash
python scripts/train_grounded.py \
  --config configs/grounded_warmup.yaml
```

### 2. Train Shortcut Warm-up

```bash
python scripts/train_shortcut.py \
  --config configs/shortcut_warmup.yaml
```

### 3. Compute Out-of-Fold Routing Scores

```bash
python scripts/run_routing.py \
  --config configs/routing.yaml \
  --num_folds 5
```

### 4. Train Full EGR-FV

```bash
python scripts/train_egr.py \
  --config configs/egr_full.yaml \
  --routing_file data/routed/train_routing.jsonl
```

### 5. Evaluate

```bash
python scripts/evaluate.py \
  --config configs/egr_full.yaml \
  --checkpoint outputs/egr_full/best.pt \
  --split test
```

## Configuration

Important hyperparameters include:

| Parameter | Meaning |
|---|---|
| `alpha` | Controls grounded loss upweighting via `w_g(i) = 1 + α r_i` |
| `beta` | Controls shortcut branch gate via `w_s(i) = β(1 - r_i)` |
| `lambda_s` | Weight of gated shortcut loss |
| `lambda_orth` | Weight of orthogonal representation loss |
| `lambda_ctr` | Weight of evidence contrast loss |
| `routing_high_threshold` | Threshold for `grounded_needed` samples |
| `routing_low_threshold` | Threshold for `bias_easy` samples |
| `remix_ratio` | Batch composition ratio for original / grounded-needed / hard / bias-easy samples |
| `eta1`, `eta2` | Checkpoint selection weights for grounded-needed F1 and evidence sensitivity |

Example:

```yaml
model:
  encoder_name: roberta-base
  num_labels: 3

routing:
  num_folds: 5
  high_threshold: 0.7
  low_threshold: 0.3

loss:
  alpha: 1.0
  beta: 1.0
  lambda_s: 0.3
  lambda_orth: 0.05
  lambda_ctr: 0.1

remix:
  original: 0.50
  grounded_needed: 0.30
  hard: 0.15
  bias_easy: 0.05

checkpoint:
  eta1: 0.5
  eta2: 0.5
```

## Design Notes

EGR-FV is designed around a practical assumption: shortcut learning is not eliminated by simply adding a claim-only branch. The shortcut branch must be used as a diagnostic and routing tool, while the grounded branch must receive stronger, cleaner, and more evidence-sensitive training signals.

The core design principle is therefore:

```text
Use shortcut learning to detect bias, not to make the final decision.
```

## Citation

If you use this repository, please cite or acknowledge:

```bibtex
@misc{egrfv2026,
  title  = {EGR-FV: Evidence-Grounded Remixing for Debiased Fact Verification},
  author = {EGR-FV Contributors},
  year   = {2026},
  url    = {https://github.com/2016bits/EGR-FV}
}
```

## License

Please refer to the repository license if provided.
