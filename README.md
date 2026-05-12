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

The main result shows that the full EGR pipeline achieves the best overall performance.

| Method | 2-hop | 3-hop | 4-hop | Overall |
|---|---:|---:|---:|---:|
| Grounded-only | 82.14 | 82.07 | 80.65 | 81.72 |
| Two-branches Joint | 82.40 | 83.01 | 81.61 | 82.74 |
| Routing-only | 82.85 | 83.95 | 82.19 | 83.18 |
| Random Remix-only | 82.40 | 82.75 | 82.48 | 82.58 |
| Full w/o Remix | 82.12 | 83.19 | 82.58 | 82.73 |
| Full w/o Evidence Contrast | 82.58 | 83.69 | 82.00 | 82.94 |
| Full w/o Grounded-dominant | 82.67 | 84.04 | 81.52 | 83.00 |
| Full hard routing | 82.14 | 83.88 | 81.62 | 82.80 |
| Full in-sample routing | 82.58 | 82.94 | 82.29 | 82.67 |
| **EGR (full pipeline)** | **82.94** | **84.07** | **83.05** | **83.49** |

## Ablation Study

The ablation results support several conclusions:

- **Routing is important**: `Routing-only` improves over `Grounded-only`, showing that identifying evidence-needed examples is useful.
- **Random remix is insufficient**: `Random Remix-only` improves less than routing-aware training, showing that remixing should be guided by evidence necessity.
- **Evidence contrast helps**: removing evidence contrast reduces overall performance from `83.49` to `82.94`.
- **Grounded-dominant inference matters**: using the grounded branch as the final predictor avoids reverting to shortcut-heavy predictions.
- **Out-of-fold routing is more reliable**: `Full in-sample routing` underperforms the full pipeline, suggesting that in-sample routing can leak training bias into routing decisions.
- **The full pipeline is best**: EGR achieves the strongest results on 2-hop, 3-hop, 4-hop, and overall metrics.

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
