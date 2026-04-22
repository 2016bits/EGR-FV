# EGR-FV: Evidence-Grounded Remixing for Debiased Fact Verification

> A practical implementation-oriented framework for debiased fact verification via **claim-evidence grounded learning**, with **sample decoupling**, **branch-specific optimization**, and **batch-level reassembling**.

---

## 1. Project Goal

本项目旨在实现一种**以 evidence-grounded reasoning 为核心**的事实核查训练框架，解决传统 fact verification 模型中过度依赖 **claim-only shortcut** 的问题。

在标准事实核查任务中，输入通常为：

- `claim`
- `evidence`
- `label ∈ {SUPPORTS, REFUTES, NEI}`

虽然模型表面上接收 `(claim, evidence)` 作为输入，但实际训练后常常学到的是：

- claim 中的表面模式
- 实体共现偏置
- 标签先验
- 数据集中的 shortcut source

而不是：

- claim 与 evidence 的真实对齐关系
- evidence 是否支持 / 反驳 claim
- 证据缺失时的 NEI 判断依据

因此，本项目的核心目标是：

1. **显式分离 shortcut branch 与 grounded branch**
2. **识别哪些样本更依赖 evidence**
3. **通过样本分流和 batch 重组，改变训练梯度结构**
4. **让 grounded branch 成为最终主导分支**
5. **提升 evidence utilization、鲁棒性和泛化能力**

---

## 2. Core Intuition

我们将事实核查中的模型行为拆解为两种不同能力：

### 2.1 Shortcut Ability

模型只根据 claim 自身的语言模式做预测：

```text
p_s(y | c)
```

这种能力可能来自：

- 实体或关系模式
- 常识偏置
- 训练集标签统计规律
- claim wording 的表面信号

这类能力在训练初期通常学得更快，因此容易主导整体优化。

---

### 2.2 Grounded Reasoning Ability

模型通过 claim 与 evidence 的交互，真正基于证据完成判断：

```text
p_g(y | c, e)
```

这种能力才是我们希望模型真正掌握的事实核查能力。

---

### 2.3 Main Idea

本项目借鉴 **Data Remixing** 的思想，将训练拆成两个核心阶段：

#### (1) Decouple

从样本角度识别：

- 哪些样本容易被 shortcut branch 正确预测
- 哪些样本需要 evidence 才能做对
- 哪些样本属于 hard / ambiguous case

#### (2) Reassemble

在 batch 层面重新组织样本：

- shortcut-heavy batch
- grounded-needed batch
- mixed batch

通过这种训练组织方式，使 grounded branch 在关键样本上获得更强、更稳定的学习信号。

---

## 3. Task Definition

给定一个样本：

```json
{
  "id": "sample_001",
  "claim": "The song recorded by Fergie that was produced by Polow da Don and was followed by Life Goes On was M.I.L.F.$.",
  "evidence": [
    "It was produced by Polow da Don and released as the second single from the record.",
    "The song serves as the third single from Fergie's second studio album, following \"M.I.L.F. $\"."
  ],
  "label": "SUPPORTS"
}
```

目标是训练分类器：

```text
f(claim, evidence) -> {SUPPORTS, REFUTES, NEI}
```

其中：

- `SUPPORTS`：evidence 支持 claim
- `REFUTES`：evidence 反驳 claim
- `NEI`：evidence 不足以判断

---

## 4. Method Overview

整个方法由五个核心模块组成：

1. **Shortcut Branch**
2. **Grounded Branch**
3. **Bias / Necessity Scoring**
4. **Sample Routing**
5. **Batch Reassembling + Debiased Optimization**

整体流程如下：

```text
Raw samples
   ↓
Warm-up shortcut & grounded branches
   ↓
Compute bias / disagreement / necessity scores
   ↓
Route samples into groups
   ↓
Reassemble batches by group
   ↓
Joint debiased training
   ↓
Grounded-dominant inference
```

---

## 5. Model Design

## 5.1 Shortcut Branch

### Goal

只输入 claim，建模 claim-only shortcut pattern。

### Input

- claim

### Output

- `shortcut_logits`
- `shortcut_probs`
- `shortcut_hidden`

### Typical Implementation

#### Option A: BERT / RoBERTa encoder + classifier

```python
claim -> tokenizer -> encoder -> CLS -> linear -> logits_s
```

#### Option B: lightweight classifier

适用于快速打分、样本分流和 warm-up。

### Interface Suggestion

```python
class ShortcutModel(nn.Module):
    def __init__(self, encoder_name: str, num_labels: int):
        ...
    def forward(self, input_ids, attention_mask):
        return {
            "logits": logits,
            "probs": probs,
            "hidden": hidden
        }
```

### Expected File

```text
src/models/shortcut_model.py
```

---

## 5.2 Grounded Branch

### Goal

输入 claim + evidence，学习真正的 grounded verification。

### Input

- claim
- evidence

### Output

- `grounded_logits`
- `grounded_probs`
- `grounded_hidden`

### Typical Implementation

#### Option A: Cross-Encoder

输入格式：

```text
[CLS] claim [SEP] evidence [SEP]
```

优点：

- 实现简单
- claim-evidence token-level interaction 强
- 适合第一版 baseline

#### Option B: Dual Encoder + Interaction

适用于长 evidence 或多证据聚合场景。

### Interface Suggestion

```python
class GroundedModel(nn.Module):
    def __init__(self, encoder_name: str, num_labels: int):
        ...
    def forward(self, input_ids, attention_mask):
        return {
            "logits": logits,
            "probs": probs,
            "hidden": hidden
        }
```

### Expected File

```text
src/models/grounded_model.py
```

---

## 5.3 Final Prediction Head

建议分为两种模式：

### Mode A: Grounded-only inference

训练时 shortcut branch 用于辅助去偏和 routing，推理时只使用 grounded branch：

```text
ŷ = argmax p_g(y | c, e)
```

这是**推荐默认方案**。

### Mode B: Gated Fusion

训练 / 推理时融合两分支输出：

```text
α = sigmoid(W[h_s ; h_g])
p = α * p_g + (1 - α) * p_s
```

但为了避免退化回 shortcut learning，建议：

- 对 `α` 做 grounded 偏置初始化
- 或对 shortcut 权重施加约束
- 或只在训练时辅助融合，推理时仍用 grounded branch

### Interface Suggestion

```python
class FusionHead(nn.Module):
    def __init__(self, hidden_size: int, num_labels: int):
        ...
    def forward(self, hs, hg, ps, pg):
        return {
            "alpha": alpha,
            "probs": p,
            "logits": logits
        }
```

### Expected File

```text
src/models/fusion_head.py
```

---

## 6. Sample Decoupling

这是整个方法的关键。

不是所有样本都应被同样对待。我们需要判断：

- 哪些样本 claim-only 就够了
- 哪些样本必须依赖 evidence
- 哪些样本两边都难

---

## 6.1 Why Sample Decoupling?

若不做分流，所有样本统一训练时会发生：

1. shortcut 分支更快收敛
2. grounded 分支被 shortcut 学到的简单模式掩盖
3. 难样本的训练信号太弱
4. evidence 的贡献被整体平均掉

因此必须先对样本进行结构化建模。

---

## 6.2 Required Scores

建议为每个样本缓存如下分数：

### (1) Shortcut confidence

```text
b_i = max p_s(y | c_i)
```

表示 claim-only 分支对该样本的最大置信度。

---

### (2) Shortcut correctness

```text
shortcut_correct_i = 1[argmax p_s == y]
```

---

### (3) Grounded confidence

```text
g_i = max p_g(y | c_i, e_i)
```

---

### (4) Grounded correctness

```text
grounded_correct_i = 1[argmax p_g == y]
```

---

### (5) Branch disagreement

可选定义：

#### KL divergence

```text
d_i = KL(p_g || p_s)
```

#### Cosine distance

```text
d_i = 1 - cos(z_g, z_s)
```

#### Prediction mismatch

```text
d_i = 1[argmax p_g != argmax p_s]
```

实践中可以同时缓存多个版本。

---

### (6) Evidence necessity score

建议定义一个组合分数，用于判断 evidence 是否必要：

```text
n_i = w1 * 1[p_g correct and p_s wrong]
    + w2 * 1[shortcut high-conf but wrong]
    + w3 * disagreement_i
    + w4 * uncertainty_term
```

其中：

- `p_g correct and p_s wrong`：最强 grounded-needed 信号
- `shortcut high-conf but wrong`：典型 shortcut trap
- `disagreement_i`：两分支语义差异
- `uncertainty_term`：如 grounded margin 或 entropy

---

## 6.3 Sample Groups

推荐划分为三类。

### Group A: bias_easy

定义特征：

- shortcut 置信度高
- shortcut 预测正确
- grounded 与 shortcut 差异小

这类样本容易强化 claim-only 偏置。

#### 训练策略

- 降低在 joint training 中的占比
- 可保留给 shortcut 分支 warm-up
- 不要让这类样本主导 grounded branch 的更新

---

### Group B: grounded_needed

定义特征：

- `p_g` 正确而 `p_s` 错误
- 或者 branch disagreement 高
- 或 evidence necessity score 高

这类样本最能体现 evidence-grounded reasoning 的价值。

#### 训练策略

- 在 grounded batch 中提高占比
- 对 grounded loss 赋更高权重
- 在 curriculum 中优先使用

---

### Group C: hard

定义特征：

- 两分支都不稳定
- 两分支都可能错误
- 或 evidence 复杂、歧义大、语义链长

#### 训练策略

- 保留一定比例
- 后期加入 mixed / hard batch
- 用于提高鲁棒性

---

## 6.4 Recommended Routing Rules

### Rule Version 1: Simple Threshold

```python
if shortcut_conf >= tau_high and shortcut_correct and disagreement <= tau_low:
    group = "bias_easy"
elif grounded_correct and not shortcut_correct:
    group = "grounded_needed"
else:
    group = "hard"
```

### Rule Version 2: Score-based Ranking

1. 计算 `necessity_score`
2. 对全部样本排序
3. 取：
   - top 30% → `grounded_needed`
   - bottom 30% → `bias_easy`
   - middle 40% → `hard`

### Rule Version 3: Hybrid Routing

先用规则打标签，再用排序修正边界样本。

---

## 6.5 Routing Cache Format

建议将路由结果写入文件，避免每次训练都重复计算：

```json
{
  "id": "sample_001",
  "shortcut_conf": 0.91,
  "shortcut_correct": true,
  "grounded_conf": 0.73,
  "grounded_correct": true,
  "disagreement": 0.48,
  "necessity_score": 0.77,
  "group": "grounded_needed"
}
```

### Expected File

```text
data/routed/train_routing.jsonl
```

### Expected Script

```text
src/data/routing.py
scripts/run_routing.sh
```

---

## 7. Batch Reassembling

在获得 sample groups 之后，下一步是 batch-level remix。

---

## 7.1 Why Reassemble at Batch Level?

即使我们已经知道哪些样本更需要 evidence，如果仍然随机采样训练，依然会发生：

- bias_easy 样本数量更多
- batch 平均梯度仍然偏向 shortcut
- grounded-needed 样本信号被稀释

因此必须控制每个 batch 的样本组成。

---

## 7.2 Batch Types

### 1) Bias Batch

主要由 `bias_easy` 样本组成。

#### Purpose

- 稳定 shortcut predictor
- 维护 shortcut branch 的可解释性
- 为后续 routing 提供可靠分数

#### Notes

- 不应在此 batch 上过度强化 grounded branch

---

### 2) Grounded Batch

主要由 `grounded_needed` 样本组成。

#### Purpose

- 强化 evidence utilization
- 提高 grounded reasoning
- 提升困难验证样本的可学习性

#### Notes

- 这是 joint training 的核心 batch 类型

---

### 3) Mixed Batch

由三类样本按比例混合：

- bias_easy
- grounded_needed
- hard

#### Purpose

- 保持分布稳定
- 提高泛化
- 防止训练仅对某类样本过拟合

---

## 7.3 Recommended Remix Schedules

### Schedule A: Alternating

```text
bias -> grounded -> mixed -> bias -> grounded -> mixed
```

实现简单，适合第一版。

---

### Schedule B: Fixed Ratio per Epoch

例如：

- 25% bias batch
- 50% grounded batch
- 25% mixed batch

适合更稳定训练。

---

### Schedule C: Curriculum

#### Early stage

- bias batch 较多
- 目的是稳定 shortcut 表征与 routing 分数

#### Middle stage

- grounded batch 占主导
- 开始真正强化 grounded reasoning

#### Late stage

- 增加 hard / mixed batch
- 提升鲁棒性与边界样本处理能力

---

## 7.4 Sampler Interface

建议单独写一个 remix sampler：

```python
class RemixBatchScheduler:
    def __init__(self, bias_loader, grounded_loader, mixed_loader, schedule_type="alternating"):
        ...
    def next_batch(self, global_step, epoch):
        return batch_type, batch
```

### Expected File

```text
src/data/remix_sampler.py
```

---

## 8. Training Objectives

## 8.1 Shortcut Loss

```text
L_s = CE(p_s, y)
```

作用：

- 学习 shortcut predictor
- 提供 routing 所需的置信度和偏置信号

---

## 8.2 Grounded Loss

```text
L_g = CE(p_g, y)
```

作用：

- 学习 claim-evidence grounded semantics
- 作为最终主损失

---

## 8.3 Weighted Grounded Loss

对不同样本赋予不同权重：

```text
L_g_weighted = w_i * CE(p_g^i, y_i)
```

建议：

- `grounded_needed` 样本权重大
- `bias_easy` 样本权重小
- `hard` 样本居中

一个简单示例：

```python
if group == "grounded_needed":
    weight = 1.5
elif group == "hard":
    weight = 1.0
else:
    weight = 0.5
```

---

## 8.4 Representation Disentanglement Loss

为了降低 shortcut 表征对 grounded 表征的干扰，可加入解耦损失。

### Option A: Cosine orthogonality

```text
L_orth = cos(h_s, h_g)^2
```

### Option B: Dot-product penalty

```text
L_orth = ||h_s^T h_g||^2
```

---

## 8.5 Residual Grounding Loss

鼓励 grounded 分支学习 shortcut 之外的增量信息：

```text
z_res = z_g - Proj(z_s)
```

然后在 `z_res` 上做监督，或约束其携带真实标签信息。

适合第二阶段扩展，不建议第一版就加太复杂。

---

## 8.6 Calibration / Consistency Loss

可选加入：

- grounded prediction consistency
- evidence dropout consistency
- confidence calibration loss

例如：

```text
L_cons = KL(p_g_full || p_g_dropout)
```

---

## 8.7 Total Loss

推荐第一版总损失：

```text
L = L_g_weighted + λ_s * L_s + λ_o * L_orth
```

推荐默认起点：

- `λ_s = 0.3`
- `λ_o = 0.05`

更保守版本：

```text
L = L_g + 0.2 * L_s
```

如果训练不稳定，可先去掉 `L_orth`。

### Expected File

```text
src/models/losses.py
```

---

## 9. Training Pipeline

建议按以下阶段实现。

---

## 9.1 Stage 0: Data Preparation

### Input format

建议使用 JSONL：

```json
{"id":"1","claim":"...","evidence":["...","..."],"label":"SUPPORTS"}
{"id":"2","claim":"...","evidence":["..."],"label":"REFUTES"}
```

### Recommended preprocessing

1. 将 evidence list 合并为单段文本
2. 保留原始 evidence list 以便后续分析
3. 映射标签到 id：
   - `SUPPORTS -> 0`
   - `REFUTES -> 1`
   - `NEI -> 2`

### Expected Files

```text
src/data/dataset.py
src/data/collator.py
```

---

## 9.2 Stage 1: Warm-up

### Goal

先分别让 shortcut 与 grounded 分支具备基础判别能力。

### Recommended procedure

#### Step 1

单独训练 shortcut branch 若干 epoch

#### Step 2

单独训练 grounded branch 若干 epoch

#### Step 3

在验证集上保存：

- `shortcut checkpoint`
- `grounded checkpoint`

### Why needed?

因为若一开始就做 routing，分数会非常不稳定。

### Suggested scripts

```text
scripts/run_warmup_shortcut.sh
scripts/run_warmup_grounded.sh
```

---

## 9.3 Stage 2: Bias / Necessity Scoring

对训练集每个样本做一次前向推理，得到：

- `shortcut_conf`
- `shortcut_pred`
- `shortcut_correct`
- `grounded_conf`
- `grounded_pred`
- `grounded_correct`
- `disagreement`
- `necessity_score`

然后写入 routing cache。

### Output

```text
data/routed/train_routing.jsonl
```

---

## 9.4 Stage 3: Sample Routing

根据预设规则将样本映射到 group：

- `bias_easy`
- `grounded_needed`
- `hard`

建议同时输出统计信息：

```json
{
  "num_total": 10000,
  "num_bias_easy": 3200,
  "num_grounded_needed": 4100,
  "num_hard": 2700
}
```

### Why this matters?

便于观察分布是否合理。如果 `grounded_needed` 太少，说明 routing 规则可能过严。

---

## 9.5 Stage 4: Build Reassembled Loaders

使用 routing 文件构建 3 个 dataset / dataloader：

- `bias_dataset`
- `grounded_dataset`
- `mixed_dataset`

mixed_dataset 可以有两种做法：

### Option A

从全量训练集随机采样，但带 group-aware 权重

### Option B

从 3 类中按固定比例采样，比如：

- 30% bias_easy
- 40% grounded_needed
- 30% hard

---

## 9.6 Stage 5: Joint Debiased Training

训练时同时前向：

- shortcut branch
- grounded branch
- optional fusion head

并根据 batch 类型决定损失重点。

### Bias batch

```python
loss = L_s + 0.1 * L_g
```

### Grounded batch

```python
loss = 1.5 * L_g + 0.2 * L_s + λ_o * L_orth
```

### Mixed batch

```python
loss = L_g_weighted + λ_s * L_s + λ_o * L_orth
```

### Notes

第一版不必过度追求复杂性，能稳定跑通最重要。

---

## 9.7 Stage 6: Inference

默认推理只使用 grounded branch：

```python
pred = grounded_model(claim, evidence)["probs"].argmax(dim=-1)
```

建议同时输出分析字段：

```json
{
  "id": "sample_001",
  "pred_label": "SUPPORTS",
  "grounded_conf": 0.88,
  "shortcut_conf": 0.74,
  "disagreement": 0.31
}
```

方便后续误差分析。

---

## 10. Codebase Structure

推荐如下目录结构：

```text
EGR-FV/
├── README.md
├── requirements.txt
├── configs/
│   ├── default.yaml
│   ├── shortcut.yaml
│   ├── grounded.yaml
│   ├── routing.yaml
│   └── remix.yaml
├── data/
│   ├── raw/
│   ├── processed/
│   └── routed/
├── src/
│   ├── models/
│   │   ├── shortcut_model.py
│   │   ├── grounded_model.py
│   │   ├── fusion_head.py
│   │   └── losses.py
│   ├── data/
│   │   ├── dataset.py
│   │   ├── collator.py
│   │   ├── routing.py
│   │   └── remix_sampler.py
│   ├── trainers/
│   │   ├── warmup_trainer.py
│   │   ├── remix_trainer.py
│   │   └── evaluator.py
│   ├── utils/
│   │   ├── metrics.py
│   │   ├── logger.py
│   │   ├── io.py
│   │   └── seed.py
│   └── main.py
├── scripts/
│   ├── run_warmup_shortcut.sh
│   ├── run_warmup_grounded.sh
│   ├── run_routing.sh
│   ├── run_remix.sh
│   └── run_eval.sh
└── outputs/
    ├── checkpoints/
    ├── logs/
    └── predictions/
```

---

## 11. Config Design

建议统一使用 YAML 配置。

### Example: `configs/remix.yaml`

```yaml
seed: 42
task: fact_verification
num_labels: 3

data:
  train_path: data/processed/train.jsonl
  val_path: data/processed/val.jsonl
  test_path: data/processed/test.jsonl
  routing_path: data/routed/train_routing.jsonl
  max_claim_len: 128
  max_evidence_len: 384

model:
  shortcut_encoder: roberta-base
  grounded_encoder: roberta-base
  hidden_size: 768
  dropout: 0.1
  use_fusion: false

training:
  epochs: 5
  batch_size: 16
  lr: 2e-5
  weight_decay: 0.01
  warmup_ratio: 0.1
  max_grad_norm: 1.0

loss:
  lambda_shortcut: 0.3
  lambda_orth: 0.05
  grounded_needed_weight: 1.5
  hard_weight: 1.0
  bias_easy_weight: 0.5

routing:
  tau_shortcut_high: 0.8
  tau_disagreement_low: 0.1
  strategy: hybrid

remix:
  schedule: alternating
  batch_types: [bias, grounded, mixed]
  mixed_ratio:
    bias_easy: 0.3
    grounded_needed: 0.4
    hard: 0.3
```

---

## 12. Data Module Design

## 12.1 Dataset

建议 dataset 返回统一字段：

```python
{
    "id": sample_id,
    "claim_text": claim,
    "evidence_text": evidence_text,
    "label": label_id,
    "group": group_name,               # optional
    "sample_weight": weight_value      # optional
}
```

### Responsibilities

- 读取 JSONL
- 合并 evidence 文本
- 对标签做映射
- 接入 routing 信息

---

## 12.2 Collator

建议根据分支类型支持两种编码方式：

### Shortcut collate

只编码 claim

### Grounded collate

编码 `(claim, evidence)`

### Joint collate

同时返回 shortcut 和 grounded 所需输入

### Output Example

```python
{
    "ids": [...],
    "labels": tensor(...),
    "shortcut_inputs": {
        "input_ids": ...,
        "attention_mask": ...
    },
    "grounded_inputs": {
        "input_ids": ...,
        "attention_mask": ...
    },
    "groups": [...],
    "weights": ...
}
```

---

## 13. Trainer Design

建议分成两个 trainer。

---

## 13.1 WarmupTrainer

### Responsibilities

- 支持训练 shortcut-only
- 支持训练 grounded-only
- 保存最佳 checkpoint
- 输出基础验证指标

### Expected File

```text
src/trainers/warmup_trainer.py
```

---

## 13.2 RemixTrainer

### Responsibilities

- 读取 routed samples
- 构造 reassembled loaders
- 按 schedule 获取 batch
- 联合前向 shortcut / grounded
- 计算 group-aware loss
- 记录 batch-level 统计信息

### Expected File

```text
src/trainers/remix_trainer.py
```

---

## 14. Evaluation Design

除了标准分类效果，还应评估 debiasing 是否真的有效。

---

## 14.1 Standard Metrics

- Accuracy
- Macro Precision
- Macro Recall
- Macro F1
- Per-label F1

---

## 14.2 Group-wise Metrics

分别在以下样本集上评估：

- bias_easy
- grounded_needed
- hard

希望看到：

- grounded_needed 上明显提升
- bias_easy 上不明显下降或可接受下降
- hard 上逐步改善

---

## 14.3 Evidence Sensitivity Tests

### Test A: Remove evidence

将 evidence 置空或替换为 `[NO_EVIDENCE]`

如果 grounded model 真的使用了证据，性能应该显著下降。

### Test B: Shuffle evidence

将 evidence 随机打乱到别的样本

若性能变化不大，说明模型可能没有真正 grounding。

### Test C: Claim-only inference comparison

对比：

- shortcut branch
- grounded branch
- final predictor

---

## 14.4 Calibration / Robustness

建议额外记录：

- ECE
- Brier score
- adversarial evidence robustness
- OOD dataset transfer

### Expected File

```text
src/trainers/evaluator.py
src/utils/metrics.py
```

---

## 15. Minimal Implementation Roadmap

为了尽快落地，建议按以下顺序实现。

### Step 1

实现最基础的 grounded verifier：

- `GroundedModel`
- `Dataset`
- `Collator`
- `WarmupTrainer`

### Step 2

实现 shortcut branch：

- `ShortcutModel`
- shortcut warm-up script

### Step 3

实现 routing：

- 读取两个 checkpoint
- 对 train 集打分
- 输出 `train_routing.jsonl`

### Step 4

实现 remix trainer：

- 多 loader
- schedule
- weighted loss

### Step 5

加入 disentanglement loss / fusion head / 更复杂 routing 规则

---

## 16. Pseudocode

### 16.1 Warm-up

```python
# train shortcut branch
for epoch in range(num_epochs):
    for batch in train_loader:
        out_s = shortcut_model(**batch["shortcut_inputs"])
        loss = ce(out_s["logits"], batch["labels"])
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

# train grounded branch
for epoch in range(num_epochs):
    for batch in train_loader:
        out_g = grounded_model(**batch["grounded_inputs"])
        loss = ce(out_g["logits"], batch["labels"])
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
```

---

### 16.2 Routing

```python
routing_records = []

for batch in train_loader:
    out_s = shortcut_model(**batch["shortcut_inputs"])
    out_g = grounded_model(**batch["grounded_inputs"])

    for i in range(len(batch["ids"])):
        ps = softmax(out_s["logits"][i])
        pg = softmax(out_g["logits"][i])

        shortcut_conf = ps.max().item()
        shortcut_pred = ps.argmax().item()
        grounded_conf = pg.max().item()
        grounded_pred = pg.argmax().item()

        shortcut_correct = int(shortcut_pred == batch["labels"][i].item())
        grounded_correct = int(grounded_pred == batch["labels"][i].item())

        disagreement = kl_div(pg, ps)

        necessity_score = (
            1.0 * int(grounded_correct and not shortcut_correct)
            + 0.5 * disagreement
            + 0.5 * int(shortcut_conf > 0.8 and not shortcut_correct)
        )

        group = route(shortcut_conf, shortcut_correct, grounded_correct, disagreement, necessity_score)

        routing_records.append({
            "id": batch["ids"][i],
            "shortcut_conf": shortcut_conf,
            "shortcut_correct": bool(shortcut_correct),
            "grounded_conf": grounded_conf,
            "grounded_correct": bool(grounded_correct),
            "disagreement": disagreement,
            "necessity_score": necessity_score,
            "group": group
        })
```

---

### 16.3 Remix Training

```python
for epoch in range(epochs):
    for step in range(steps_per_epoch):
        batch_type, batch = scheduler.next_batch(global_step=step, epoch=epoch)

        out_s = shortcut_model(**batch["shortcut_inputs"])
        out_g = grounded_model(**batch["grounded_inputs"])

        loss_s = ce(out_s["logits"], batch["labels"])
        loss_g = weighted_ce(out_g["logits"], batch["labels"], batch["weights"])
        loss_o = orth_loss(out_s["hidden"], out_g["hidden"])

        if batch_type == "bias":
            loss = loss_s + 0.1 * loss_g
        elif batch_type == "grounded":
            loss = 1.5 * loss_g + 0.2 * loss_s + lambda_orth * loss_o
        else:
            loss = loss_g + lambda_shortcut * loss_s + lambda_orth * loss_o

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
```

---

## 17. Suggested Shell Scripts

## 17.1 Warmup Shortcut

`scripts/run_warmup_shortcut.sh`

```bash
python -m src.main \
  --config configs/shortcut.yaml \
  --mode warmup_shortcut
```

---

## 17.2 Warmup Grounded

`scripts/run_warmup_grounded.sh`

```bash
python -m src.main \
  --config configs/grounded.yaml \
  --mode warmup_grounded
```

---

## 17.3 Run Routing

`scripts/run_routing.sh`

```bash
python -m src.main \
  --config configs/routing.yaml \
  --mode routing \
  --shortcut_ckpt outputs/checkpoints/shortcut_best.pt \
  --grounded_ckpt outputs/checkpoints/grounded_best.pt
```

---

## 17.4 Run Remix

`scripts/run_remix.sh`

```bash
python -m src.main \
  --config configs/remix.yaml \
  --mode remix
```

---

## 17.5 Evaluate

`scripts/run_eval.sh`

```bash
python -m src.main \
  --config configs/remix.yaml \
  --mode eval \
  --ckpt outputs/checkpoints/remix_best.pt
```

---

## 18. Recommended Main Entry Design

建议 `src/main.py` 支持以下 mode：

- `warmup_shortcut`
- `warmup_grounded`
- `routing`
- `remix`
- `eval`

示例：

```python
if args.mode == "warmup_shortcut":
    trainer = WarmupTrainer(...)
    trainer.train_shortcut()
elif args.mode == "warmup_grounded":
    trainer = WarmupTrainer(...)
    trainer.train_grounded()
elif args.mode == "routing":
    run_routing(...)
elif args.mode == "remix":
    trainer = RemixTrainer(...)
    trainer.train()
elif args.mode == "eval":
    evaluate(...)
```

---

## 19. Logging Recommendations

建议记录以下日志：

### Train-level

- total loss
- shortcut loss
- grounded loss
- orth loss
- learning rate

### Group-level

- 每类样本数
- 每类样本 accuracy / f1
- 每类样本平均 loss

### Batch-level

- 当前 batch type
- batch 中各 group 占比
- grounded vs shortcut 平均置信度

### Useful tools

- TensorBoard
- Weights & Biases
- CSV logger

---

## 20. Common Failure Modes

### Failure 1: Routing 全部偏向 bias_easy

可能原因：

- shortcut warm-up 太强
- necessity score 设计太弱
- grounded warm-up 不够

### Failure 2: Grounded batch 上仍然学不到 evidence

可能原因：

- evidence 编码方式太弱
- claim 与 evidence 拼接不合理
- weighted loss 太小

### Failure 3: Joint training 不稳定

可能原因：

- `λ_s` 太大
- `L_orth` 太强
- batch ratio 不合理

### Failure 4: 推理时 grounding 不敏感

可能原因：

- fusion head 过度依赖 shortcut
- 训练期间 bias_easy 仍占主导

---

## 21. Recommended Default Settings

用于第一版跑通的保守配置：

```yaml
model:
  shortcut_encoder: roberta-base
  grounded_encoder: roberta-base

training:
  epochs: 3
  batch_size: 16
  lr: 2e-5

loss:
  lambda_shortcut: 0.2
  lambda_orth: 0.0

routing:
  strategy: hybrid
  tau_shortcut_high: 0.8
  tau_disagreement_low: 0.1

remix:
  schedule: alternating
  mixed_ratio:
    bias_easy: 0.3
    grounded_needed: 0.4
    hard: 0.3
```

原因：

- 先追求稳定
- 暂时不引入太强的 disentanglement regularization
- 等第一版通了再加复杂模块

---

## 22. Future Extensions

后续可以扩展到：

1. **Multi-evidence verification**
2. **Sentence selection + verification joint training**
3. **Graph-based evidence aggregation**
4. **Counterfactual evidence augmentation**
5. **Adversarial claim debiasing**
6. **Multi-hop fact verification**

---

## 23. Deliverables Checklist

建议按以下完成情况推进项目：

- [ ] 读取并预处理数据
- [ ] 实现 `ShortcutModel`
- [ ] 实现 `GroundedModel`
- [ ] 实现 warm-up trainer
- [ ] 实现 routing score 计算
- [ ] 实现 routing cache 输出
- [ ] 实现 remix sampler
- [ ] 实现 remix trainer
- [ ] 实现 group-wise evaluation
- [ ] 实现 evidence sensitivity evaluation

---

## 24. Summary

EGR-FV 的核心不是简单增加一个 claim-only baseline，而是构造一个**显式可控的双分支训练框架**：

- 用 shortcut branch 识别偏置
- 用 grounded branch 学习真实验证语义
- 用 routing 找出 evidence-needed 样本
- 用 reassembled batches 改变训练梯度的主导方向
- 最终让 grounded branch 成为真正负责决策的主分支

从代码实现角度，建议你优先完成：

1. `GroundedModel`
2. `ShortcutModel`
3. `routing.py`
4. `remix_sampler.py`
5. `remix_trainer.py`

先跑通最小版本，再逐步加：

- weighted loss
- orthogonality loss
- fusion head
- curriculum remix
- robustness evaluation

---

## 25. Reference

### Paper

- Evidence-Grounded Remixing for Debiased Fact Verification  
- Reference paper URL: `http://arxiv.org/pdf/2506.11550`

### Code

- Reference repository: `https://github.com/MatthewMaxy/Remix_ICML2025`

> 注意：本 README 是**面向工程落地的实现设计文档**。其中部分 scoring、routing、loss 形式是为便于实现而做的工程化展开，真正落地时可根据实验结果继续调整。