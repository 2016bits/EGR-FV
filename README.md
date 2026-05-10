# EGR-FV v2 主实验与消融实验设计说明

本文档用于说明当前重新设计后的 EGR-FV v2 主实验与相应消融实验方案。设计目标是解决原实验中“消融实验效果高于主实验、实验逻辑不够合理”的问题，并让主实验更好地对应事实核查去偏任务的核心目标：**提升模型对证据的依赖，而不是仅仅提升普通分类准确率**。

---

## 1. 背景与问题

原实验中，主实验的总体结果为：

```text
avg_acc      = 83.93
avg_macro_f1 = 83.90
```

但部分消融实验结果反而更高，例如：

```text
routing_only:  avg_acc = 87.29, avg_f1 = 86.45
two_branches:  avg_acc = 86.74, avg_f1 = 85.91
```

这会导致主实验叙事不够稳定，因为审稿人可能会质疑：

```text
如果只做 routing-only 或 two-branches joint 就已经更好，为什么还需要完整方法？
```

因此，新的实验设计不再只围绕 clean Acc / Macro-F1 展开，而是围绕以下目标重新组织：

1. 主实验是否提升普通事实核查性能；
2. 主实验是否更依赖 evidence；
3. 主实验是否在 evidence-needed 样本上表现更好；
4. routing、remix、grounded-dominant loss、evidence contrast 是否分别有贡献。

---

## 2. 新主实验：Full EGR-FV v2

新的主实验命名为：

```text
Full EGR-FV v2
```

完整方法由以下模块组成：

```text
Full EGR-FV v2 =
  grounded branch warm-up
+ shortcut branch warm-up
+ out-of-fold routing
+ soft evidence necessity score
+ distribution-preserving remix
+ grounded-dominant objective
+ evidence contrastive loss
+ grounded-only inference
```

核心思想是：

```text
先分别训练 grounded branch 和 shortcut branch
→ 用二者的行为差异估计每个样本对 evidence 的需求程度
→ 根据 evidence necessity score 调整 loss 权重和 batch 构造
→ 最终训练一个以 grounded branch 为主导的模型
→ 推理阶段只使用 grounded branch
```

---

## 3. 主实验训练流程

主实验分为五个阶段。

---

### Stage 0: Grounded Branch Warm-up

首先训练一个标准的 claim-evidence fact verification model。

输入：

```text
claim + evidence
```

输出：

```text
SUPPORTS / REFUTES / NEI
```

损失函数：

```text
L_g = CE(p_g(y | claim, evidence), y)
```

该阶段的目的有两个：

1. 让 grounded branch 学会基础的证据理解能力；
2. 为后续 routing 阶段提供 grounded confidence 和 grounded prediction。

---

### Stage 1: Shortcut Branch Warm-up

接着训练一个 claim-only shortcut branch。

输入：

```text
claim
```

不输入 evidence。

损失函数：

```text
L_s = CE(p_s(y | claim), y)
```

该阶段用于显式建模数据集中的 shortcut pattern。例如：

```text
某些 claim 表述本身高度暗示标签
某些实体、模板或词汇与标签强相关
某些样本即使不看 evidence 也容易被正确分类
```

shortcut branch 不用于最终推理，而是用于识别哪些样本属于 bias-easy，哪些样本更需要 evidence。

---

### Stage 2: Out-of-Fold Routing Estimation

这一阶段用于给每个训练样本计算 evidence necessity score。

不建议直接用在完整训练集上训练好的模型给训练集打分，因为这可能导致 routing 结果过拟合训练集。

推荐使用 K-fold out-of-fold routing。

例如使用 5-fold：

```text
fold 1: train on D2+D3+D4+D5, score D1
fold 2: train on D1+D3+D4+D5, score D2
fold 3: train on D1+D2+D4+D5, score D3
fold 4: train on D1+D2+D3+D5, score D4
fold 5: train on D1+D2+D3+D4, score D5
```

最终，每个样本都由“没有见过它的模型”产生 routing 分数。

对每个样本保存以下信息：

```text
shortcut_conf_i
shortcut_pred_i
shortcut_correct_i

grounded_conf_i
grounded_pred_i
grounded_correct_i

disagreement_i
necessity_score_i
```

其中：

```text
shortcut_conf_i = max p_s(y | claim_i)
grounded_conf_i = max p_g(y | claim_i, evidence_i)
```

`disagreement_i` 可以使用预测是否不同，也可以使用分布差异，例如 KL divergence。

---

### Stage 3: Routing Policy Construction

Stage 2 得到的是原始 routing 信息。Stage 3 将这些信息转化为训练时真正使用的控制信号：

```text
group label
loss weight
sampling priority
shortcut loss gate
```

#### 3.1 Evidence Necessity Score

对每个样本定义连续分数：

```text
r_i ∈ [0, 1]
```

`r_i` 越大，表示该样本越需要 evidence。

一个简单规则如下：

```text
if shortcut_wrong and grounded_correct:
    r_i = 1.0
elif shortcut_conf high and shortcut_correct:
    r_i = 0.0
elif shortcut_pred != grounded_pred:
    r_i = 0.7
else:
    r_i = 0.5
```

也可以使用连续加权形式：

```text
r_i =
  0.4 * grounded_correct_i
+ 0.3 * shortcut_wrong_i
+ 0.2 * disagreement_i
+ 0.1 * evidence_length_score_i
```

推荐主实验使用连续版本，避免 hard routing 引入过强噪声。

#### 3.2 样本分组

根据 `r_i` 将样本分为三类：

```text
if r_i >= 0.7:
    group_i = grounded_needed
elif r_i <= 0.3:
    group_i = bias_easy
else:
    group_i = hard
```

三类样本含义如下：

```text
grounded_needed:
  shortcut branch 容易错，但 evidence 有帮助，应重点训练 grounded branch

bias_easy:
  claim-only 即可预测，容易诱导模型使用 shortcut，应降低其训练影响

hard:
  两个分支都不稳定，正常训练
```

#### 3.3 Grounded Loss Weight

对 grounded branch 使用 evidence-necessity-aware 权重：

```text
w_g(i) = 1 + α * r_i
```

例如 `α = 0.5` 时：

```text
r_i = 0.0 → w_g = 1.0
r_i = 0.5 → w_g = 1.25
r_i = 1.0 → w_g = 1.5
```

#### 3.4 Shortcut Loss Gate

对 shortcut branch 使用 gated loss：

```text
w_s(i) = β * (1 - r_i)
```

也就是说：

```text
样本越需要 evidence，shortcut branch 的 loss 权重越小
样本越 bias-easy，shortcut branch 的 loss 权重越大
```

这样可以避免 shortcut branch 在 grounded-needed 样本上与 grounded branch 竞争。

---

### Stage 4: Full EGR Training

最终训练阶段使用 routing-guided distribution-preserving remix 和 grounded-dominant objective。

#### 4.1 Batch 构造

推荐 batch 构造比例：

```text
50% original random samples
30% grounded_needed samples
15% hard samples
5% bias_easy samples
```

例如 batch size = 16 时：

```text
8 个样本：原始随机采样
5 个样本：grounded_needed
2 个样本：hard
1 个样本：bias_easy
```

该设计比 homogeneous remix 更稳定，因为每个 batch 仍然保留一半原始训练分布。

#### 4.2 主训练损失

完整损失函数为：

```text
L_full =
    L_g_weighted
  + λ_s L_s_gated
  + λ_orth L_orth
  + λ_ctr L_evidence_contrast
```

其中：

```text
L_g_weighted = w_g(i) * CE(p_g(y | claim, evidence), y)
L_s_gated   = w_s(i) * CE(p_s(y | claim), y)
```

`L_orth` 用于约束 grounded representation 和 shortcut representation 尽量解耦：

```text
L_orth = || H_g^T H_s ||^2
```

#### 4.3 Evidence Contrastive Loss

为了显式增强模型对 evidence 的依赖，构造三种输入：

```text
x_pos  = claim + gold evidence
x_neg  = claim + shuffled / mismatched evidence
x_null = claim + empty evidence
```

要求模型在正确证据下比错误证据下更容易预测正确标签：

```text
L_ctr = max(0, m + CE(p_pos, y) - CE(p_neg, y))
```

也可以加入 no-evidence sensitivity 项：

```text
L_null = KL(p_pos || p_null)
```

最终：

```text
L_evidence_contrast = L_ctr + γ L_null
```

---

### Stage 5: Inference

推理阶段只使用 grounded branch：

```text
ŷ = argmax p_g(y | claim, evidence)
```

shortcut branch 只用于训练阶段的 routing 和辅助约束，不参与最终预测。

这样可以避免最终模型重新退化为 shortcut-based classifier。

---

## 4. 推荐主实验配置

```yaml
experiment:
  name: full_egr_fv_v2

warmup:
  grounded_epochs: 2
  shortcut_epochs: 2
  learning_rate: 2.0e-5

routing:
  strategy: out_of_fold
  num_folds: 5
  score_type: soft_evidence_necessity
  grounded_needed_threshold: 0.7
  bias_easy_threshold: 0.3

remix:
  type: distribution_preserving
  original_ratio: 0.50
  grounded_needed_ratio: 0.30
  hard_ratio: 0.15
  bias_easy_ratio: 0.05

loss:
  alpha_grounded_weight: 0.5
  beta_shortcut_gate: 1.0
  lambda_shortcut: 0.1
  lambda_orth: 0.03
  lambda_contrast: 0.2
  contrast_margin: 0.3
  null_kl_gamma: 0.1

training:
  epochs: 5
  batch_size: 16
  learning_rate: 2.0e-5
  warmup_ratio: 0.1
  max_grad_norm: 1.0

inference:
  branch: grounded_only
```

---

## 5. 消融实验设计原则

新的消融实验围绕以下问题设计：

```text
Full EGR-FV v2 的提升来自哪里？
```

具体拆解为：

1. 是否只是 grounded baseline 本身有效；
2. 是否只是多了 shortcut branch；
3. routing 是否有效；
4. remix 是否有效；
5. soft routing 是否优于 hard routing；
6. out-of-fold routing 是否优于 in-sample routing；
7. grounded-dominant loss 是否必要；
8. evidence contrastive loss 是否真的增强证据依赖；
9. grounded-only inference 是否比 fusion inference 更符合去偏目标。

---

## 6. 推荐消融实验列表

### B0. Grounded-only

基础 claim-evidence fact verification model。

```text
输入：claim + evidence
训练：CE loss
推理：grounded branch
```

损失函数：

```text
L = CE(p_g(y | claim, evidence), y)
```

目的：验证 Full 是否优于普通 evidence-aware 模型。

---

### B1. Two-branches Joint

保留 grounded branch 和 shortcut branch，但不使用 routing、remix 和 contrast。

```text
grounded branch: claim + evidence
shortcut branch: claim only
```

损失函数：

```text
L = CE(p_g, y) + λ_s CE(p_s, y)
```

推理仍然只使用 grounded branch。

目的：排除 Full 的提升只是因为多了一个 shortcut branch。

---

### B2. Routing-only

使用 out-of-fold routing 和 grounded-dominant loss，但不做 remix，不加 evidence contrast。

```text
有：
  out-of-fold routing
  soft evidence necessity score
  grounded weighted loss
  shortcut gated loss

无：
  batch remix
  evidence contrast
```

损失函数：

```text
L = w_g(i) L_g + λ_s w_s(i) L_s
```

目的：验证 sample-level routing weight 本身是否有效。

---

### B3. Random Remix-only

不使用 learned routing，只做随机 remix。

```text
无 routing
无 evidence necessity score
无 group weight
无 evidence contrast
```

将训练集随机分成若干组，并按照类似 Full 的 batch 比例进行采样。

目的：验证 Full 的提升是否只是来自 batch composition 改变。

---

### B4. Length Remix-only

不使用 learned routing，只根据 claim/evidence 长度进行启发式分组。

推荐规则：

```text
if evidence_len >= 90 or evidence_len >= max(3 * claim_len, 40):
    group = grounded_needed
elif evidence_len <= 25 and claim_len <= 32:
    group = bias_easy
else:
    group = hard
```

注意：正式消融中不能使用 label 信息进行分组。

不推荐：

```python
if label == "NEI":
    group = grounded_needed
```

因为这会变成 label-aware sampler，不是公平消融。

目的：验证 learned routing 是否优于简单启发式规则。

---

### B5. Full w/o Remix

保留 Full 的其他模块，只去掉 distribution-preserving remix。

```text
保留：
  out-of-fold routing
  soft evidence necessity score
  grounded-dominant loss
  evidence contrastive loss

去掉：
  distribution-preserving remix
```

训练 batch 使用普通 random shuffle。

目的：在公平条件下验证 remix 的独立贡献。

---

### B6. Full w/o Routing

保留 remix 框架和 contrast loss，但不使用 learned routing。

可选实现：

```text
A. random routing
B. length-only routing
```

主表建议使用 random routing，appendix 可以加入 length-only routing。

目的：验证 Full 的提升是否依赖 learned routing，而不是任意采样策略。

---

### B7. Full w/o Evidence Contrast

去掉 evidence contrastive loss，其余保持与 Full 一致。

```text
Full:
  L = L_g_weighted + λ_s L_s_gated + λ_orth L_orth + λ_ctr L_contrast

B7:
  L = L_g_weighted + λ_s L_s_gated + λ_orth L_orth
```

目的：验证 evidence contrast 是否真的增强模型对 evidence 的依赖。

预期：B7 的 clean Acc / Macro-F1 可能接近 Full，但 Full 应在证据扰动指标上更好。

---

### B8. Full w/o Grounded-dominant Objective

保留 routing、remix 和 contrast，但把 loss 改为普通 two-branch joint loss。

```text
Full:
  L = w_g(i)L_g + λ_s w_s(i)L_s + λ_orth L_orth + λ_ctr L_contrast

B8:
  L = L_g + λ_s L_s + λ_orth L_orth + λ_ctr L_contrast
```

也就是说：

```text
不使用 grounded loss weighting
不使用 shortcut loss gate
```

目的：验证 grounded-dominant training objective 是否必要。

---

### B9. Full w/ Hard Routing

将 Full 中的 continuous soft score 替换成 hard group。

```text
bias_easy        → r_i = 0
grounded_needed → r_i = 1
hard             → r_i = 0.5
```

或者使用固定权重：

```text
bias_easy:        w_g = 0.5, w_s = 1.0
hard:             w_g = 1.0, w_s = 0.5
grounded_needed:  w_g = 1.5, w_s = 0.0
```

目的：验证 soft routing 是否比 hard routing 更稳定。

---

### B10. Full w/ In-sample Routing

保留 routing 机制，但不使用 out-of-fold。

```text
Full:
  out-of-fold routing

B10:
  in-sample routing
```

即直接用在完整训练集上 warm-up 的模型对训练集打分。

目的：验证 out-of-fold routing 是否能够减少 routing overfitting。

---

### B11. Full w/ Fusion Inference

训练过程与 Full 一样，但推理时融合 grounded branch 和 shortcut branch。

例如：

```text
p = λ p_g + (1 - λ) p_s
```

目的：验证为什么最终推理阶段选择 grounded-only inference。

预期：fusion inference 可能 clean Acc 更高，但 evidence sensitivity 更差。

---

### B12. Shortcut-only Inference

只使用 shortcut branch 进行推理。

```text
输入：claim only
输出：p_s(y | claim)
```

目的：诊断数据集本身的 shortcut 强度。

该实验不一定放主表，可以作为分析实验。

---

## 7. 推荐主消融表

如果篇幅有限，建议主表包含以下实验：

```text
B0  Grounded-only
B1  Two-branches Joint
B2  Routing-only
B3  Random Remix-only
B5  Full w/o Remix
B7  Full w/o Evidence Contrast
B8  Full w/o Grounded-dominant Objective
Full EGR-FV v2
```

对应表格：

| Method                               |  Acc | Macro-F1 | Grounded-needed F1 | ΔRemove | ΔShuffle | Claim-only Gap |
| ------------------------------------ | ---: | -------: | -----------------: | ------: | -------: | -------------: |
| Grounded-only                        |      |          |                    |         |          |                |
| Two-branches Joint                   |      |          |                    |         |          |                |
| Routing-only                         |      |          |                    |         |          |                |
| Random Remix-only                    |      |          |                    |         |          |                |
| Full w/o Remix                       |      |          |                    |         |          |                |
| Full w/o Evidence Contrast           |      |          |                    |         |          |                |
| Full w/o Grounded-dominant Objective |      |          |                    |         |          |                |
| Full EGR-FV v2                       |      |          |                    |         |          |                |

---

## 8. 推荐结果汇报方式

不建议只汇报：

```text
Acc
Macro-F1
```

因为去偏事实核查任务的核心不是普通分类性能，而是 evidence dependence。

建议至少汇报以下指标。

---

### 8.1 Clean Acc / Macro-F1

标准测试集上的分类性能：

```text
Acc_clean
Macro-F1_clean
```

用于证明 Full 不牺牲基础事实核查能力。

---

### 8.2 Grounded-needed F1

只在 routing 判定为 grounded_needed 的测试子集上计算 Macro-F1。

```text
F1_grounded_needed
```

该指标用于回答：

```text
模型是否真的提升了需要 evidence 的样本？
```

---

### 8.3 No-evidence Evaluation

将 evidence 移除，只保留 claim：

```text
claim + empty evidence
```

得到：

```text
F1_no_evidence
```

定义：

```text
ΔRemove = F1_clean - F1_no_evidence
```

如果模型真的依赖 evidence，移除 evidence 后性能应该明显下降，因此 `ΔRemove` 应该更大。

---

### 8.4 Shuffled-evidence Evaluation

将 evidence 替换为其他样本的 evidence：

```text
claim + shuffled evidence
```

得到：

```text
F1_shuffled_evidence
```

定义：

```text
ΔShuffle = F1_clean - F1_shuffled_evidence
```

如果模型真的依赖正确 evidence，那么换成错误 evidence 后性能应该下降，因此 `ΔShuffle` 应该更大。

---

### 8.5 Claim-only Gap

定义：

```text
Claim-only Gap = F1_grounded_input - F1_claim_only_input
```

该指标越大，说明模型越依赖 evidence，而不是只依赖 claim shortcut。

---

## 9. 推荐三张实验表

### Table 1: Main Results

展示整体性能：

```text
Grounded-only
Two-branches Joint
Routing-only
Random Remix-only
Full EGR-FV v2
```

指标：

```text
Acc
Macro-F1
```

---

### Table 2: Module Ablation

展示各模块贡献：

```text
Full EGR-FV v2
Full w/o Remix
Full w/o Evidence Contrast
Full w/o Grounded-dominant Objective
Full w/ Hard Routing
Full w/ In-sample Routing
```

指标：

```text
Acc
Macro-F1
Grounded-needed F1
```

---

### Table 3: Evidence Dependence Analysis

展示模型是否真正依赖 evidence：

```text
Grounded-only
Two-branches Joint
Routing-only
Full w/o Evidence Contrast
Full EGR-FV v2
```

指标：

```text
Clean F1
No-evidence F1
Shuffled-evidence F1
ΔRemove
ΔShuffle
Claim-only Gap
```

---

## 10. 合理的预期结果模式

理想情况下，结果应呈现以下趋势。

### 10.1 Clean Acc / Macro-F1

```text
Full EGR-FV v2 ≥ Grounded-only
Full EGR-FV v2 ≥ Two-branches Joint
Full EGR-FV v2 ≥ Random Remix-only
Full EGR-FV v2 ≈ or ≥ Routing-only
```

如果 Routing-only 的 clean Acc 略高于 Full，也不是致命问题。关键是 Full 应该在 evidence-dependence 指标上更好。

---

### 10.2 Grounded-needed F1

Full 应该优于：

```text
Grounded-only
Two-branches Joint
Routing-only
Full w/o Remix
Full w/o Grounded-dominant Objective
```

这说明 Full 确实提升了需要 evidence 的样本。

---

### 10.3 Evidence Sensitivity

Full 应该具有更高的：

```text
ΔRemove
ΔShuffle
Claim-only Gap
```

这说明模型不是只靠 claim shortcut，而是真正使用 evidence。

---

## 11. 实验公平性要求

为了避免消融实验不公平，所有实验应尽量保持以下设置一致：

```text
same train/dev/test split
same backbone
same optimizer
same learning rate
same batch size
same final training epochs
same random seeds
same inference branch, unless doing inference ablation
```

特别注意以下几点。

---

### 11.1 Remix-only 不应使用 label 信息

不建议在正式消融中使用：

```python
if label == "NEI":
    group = grounded_needed
```

这会让消融变成 label-aware sampler，可能人为提升 macro-F1，尤其是 NEI 类。

正式消融中只允许使用：

```text
random grouping
length-only grouping
model-free metadata grouping
```

---

### 11.2 除 inference ablation 外，推理方式必须一致

所有主要方法都应使用：

```text
grounded-only inference
```

否则 fusion 或 shortcut branch 可能在 clean Acc 上占便宜，但这不符合 evidence-grounded debiasing 的目标。

---

### 11.3 Full 增加了 contrast loss，因此必须有 w/o contrast

Evidence contrastive loss 本质上引入了额外训练信号，因此必须设置：

```text
Full w/o Evidence Contrast
```

否则无法判断提升来自 routing/remix，还是来自 contrastive data augmentation。

---

## 12. 论文中可使用的简短描述

英文版本：

```text
The full model is trained in three stages. First, we independently warm up a grounded branch on claim-evidence pairs and a shortcut branch on claim-only inputs. Second, we perform out-of-fold routing to estimate an evidence-necessity score for each training instance based on the disagreement and correctness patterns between the two branches. Third, we train the final model with routing-guided distribution-preserving remixing and a grounded-dominant objective, where evidence-needed samples receive larger grounded loss weights and smaller shortcut loss weights. We further introduce an evidence contrastive loss by comparing gold evidence, shuffled evidence, and empty evidence inputs. At inference time, only the grounded branch is used.
```

中文版本：

```text
完整模型分三阶段训练。首先分别预训练 grounded branch 和 shortcut branch。其次，采用 out-of-fold routing 为每个训练样本估计 evidence necessity score。最后，基于该分数进行 distribution-preserving batch remix，并使用 grounded-dominant objective 训练最终模型。对于 evidence-needed 样本，提高 grounded branch 的 loss 权重，同时降低 shortcut branch 的 loss 权重。此外，通过 gold evidence、shuffled evidence 和 empty evidence 构造 evidence contrastive loss，使 grounded branch 更依赖证据。推理阶段仅使用 grounded branch。
```

---

## 13. 最小可执行实验版本

如果当前时间有限，建议优先实现以下 6 个实验：

```text
1. Grounded-only
2. Two-branches Joint
3. Routing-only
4. Full w/o Remix
5. Full w/o Evidence Contrast
6. Full EGR-FV v2
```

这 6 个实验已经能够支撑主要结论：

```text
Grounded-only:
  提供基础事实核查模型对照

Two-branches Joint:
  排除“只是多了 shortcut branch”的解释

Routing-only:
  验证 routing weight 的作用

Full w/o Remix:
  验证 remix 的作用

Full w/o Evidence Contrast:
  验证 evidence contrast 的作用

Full EGR-FV v2:
  完整方法
```

---

## 14. 总结

新的实验设计不再把目标简单定义为：

```text
Full 必须在 clean Acc / Macro-F1 上高于所有消融
```

而是定义为：

```text
Full 应在保持 clean performance 的同时，显著提升 evidence dependence、grounded-needed group performance 和 robustness under evidence perturbation。
```

因此，最终判断 Full 是否合理，应同时观察：

```text
Acc_clean
Macro-F1_clean
F1_grounded_needed
ΔRemove
ΔShuffle
Claim-only Gap
```

只要 Full 在 clean performance 上不显著退化，并且在 evidence-dependence 指标上明显优于消融实验，就能更有力地证明 EGR-FV v2 的去偏价值。

---

## 15. Linux 环境运行命令

推荐使用总控脚本：

```bash
sh run_scripts/run_egrfv_v2.sh main
```

常用命令如下：

```bash
# 只跑主实验：warm-up + routing + Full EGR-FV v2 + eval
sh run_scripts/run_egrfv_v2.sh main

# 跑最小 6 个实验：Grounded-only / Two-branches / Routing-only / w/o Remix / w/o Contrast / Full
sh run_scripts/run_egrfv_v2.sh ablation-min

# 跑完整消融实验矩阵
sh run_scripts/run_egrfv_v2.sh ablation-all

# 只重新评估完整实验矩阵
sh run_scripts/run_egrfv_v2.sh eval-all
```

指定 GPU 或 Python 环境：

```bash
CUDA_VISIBLE_DEVICES=0 PYTHON_BIN=/path/to/python sh run_scripts/run_egrfv_v2.sh ablation-min
```

结果汇总文件：

```text
outputs/HOVER/predictions/ablation_summary.csv
```
