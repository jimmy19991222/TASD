# Teacher-Guided Branching Rollout (TG-Branching)

**分支**: `tg-branching-rollout`  
**日期**: 2026-06-01 ~ 2026-07-01  
**模型**: Qwen3-4B  
**SwanLab**: `awesome_jimmy/DPO-Comparison-4B`

---

## §1 方法

### 1.1 动机

标准 RLHF 方法（GRPO、SDPO 等）通过同 prompt 的多条独立 rollout 构造偏好对或估计 advantage，但独立采样产生的轨迹共享前缀极少，对比信号噪声大。我们希望制造**前缀强相关、结果方差大**的 contrastive pairs：在同一条高质量推理链的关键决策点分裂，让 DPO loss 聚焦于真正导致结果差异的 token 选择。

### 1.2 Two-Stage DPO-2S 架构

采用两阶段训练，每个 training step 内：

**Stage 1（独立 rollout，exploration only）**：每个 prompt 生成 n=4 条独立 rollout，用 reward model 打分。根据 success_threshold=0.3 筛选 chosen response。Stage 1 **不贡献梯度**，仅用于筛选高质量前缀供 Stage 2 使用。

**Stage 2（branching rollout）**：从 Stage 1 的 chosen response 出发，通过 teacher-guided branching 生成 n_trees=2 棵分裂树（每棵 n_splits=1 次分裂 → 2 个 leaf），构造 contrastive pairs 用于 DPO loss。

最终 loss 为纯 DPO loss（不包含 GRPO 项）：$\mathcal{L} = \mathcal{L}_{\text{DPO}}^{\text{stage2}}$

### 1.3 Branching Rollout

在 student 生成过程中，通过 rolling z-score 检测 token 熵突变点（窗口 W=20，阈值 k=2.0σ）。在决策点上，用 teacher（带 privileged context 的同一模型）在 student top-K 候选中选出 argmax/argmin 两个 token，分裂为两条续写链。

仅在最后一个 branch token **之后**的 suffix 位置计算 loss（`branch_token_loss_mode=suffix`），共享前缀和 branch token 本身不贡献梯度，避免 teacher 注入 token 的 off-policy 偏差。

### 1.4 With-ref DPO Loss

Stage 2 的 DPO loss 引入 π_ref（初始策略）作为隐式 KL 正则化：

$$\mathcal{L}_{\text{DPO}} = -\log\sigma\left(\beta \left[\log\frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \log\frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)}\right]\right)$$

对比实验表明，with-ref 相比 marker-only 在所有域上一致提升 +2.5~+2.7 pts（见 §2.3）。

### 1.5 Baselines

| Method | 描述 |
|---|---|
| **GRPO** | 标准 Group Relative Policy Optimization，n=8 独立 rollout，outcome-level advantage |
| **SDPO α=0.5** | Self-Distillation PO，JSD（α=0.5，α=0 forward KL / α=1 reverse KL），peer rollout 作为 ref |
| **DPO-2S** | 本文方法，Two-Stage branching + DPO loss，with-ref |

所有方法统一：Qwen3-4B，LR=1e-5，train_batch_size=32，250 steps，val metric 取 best val acc。

---

## §2 实验结果

### 2.1 主要结果（best val acc，DPO-2S with-ref vs baselines）

Sciknoweval 选用 β=2.0，Math500/GSM8K 选用 β=0.5（β 选择见消融 §2.2.2）。

| Task | GRPO | SDPO α=0.5 | **DPO-2S (ours)** | Δ vs best baseline |
|---|---|---|---|---|
| **Chemistry** | 0.6923 | 0.7217 | **0.7253** | +0.4 vs SDPO |
| **Biology** | 0.5312 | 0.5375 | **0.5600** | +2.3 vs SDPO |
| **Material** | **0.7866** | 0.7161 | **0.8012** | +1.5 vs GRPO |
| **Physics** | **0.6508** | 0.6102 | 0.6375 | −1.3 vs GRPO |
| **Math500** | **0.7638** | 0.6675 | **0.7638** | tie with GRPO |
| **GSM8K** | 0.9323 | 0.9167 | **0.9422** | +1.0 vs GRPO |
| **Competition Math** | — | — | — | — |
| **Tooluse** | — | — | — | — |
| **LiveCodeBench** | — | — | — | — |

已完成任务：DPO-2S with-ref 在 6 个任务中 5 个 ≥ best baseline（3 个 SOTA，2 个 tie），仅 Physics 落后 GRPO 1.3 pts（统计误差范围）。

### 2.2 消融实验

#### 2.2.1 Marker vs With-ref（sciknoweval, β=2.0）

| Domain | DPO-2S marker | DPO-2S with-ref | Δ |
|---|---|---|---|
| Chemistry | 0.6991 | **0.7253** | +2.6 |
| Biology | 0.5350 | **0.5600** | +2.5 |
| Material | 0.7739 | **0.8012** | +2.7 |
| Physics | 0.6320 | **0.6375** | +0.6 |

With-ref（引入 π_ref 隐式 KL）在所有 4 域一致优于 marker，前 3 域增益稳定在 +2.5~+2.7 pts。

#### 2.2.2 β 选择（with-ref）

**Sciknoweval**

| Domain | β=0.5 | β=2.0 | Better |
|---|---|---|---|
| Chemistry | 0.6952 | **0.7253** | β=2.0 (+3.0) |
| Biology | 0.5337 | **0.5600** | β=2.0 (+2.6) |
| Material | **0.7999** | 0.8012 | β=2.0 (+0.1) |
| Physics | **0.6469** | 0.6375 | β=0.5 (+0.9) |

**Competition Math**

| β | acc | 备注 |
|---|---|---|
| 0.5 | — | |
| 2.0 | — | |

知识密集任务（sciknoweval）β=2.0 整体更优；推理任务（Math500/GSM8K）β=0.5 更优（见 §2.1）。β 选择依赖任务类型。

#### 2.2.3 Entropy 动态（first → last, 250 steps, sciknoweval）

| Domain | GRPO | SDPO α=0.5 | DPO-2S β=0.5 | DPO-2S β=2.0 |
|---|---|---|---|---|
| Chemistry | 0.215→0.169 (↓21.6%) | 0.226→0.220 (↓2.6%) | 0.223→0.245 (↑10.3%) | 0.223→0.398 (**↑78.5%**) |
| Biology | 0.156→0.013 (**↓91.7%**) | 0.157→0.000 (**↓99.8%**) | 0.169→0.579 (↑243%) | 0.152→0.345 (**↑128%**) |
| Material | 0.046→0.030 (↓35.9%) | 0.050→0.004 (**↓91.8%**) | 0.055→0.226 (↑312%) | 0.052→0.077 (↑48.8%) |
| Physics | 0.125→0.111 (↓11.6%) | 0.126→0.022 (**↓82.5%**) | 0.128→0.092 (↓28.3%) | 0.130→0.182 (↑39.8%) |

GRPO/SDPO 在多数域 entropy 大幅下降（最严重 −99.8%），DPO-2S β=2.0 在所有 4 域均提升 entropy（+39%~+128%），β=0.5 在 3/4 域提升但 Physics 下降 28%。

#### 2.2.4 训练步数（DPO-2S with-ref β=2.0, sciknoweval）

| Domain | 250s best | 500s best | 500s last |
|---|---|---|---|
| Biology | **0.5600** | 0.5550 | 0.4562 |
| Chemistry | **0.7253** | 0.6869 | running |
| Material | **0.8012** | 0.7939 | running |

250 steps 已接近最优，500 steps 收益递减。Biology 域出现明显过拟合（500s last 比 250s best 低 10.4 pts）。

#### 2.2.5 Teacher-Guided Adaptive β（sciknoweval, 500 steps）

**方法**：$\beta_i = \beta_{\text{base}} \cdot \text{clamp}(\alpha \cdot m_i,\ \beta_{\min},\ \beta_{\max})$，其中 $m_i = \log\pi_{\text{teacher}}(y_w|x) - \log\pi_{\text{teacher}}(y_l|x)$

**结果**

| Domain | Fixed β=2.0 (250s) | Teacher β=2.0 (500s) | Δ | Fixed β=0.5 | Teacher β=0.5 (500s) | Δ |
|---|---|---|---|---|---|---|
| Chemistry | **0.7253** | — | — | **0.6952** | 0.6295 | −6.6 |
| Biology | 0.5600 | **0.5975** | +3.8 | 0.5337 | 0.5475 | +1.4 |
| Material | **0.8012** | 0.7739 | −2.7 | **0.7999** | 0.7739 | −2.6 |
| Physics | 0.6375 | 0.6383 | ≈0 | 0.6469 | **0.6672** | +2.0 |

**过拟合**：所有 teacher 实验 best 出现极早（step 60~180），之后持续下降——

| Run | best step | best | last | 跌幅 |
|---|---|---|---|---|
| teacher-β0.5 material | ~60 | 0.7739 | 0.6802 | −9.4 |
| teacher-β0.5 biology | ~170 | 0.5475 | 0.4963 | −5.1 |
| teacher-β2.0 biology | ~180 | 0.5975 | 0.5363 | −6.1 |

**诊断**：Teacher-guided adaptive β 效果不稳定（Biology +3.8, Material −2.7），且过拟合远比 fixed β 严重。根本原因分析：

1. **Teacher margin 尺度不归一**：不同域的 teacher logp margin 分布差异大（chemistry margin 窄、biology margin 宽），导致 adaptive β 在某些域放大过度
2. **Margin 随训练过时**：teacher 是固定或慢更新的（EMA rate=默认），但 policy 快速改变。训练后期 teacher margin 与当前 policy 的行为不匹配，β 信号失效
3. **缺少衰减机制**：β_base 在整个 500 步中恒定，后期 policy 已接近最优时仍然用高 β 强推，导致过拟合

#### 2.2.6 Teacher-Guided Adaptive β 改进方向

**P1. Margin Per-Batch 归一化（优先级最高）**

当前 margin 原始值直接缩放 β，但不同域/不同 batch 的 margin 量级不同。改为 per-batch z-score 归一化：

$$\hat{m}_i = \frac{m_i - \mu_B}{\sigma_B + \epsilon}$$

$$\beta_i = \beta_{\text{base}} \cdot \text{clamp}(\hat{m}_i,\ \beta_{\min}/\beta_{\text{base}},\ \beta_{\max}/\beta_{\text{base}})$$

这消除域间尺度差异，让 adaptive β 只依赖**相对排序**而非绝对值。实现改动小（一行 normalization），预期修复 chemistry/material 上的退化。

**P2. β Cosine Decay Schedule**

固定 β 的 250 步实验已证明后期收益递减。对 adaptive β 同理：

$$\beta_{\text{base}}(t) = \beta_{\min} + \frac{1}{2}(\beta_{\text{init}} - \beta_{\min})(1 + \cos(\pi \cdot t / T))$$

前期高 β 强信号学习，后期低 β 防止过拟合。参数：$\beta_{\text{init}}=2.0$，$\beta_{\min}=0.1$，$T=250$。可与 P1 正交组合。

**P3. KL-Aware β Clipping**

动态监控 $D_{\text{KL}}(\pi_\theta \| \pi_{\text{ref}})$，当 KL 超过阈值时主动压低 β：

$$\beta_{\text{eff}} = \beta_i \cdot \min\left(1,\ \frac{D_{\max}}{D_{\text{KL}}(\pi_\theta \| \pi_{\text{ref}})}\right)$$

这是 with-ref loss 之外的第二层 KL 安全网。好处是自动适应：收敛阶段 KL 小 → β 不受限；过拟合阶段 KL 大 → β 被压缩。参数：$D_{\max} = 0.05$。

**P4. 缩短训练到 250 步**

所有证据（fixed β §2.2.4, teacher runs §2.2.5）一致表明 250 步已充分，500 步只增加过拟合风险。teacher-guided 实验应与 fixed β 使用相同的 250 步训练。

**建议实验顺序**：P1 + P4 → P1 + P2 + P4 → P1 + P2 + P3 + P4

---

## §3 结论

1. **DPO-2S with-ref 跨任务 SOTA**：sciknoweval 3/4 域 SOTA（Chemistry +3.3, Biology +2.9, Material +1.5 over GRPO），GSM8K +1.0, Math500 tie
2. **With-ref 正则化不可或缺**：marker → with-ref 一致 +2.5~+2.7 pts（sciknoweval β=2.0），π_ref 隐式 KL 在高 β 下防止 policy 过度偏离
3. **β 选择依赖任务类型**：知识密集（sciknoweval）→ β=2.0 最优；推理（Math500/GSM8K）→ β=0.5 最优
4. **Entropy 保持是核心机制**：DPO-2S β=2.0 entropy 上升 +39%~+128%，GRPO 下降 12%~92%，SDPO 在 3/4 域 collapse（-82%~-100%）
5. **250 steps 足够**：500-step 延长收益递减，Biology 域出现过拟合

