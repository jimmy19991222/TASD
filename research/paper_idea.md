# TCCA — Token-level Causal Credit Assignment

> **Building on OPD literature** (see [OPD_Deep_Analysis](file:///Users/awesome_jimmy/lazada/papers/raw/opd_papers/OPD_Deep_Analysis.html)): MiniLLM (2306) → GKD (2306) → SDPO/OPSD (2604) → RLSD (2604) → **TCCA (ours)**
> **目的**：让你 check 我们对论文思路的理解是否一致
> **更新**：2026-05-16 19:30

---

## 1. 论文 one-sentence 定位

> **TCCA：把"token 的 credit 应该多少"这个问题，用 teacher 真实改写 + reward 复算的因果反事实方式来回答。**

不是 "teacher 在哪里感到惊讶"（Prior-Shift，相关性），
不是 "teacher 和 student 哪里不合"（RLSD，相关性 + log-ratio 启发式），
不是 "哪类 token 启发式上重要"（TIP，entropy×divergence 经验加权），
而是 **"如果在 t 处听 teacher 一会儿，最终 reward 真的涨了吗？"**——直接做实验，看 ΔR_t。

---

## 2. OPD 系列 → TCCA 的一步推进

| 阶段 | 代表方法 | 解决了什么 | 留下什么问题 |
|---|---|---|---|
| 1 | MiniLLM / GKD | 解决 off-policy 蒸馏的 exposure bias，引入 on-policy student rollout | token-level credit 还是均匀 KL，无差异化 |
| 2 | TIP / SCOPE / SelecTKD | 用启发式 (student entropy, teacher divergence) 选 token 加权 | **相关性启发式**，没有因果证据 |
| 3 | OPSD / SDPO | self-distillation 范式，用 privileged-context teacher 做信号源 | RLSD Theorem 1: 不可消除的 MI leakage → progressive degradation |
| 4 | RLSD | "方向 ⊥ 大小" 解耦：方向锚到 env reward，大小用 evidence ratio (P_T/P_S) | evidence ratio 仍是**统计学**信号，不是真实因果 |
| 5 | SRPO | failed→SDPO / correct→GRPO 的 sample routing | 用 distribution match 做失败样本，没改 token-level credit |
| **6 (ours)** | **TCCA** | **用真实 counterfactual ΔR_t 做 token credit；保留 RLSD 的方向⊥大小原则，但用因果代替统计学** | (待论文 finalize) |

**TCCA 的 one more step**：把 RLSD 的"用 P_T/P_S 估计 token 重要性"升级为"做 K 次真实 intervention，用真 ΔR_t 测量 token 重要性"。

---

## 3. 核心创新点（论文 contribution）

### 3.1 Token-level causal credit signal（新概念）

**定义**：对 token y_t 的 causal credit
```
c_t := ΔR_t = R(y') - R(y),  其中  y' = y_<t + teacher_replacement + y_>t
```

**性质**（论文 Theorem 1 待证）：
- **因果性**：c_t 是 token y_t 的 individual causal effect on outcome reward（满足 Rubin causal model 的 SUTVA）
- **稀疏可计算**：只需在 top-K positions 计算 K 次 intervention，不是 O(T) full sweep
- **base-agnostic**：c_t 是数据，可叠加在任何 base RL 算法上

### 3.2 TCCA advantage 公式（新方法）

```
A_t (token-level) = base_seq_advantage · base_reweight · (1 + λ_token · c_t_clipped) · length_scale
                    ┌─────────────────┘  ┌─────────────────┘  ┌─────────────────────┘
                    GRPO/RLSD/SDPO    base 各自的            TCCA 核心 (1 + λ · ΔR_t)
                    seq 级 outcome    token 级 reweight
                    advantage
```

**3 项的作用**：
1. **base seq advantage**：方向锚（reliable，sparse）—— 沿用 RLSD 的"方向必须可靠"
2. **base token reweight**：base 自身的 token 级权重（GRPO uniform / RLSD log-ratio）
3. **(1 + λ · c_t)**：TCCA 的因果调制层 —— 真实 ΔR_t 在 intervention 位置加强/削弱权重

### 3.3 Top-K intervention 设计（工程贡献）

**问题**：直接对每个 token 都做 intervention 太贵 (O(T·n) per batch)
**TCCA 解法**：
- 只对 **failed samples** (R < threshold) 做 intervention（OPSD-style routing，借自 SRPO）
- 每个 failed sample 只在 **top-K divergence positions** 上 intervention
- 实际成本：O(K · n_failed) per batch，K=3 + 50% failed → +1.5× compute

### 3.4 Per-token causal credit construction（新机制）

每次 intervention 产生 1 个 ΔR_k，但**写入两个地方**：

```
失败 sample y: c_t[t_k..t_k+intervention_length) = -ΔR_k   ← 这些 token 是"错"，给负权重
composite y'_k: c_t[t_k..t_k+intervention_length) = +ΔR_k   ← 这些 token 是 teacher 的"对"，给正权重
```

**这一对正负 credit 形成 contrastive pair**（同 prefix，分歧 token 反向 credit）→ 类似 DPO 的对比信号，但用真实 outcome 差驱动。

---

## 4. 完整 pipeline：从 rollout 到 loss

### Step 1 — Standard student rollout

输入：prompt x；student π_θ；group_size n (我们用 n=7)
输出：n 个 student rollouts {y_1, ..., y_n}，每个有 outcome reward R_i

```
for i in 1..n:
    y_i ~ π_θ(·|x)
    R_i = reward_fn(x, y_i)
```

### Step 2 — Teacher forward & divergence 计算

teacher π_T（用 EMA，r=0.05）在 student rollouts 上 forward，得到 token-level log-probs：

```
for each (x, y_i):
    logp_T[i, t] = log π_T(y_{i,t} | x, y_{i,<t})
    logp_S[i, t] = log π_θ(y_{i,t} | x, y_{i,<t))    # 等于 old_log_probs
    divergence[i, t] = |logp_T[i, t] - logp_S[i, t]|
```

### Step 3 — 失败样本检测 + Top-K position selection

```
seq_reward[i] = R_i
failed_mask = seq_reward < failed_threshold  (e.g., 0.5)

For each failed sample i:
    top_k_positions[i] = greedy_topk(divergence[i] · response_mask[i],
                                     k=K=3,
                                     exclude_tail=8,        # 防选 EOS
                                     min_gap=intervention_length=2)  # 避免重叠
```

### Step 4 — K 次 teacher intervention

对每个 failed sample i 和每个 t_k (k=1..K)：

```
for k in 1..K:
    t = top_k_positions[i, k]
    prefix = (prompt + y_i[:t])               # 在位置 t 之前保持 student rollout
    # teacher 在 prefix 后 greedy/sampling 生成 intervention_length=2 个 token
    intervention_tokens = teacher.argmax_decode(prefix, k_steps=intervention_length)
    # 拼接 composite rollout（teacher 改写 [t, t+intervention_length)，其余沿用 student tail）
    y'_{i,k} = y_i[:t] + intervention_tokens + y_i[t+intervention_length:]
    R'_{i,k} = reward_fn(x, y'_{i,k})
    ΔR_{i,k} = R'_{i,k} - R_i
```

(K=3 → 每个失败 sample 产 3 个 composites)

### Step 5 — 构造 per-token causal credit c_t

```
c_t = zeros(B, T)   # B = total samples (original + composites), T = response length

# 对原 failed sample i:
for k in 1..K:
    for d in 0..intervention_length:
        c_t[i, top_k_positions[i,k] + d] -= ΔR_{i,k}      # 累加负 credit

# 对 composite y'_{i,k}:
for k in 1..K, for d in 0..intervention_length:
    c_t[composite_y'_{i,k}, top_k_positions[i,k] + d] = +ΔR_{i,k}     # 正 credit
```

### Step 6 — Mode B append (variable group size)

```
augmented_batch = concat([
    original_batch,      # B = batch_size * n
    composite_1,         # 第 1 组 composites (n_failed 个)
    composite_2,         # 第 2 组
    ...
    composite_K,
])
# 同一 prompt 的 group size = n + (failed count in this prompt) × K
```

### Step 7 — GRPO group-relative advantage (seq level)

```
for each prompt uid:
    group_indices = augmented_batch.where(uid == this_uid)
    A_seq[i] = R_i - mean_{j in group}(R_j)     # 自动处理变长 group
```

### Step 8 — Token-level advantage 合成（TCCA 核心公式）

```
For base_estimator == "grpo":
    base_reweight = response_mask  (uniform)
For base_estimator == "rlsd":
    base_reweight = clip(exp(sign(A_seq) · (logp_T - logp_S)), 1±0.2)
For base_estimator == "prior_shift":
    base_reweight = ĝ_t = g_t / mean_t(g_t)

# TCCA modulation (核心)
tcca_factor = (1 + λ_token · clip(c_t, ±2.0)).clamp(min=0)

# Final per-token advantage
A_t = A_seq · base_reweight · tcca_factor · length_scale(L)
```

**关键**：c_t > 0 的位置（composite 的 teacher tokens）→ tcca_factor > 1 → A_t 被放大 → 推动 π_θ 学这些 token；c_t < 0 的位置（原 failed sample 的同位置 student tokens）→ tcca_factor < 1 → A_t 被削弱 → 推动 π_θ 远离这些 token。

### Step 9 — PPO clipped surrogate loss (沿用)

```
ratio = exp(log π_θ(y_t|·) - log π_θ_old(y_t|·))
L_PPO = E_t[ min(ratio · A_t, clip(ratio, 1±ε) · A_t) ]
```

不动 loss function，只动 advantage。

---

## 5. 为什么这么做（与 OPD 文献的论证对接）

| 设计选择 | 论证依据 |
|---|---|
| 用 ΔR 作 magnitude，env reward 作 direction | **RLSD Theorem 1**：方向⊥大小，方向必须可靠+稀疏，大小可稠密+容噪 |
| 只对 failed samples intervention | **SRPO**：sample routing 实证 (correct→GRPO/failed→特殊处理) 5-bench +3.4% |
| t\* = argmax \|logp_T - logp_S\| 排尾 | **TIP Q3 (低熵+高分歧)** 的近似；实证 v3p1-aexEOS 0.5737 > araw 0.42 |
| 不用 forward Bayes surprise g_t | **实证**：v3p1-gtarg val 0.34 + length 爆 1686，验证 g_t 选位置反向 |
| 每个 ΔR 同时给原 sample (-) 和 composite (+) | **DPO-style contrastive**：同 prefix、反向 credit → 形成对比对 |
| K=3 而非 K=1 | **TIP 启发**：Q3 token 占 3-15%，K=3 捕获多个高密度信号；vs K=1 信号稀疏 |
| 保留 student 原 tail（V1 简化）| 工程简化；理论上 student tail 应该 conditional on teacher intervention，留 V2 改进 |

---

## 6. 与既有方法的 head-to-head 对比

| 方法 | direction signal | token weight 来源 | leakage | 用 outcome reward? |
|---|---|---|---|---|
| SFT (off-policy KD) | teacher tokens | uniform | N/A | ❌ |
| GRPO (RLVR baseline) | env reward (sparse) | uniform | 无 | ✅ |
| SDPO / OPSD | distribution match | distribution match | **严重** | ❌ |
| RLSD | env reward (sparse) | exp(sign(A)·(logp_T - logp_S)) (统计学) | 无 | ✅ |
| TIP | distribution match | (1-H_S) · KL_TS (启发式) | 部分 | ❌ |
| SRPO | env reward + KL | uniform on correct / KL on failed | 弱 | ✅ |
| **TCCA (ours)** | **env reward (sparse)** | **真实 ΔR_t (counterfactual)** | **无** | **✅** |

**TCCA 的 unique selling point**：唯一同时满足
- ✅ on-policy
- ✅ env reward 锚定方向（无 leakage）
- ✅ **真实因果 token weight**（不是统计学）
- ✅ base-agnostic（可叠加在 GRPO/RLSD/SDPO 上）

---

## 7. 实验设计（待跑）

### Main figure：base-agnostic claim

| Method | biology | chemistry | physics | (multi-seed mean±std) |
|---|---|---|---|---|
| GRPO baseline | 0.66 (有 baseline) | 0.78 | 0.78 | |
| **GRPO + TCCA** | 0.6X (期 +1~3) | 0.X | 0.X | |
| RLSD baseline | 0.58 (待补) | ? | ? | |
| **RLSD + TCCA** | 0.6X (期 +1~5) | 0.X | 0.X | 📌 **论文 headline** |
| SDPO baseline | 0.59 | 0.74 | 0.65 | |
| **SDPO + TCCA** | 0.6X | 0.X | 0.X | |
| Prior-Shift (ablation) | 0.55 | — | — | |

### Ablation figures

1. **K 敏感性**：K ∈ {1, 3, 5}，验证 multi-position intervention 价值
2. **λ_token 敏感性**：λ ∈ {0.5, 1.0, 2.0}
3. **t\* 策略消融**：argmax_excl_eos / argmax_raw / g_t_argmax / Q3 (TIP-inspired)
4. **causal vs correlational**：TCCA vs Prior-Shift on same base
5. **Mode B append vs in-place replace**：验证 contrastive pair 的价值

---

## 8. 你需要确认的几个点

请 check 这几个 understanding 是否与你一致：

1. ✅ TCCA 创新在 **token-level credit assignment 这一层**，不在 reward / loss 层
2. ✅ ΔR_t 是 **token y_t 的真实因果效应**（counterfactual outcome difference），不是相关性 proxy
3. ✅ Top-K positions 是为了 **稀疏可计算**，不是因为只有 K 个位置重要
4. ✅ **base-agnostic** 意味着 TCCA 可叠加在任意 base RL（GRPO/RLSD/SDPO），不取代它们
5. ✅ Prior-Shift 已**降为 ablation**（"我们试过相关性，发现不如因果"）
6. ✅ Mode B append 提供 **contrastive pair**（原 sample + composite 同 prefix、反向 credit）
7. ✅ **不动 PPO loss function**，只重新计算 advantage A_t

如以上有任何不对齐，请告诉我具体哪点，我修正后再继续。

---

## 9. 与论文 OPD survey 的差异化定位

OPD survey 84 篇论文里，**没有一篇做真正的 token-level causal credit**：
- 大多数 OPD 用 KL distribution matching（相关性）
- 少数 (TIP / SCOPE) 用启发式 token 重要性（相关性 + 工程）
- RLSD / RLSD-family 用 evidence ratio（统计学近似，仍非因果）
- TCCA 用**真实 counterfactual intervention**——这是 OPD literature 的空白

**TCCA = OPD on more step**：把 OPD 的"token 级信号"从 distribution matching → 启发式权重 → 统计学比值 → **真实因果 ΔR_t** 的演化推进一步。

---

## 附录：术语索引

- **g_t** (Prior-Shift): teacher 自反思 forward Bayesian surprise (KL between consecutive teacher distributions)
- **ΔR / ΔR_t / ΔR_k**: causal counterfactual reward delta = R(y') - R(y)
- **t\* / top-K positions**: teacher-student divergence 最大的位置
- **c_t** (TCCA): per-token causal credit vector (B, T)，TCCA 的核心新字段
- **base_estimator**: TCCA 叠加在哪个 RL 算法上 ({grpo, rlsd, prior_shift})
- **Mode B append**: composite samples 加到 batch (不替换原样本)，形成变长 group
