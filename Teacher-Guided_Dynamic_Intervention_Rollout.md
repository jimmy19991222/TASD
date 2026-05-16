# Teacher-Guided Dynamic Intervention Rollout

> **状态**：方案设计阶段  
> **关联**：[Prior-Shift](file:///Users/awesome_jimmy/lazada/SDPO/verl/trainer/ppo/bayesian_credit/prior_shift.py) / [Bayesian Credit Assignment](#7-与-bayesian-credit-assignment-叙事的关系)  
> **更新**：2026-05-15

---

## 1. Motivation

### 问题

当前 Prior-Shift (Tier 1) 的 credit assignment 是**纯相关性**的：

```
A_t = A_seq · ĝ_t   （ĝ_t 来自 teacher 的 forward KL surprise）
```

它回答的是："teacher 在哪个位置感到惊讶？"  
**而不是**："student 在哪个 token 犯了错误？"

v1 smoke 实验暴露了这种相关性归因的脆弱性——当 max_ratio 过大时，credit 分配失准导致 length collapse（372→18 token）。

### 核心洞察

**Student 的错误本质上是某个 token 的选择偏离了 teacher 的认知**。如果能**精确找到分歧最大的位置**，让 teacher **在那个点上接管生成**，student 就能看到：

> "如果我在 t* 处采用了 teacher 的建议，后续会怎样？"

这就是 **causal intervention** — 不是猜测哪个 token 重要，而是直接做实验验证。

### 直觉类比

- **Prior-Shift**：teacher 站在旁边说"我觉得第 5 个 token 很奇怪"（相关性）
- **Teacher-Guided Intervention**：teacher 走到键盘前，把第 5 个 token 改掉，让 student 接着写，看看结果变好还是变差（因果性）

---

## 2. 核心算法

### 2.1 单轮流程

```
Algorithm: Teacher-Guided Dynamic Intervention (Single Round)
Input:  prompt x, teacher T, student S, intervention length k
Output: modified rollout y', divergence position t*, reward delta ΔR

1. Student rollout:
   y_s = (y_1, y_2, ..., y_T) ~ P_S(· | x)

2. Teacher evaluation:
   for t = 1..T on response_mask:
       p_T(t) = P_T(y_t | x, y_<t, teacher_context)
       p_S(t) = P_S(y_t | x, y_<t)

3. Find divergence peak:
   t* = argmax_{t} |log P_T(y_t) - log P_S(y_t)|

4. Teacher intervention:
   y_intervention = (y'_t*, y'_{t*+1}, ..., y'_{t*+k-1}) ~ P_T(· | x, y_<t*, teacher_context)

5. Student continuation:
   y_tail = (y_{t*+k}, ..., y_{T'}) ~ P_S(· | x, y_<t*, y_intervention)

6. Composite rollout:
   y' = (y_<t*, y_intervention, y_tail)

7. Credit computation:
   ΔR = R(y') - R(y_s)
   A_seq = ΔR · sign(original group advantage)
```

### 2.2 多轮迭代（可选）

```
Round 1: y_s    → 找 t*₁ → teacher 介 k tokens → y'₁, ΔR₁
Round 2: y'₁    → 找 t*₂ → teacher 介 k tokens → y'₂, ΔR₂
...
Round N: y'_{N-1} → 找 t*ₙ → teacher 介 k tokens → y'_N, ΔRₙ
```

每轮 divergence 检测基于**当前 rollout**（y'_{i-1}），因此后续轮次可能找到新的分歧点。

**收敛条件**：
- |ΔR_i| < ε（intervention 不再改变 reward）
- 或 t* 已在之前轮次的 intervention 区域（避免死循环）

### 2.3 Advantage 构造

```
# 方式 1: 直接用 ΔR 作为 seq-level advantage
A_seq = ΔR

# 方式 2: ΔR 在 GRPO group 内中心化 + 保留 g_t 信息
A_seq = ΔR - mean_{j ∈ group} ΔR_j
A_t   = A_seq · g_t / mean_t(g_t) · length_scale(L)
```

**推荐方式 2**：保留 Prior-Shift 的 token-level 分配能力，但 seq-level credit 用 ΔR 替代原始 R。

---

## 3. 数学公式

### 3.1 Divergence Metric

**选项 A: Log-prob difference（简单，推荐先用）**

```
d_t = |log P_T(y_t | x, y_<t) - log P_S(y_t | x, y_<t)|
t*  = argmax_t d_t · response_mask_t
```

- 含义：teacher 和 student 在 token y_t 上的概率差
- 计算成本：极低（teacher forward 本身就要算 logp）

**选项 B: Full-distribution KL（更精确，但更贵）**

```
d_t = KL( P_T(· | x, y_<t) ‖ P_S(· | x, y_<t) )
t*  = argmax_t d_t · response_mask_t
```

- 含义：teacher 和 student 在位置 t 的整体分布差异
- 计算成本：需要 full-vocab logits（与 Prior-Shift 的 g_t 计算同款）

### 3.2 Intervention Length

```
k = max(1, min(k_max, α · (T - t*)))
```

- k=1：仅修正分歧 token（最小干预）
- k=2~3：给 teacher 一点"上下文"（推荐）
- α=0.1：干预长度不超过剩余序列的 10%

### 3.3 Credit Assignment 因果性

```
如果 ΔR > 0：intervention 让 reward 变好了 → t* 处是 student 的关键错误
如果 ΔR < 0：intervention 让 reward 变差了 → t* 处的"错误"可能是 student 的策略选择
如果 ΔR ≈ 0：t* 处的分歧对最终结果不重要
```

**关键属性**：这是**真实的因果效应**（counterfactual），不是相关性推断。

---

## 4. 开销分析

### 4.1 单轮版本（Mode B 追加，推荐）

| 操作 | 次数 (n_initial=7) | 相对 GRPO |
|---|---|---|
| Student rollout | 7× T | 0.875× |
| Teacher forward (evaluation) | 7 × p × T | 0.1× |
| Teacher generation (intervention) | 7 × p × k | ~0.05× |
| Student continuation | 7 × p × (T-k) | ~0.3× |
| **合计** (p=0.5) | | **~1.05×** |

### 4.2 仅错误样本（推荐优化）

```python
# 仅对 reward < threshold 的样本做 intervention
if reward < 0.5:
    do_intervention()
else:
    skip()
```

| 错误率 p | 相对 GRPO (n_initial=7) |
|---|---|
| 100% (全量) | 1.15× |
| 50% (典型) | **1.05×** |
| 25% | 1.02× |

**结论：几乎无额外开销，仅 +5% 相对传统 GRPO n=8。**

### 4.3 与现有方案对比

| 方案 | 相对 GRPO | Credit 类型 |
|---|---|---|
| GRPO baseline | 1.0× | 均匀分配 |
| Prior-Shift v2 | 1.5× | 相关性（teacher surprise） |
| RLSD | 1.5× | 相关性（teacher-student ratio） |
| **Teacher-Guided Intervention** | **~1.1×** | **因果性（counterfactual）** |
| 方向 2 Counterfactual | 4.0× | 因果性（full trajectory） |

### 4.4 Intervention 结果如何处理：替换 vs 追加

关键设计决策：intervention 后的新 rollout y' 如何处理？

#### Mode A：替换（不推荐）

```
原始:    y₁, y₂, y₃, y₄, y₅, y₆, y₇, y₈  (n=8)
y₃ 失败 → intervention → y₃' 替换 y₃：
新组:    y₁, y₂, y₃', y₄, y₅, y₆, y₇, y₈  (仍是 n=8)
```

**问题**：原始错误 y₃ 被擦除了。student 看不到"如果不听 teacher 会怎样"。

#### Mode B：追加（✅ 推荐）

```
原始:    y₁, y₂, y₃, y₄, y₅, y₆, y₇, y₈  (n=8)
y₃ 失败 → intervention → y₃' 追加：
新组:    y₁, y₂, y₃, y₃', y₄, y₅, y₆, y₇, y₈  (n=9)
```

**天然的 contrastive pair**：

```
同一 prompt，同一前缀 y_<t*：
  y₃:  选了 student 的 token → reward 低 → A < 0  → 推低此 token
  y₃': 选了 teacher 的 token → reward 高 → A > 0  → 推高此 token
```

GRPO advantage 天然支持变长 group：

```python
# 现有 prior_shift.py 已按 uid 分组，每组独立算 mean
for uid, idxs in uid_to_indices.items():
    group_r = seq_reward[idxs]   # Prompt A: 9 个, Prompt B: 8 个 — 各自算
    mu = group_r.mean()
    seq_advantage[idxs] = group_r - mu
```

**不需要改任何 advantage 计算代码。** `uid_to_indices` 天然支持变长 group。

#### 自适应 group size（进阶）

```
策略：
  n_initial = 7（比传统 n=8 少 1 个）

  for each prompt:
    正常 rollout n_initial 个
    对失败的做 intervention，追加到 group
    
    最终 group_size = n_initial + n_failed（可变）

开销对比 (p=0.5, n=7):
  传统 GRPO:    8 × student_generate = 8.0×
  自适应方案:   7 × student_generate + 7×0.5 × teacher_forward + 7×0.5 × teacher_generate(k=2)
              = 7.0 + 1.05 + 0.35     = 8.4×
              = 1.05× 传统 GRPO（仅 +5% 开销！）
```

| 对比 | Mode A（替换） | Mode B（追加）✅ |
|---|---|---|
| 训练信号 | 仅正例 | 正 + 负 contrastive pair |
| 因果性 | 弱 | 强（同一前缀，不同 token → 不同 reward） |
| Credit 精度 | 低 | 高（精确到 divergence token） |
| GRPO 代码改动 | 无 | 无（uid_to_indices 天然支持） |
| Group 大小 | 固定 | 可变（n + n_failed） |

---

## 5. 工程实现

### 5.1 架构概览

```
ray_trainer.py
    │
    ├── rollout_worker（vLLM/SGLang backend）
    │   ├── student.generate(x)           → y_s
    │   ├── teacher.evaluate(x, y_s)      → logp_T, d_t
    │   ├── find_divergence(d_t)          → t*
    │   ├── teacher.generate(x, y_<t*)    → y_intervention
    │   └── student.generate(x, y_<t*, y_intervention) → y_tail
    │
    └── advantage estimator（新增）
        └── intervention_credit.py
            ├── compute ΔR = R(y') - R(y_s)
            ├── 可选: per-token g_t reweighting
            └── 返回 A_t
```

### 5.2 核心接口

```python
# verl/trainer/ppo/bayesian_credit/intervention_credit.py

@register_adv_est("intervention_credit")
def compute_intervention_credit(
    token_level_rewards: torch.Tensor,      # (B, T)
    response_mask: torch.Tensor,            # (B, T)
    index,                                   # uid array
    teacher_prior_shift_surprise: torch.Tensor,  # (B, T) g_t
    intervention_delta_reward: torch.Tensor,     # (B,) ΔR (v3 新增)
    divergence_position: torch.Tensor,           # (B,) t* (v3 新增，可选)
    config: Optional[dict] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Teacher-Guided Intervention advantage (v3).
    
    Compared to prior_shift.py:
    - Replaces R_i with ΔR (causal credit instead of outcome reward)
    - Retains g_t = KL(P_T‖P_{T-1}) for per-token reweighting
    - Optionally uses divergence_position for position-aware scaling
    """
```

### 5.3 Rollout 引擎改动

```python
# 伪代码：在 verl/workers/rollout/ 中新增 intervention 模式

def rollout_with_intervention(prompts, teacher_model, student_model, config):
    """
    单轮 dynamic intervention rollout.
    """
    # Step 1: Student generation
    responses = student_model.generate(prompts)
    
    # Step 2: Teacher evaluation (reuse existing lightweight forward)
    teacher_logps = teacher_model.forward(prompts, responses)
    student_logps = student_model.forward(prompts, responses)
    
    # Step 3: Find divergence peaks
    divergence = (teacher_logps - student_logps).abs()
    t_star = divergence.argmax(dim=-1)  # (B,)
    
    # Step 4: Teacher intervention at t*
    intervention_tokens = teacher_model.generate(
        prompts, 
        prefix=responses[:, :t_star],  # teacher 从 t* 开始生成
        max_new_tokens=k,
    )
    
    # Step 5: Student continuation
    tail_tokens = student_model.generate(
        prompts,
        prefix=torch.cat([responses[:, :t_star], intervention_tokens], dim=-1),
    )
    
    # Step 6: Composite rollout
    new_responses = torch.cat([
        responses[:, :t_star],
        intervention_tokens,
        tail_tokens,
    ], dim=-1)
    
    # Step 7: Register delta reward for advantage computation
    delta_reward = compute_reward(new_responses) - compute_reward(responses)
    
    return new_responses, delta_reward, divergence
```

### 5.4 与现有基础设施的复用

| 组件 | 来源 | 说明 |
|---|---|---|
| Teacher lightweight forward | [dp_actor.py](file:///Users/awesome_jimmy/lazada/SDPO/verl/workers/actor/dp_actor.py#L776-L880) | 已有 `compute_prior_shift_surprise` 路径 |
| g_t (KL surprise) | [prior_shift.py](file:///Users/awesome_jimmy/lazada/SDPO/verl/trainer/ppo/bayesian_credit/prior_shift.py) | 复用为 per-token reweighting |
| max_ratio / renorm / len_penalty | [prior_shift.py](file:///Users/awesome_jimmy/lazada/SDPO/verl/trainer/ppo/bayesian_credit/prior_shift.py#L72-L75) | v2 的防护机制全部保留 |
| Teacher context 构建 | [ray_trainer.py](file:///Users/awesome_jimmy/lazada/SDPO/verl/trainer/ppo/ray_trainer.py#L2158) | 复用 teacher_context + GT |
| Teacher EMA | [prior_shift.yaml](file:///Users/awesome_jimmy/lazada/SDPO/verl/trainer/config/prior_shift.yaml) | r=0.05 |

---

## 6. 实验设计（v3）

### 6.1 v3-lite（首选：context 引导版，不改 rollout 引擎）

**思路**：在 teacher context 中显式标记 divergence 位置，依赖 teacher context 的 reprompt 机制引导修正。

```
reprompt template:
"You generated: {student_rollout}
At token position {t*}, your choice of '{y_t*}' may be suboptimal.
The teacher suggests: '{teacher_intervention}'
Please re-generate from position {t*} with this guidance."
```

**改动范围**：
- [x] 复用现有 teacher context 构建（`include_environment_feedback` 机制）
- [ ] 新增 divergence detection 逻辑（~30 行）
- [ ] 修改 reprompt template（~20 行）

**预期开销**：~1.3× GRPO（仅多一次 teacher evaluation）

### 6.2 v3-full（真实 intervention 版，需改 rollout 引擎）

**方法**：Section 5.3 的完整实现。

**改动范围**：
- [ ] 新增 `intervention_credit.py` advantage estimator
- [ ] 改 `verl/workers/rollout/` 支持中途接管
- [ ] 新增 `intervention_credit.yaml` config

**预期开销**：~1.65× GRPO（见 Section 4.3）

### 6.3 Ablation 矩阵

| 实验 | 配置 | 验证目标 |
|---|---|---|
| v3a | v3-lite, k=2, 单轮 | Context 引导是否有效 |
| v3b | v3-lite, k=5, 单轮 | 干预长度影响 |
| v3c | v3-full, k=2, 单轮 | 真实 intervention vs context 引导 |
| v3d | v3-full, k=2, 仅错误样本 | 开销优化版 |

---

## 7. 与 Bayesian Credit Assignment 叙事的关系

```
Bayesian Credit Assignment from a Self-Distilled Teacher
│
├── Tier 1: Prior-Shift (✅ 已实现)
│   A_t = A_seq · g_t / mean_t(g_t)
│   类型：相关性（teacher forward surprise）
│
├── Tier 2: Posterior-Shift (⏳ 待实现)
│   A_t ∝ log P_T(y_t|x,y,r) - log P_T(y_t|x,y_<t)
│   类型：相关性（reward-conditioned hindsight）
│
└── Tier 3: Causal Intervention ← 本文档 (📋 规划中)
    A_seq = ΔR = R(y_intervened) - R(y_original)
    类型：因果性（counterfactual intervention）
    + 保留 Prior-Shift 的 per-token g_t reweighting
```

**论文差异化**：
- RLSD：teacher-student prob ratio（相关性）
- Prior-Shift：teacher own belief shift（相关性）
- **Ours v3**：**causal counterfactual** — 唯一用真实干预验证 credit 的方法

---

## 8. 关键设计决策

### 待定项

| 决策 | 选项 | 推荐 |
|---|---|---|
| Divergence metric | Log-prob diff / KL | **Log-prob diff**（复用 teacher forward） |
| 单轮 vs 多轮 | 1 / 2~3 | **单轮**（边际收益递减，开销更可控） |
| 仅错误样本？ | Yes / No | **Yes**（省 50% 开销） |
| 先做 lite 还是 full？ | lite / full | **先 lite**（1-2 天跑通，验证 idea） |
| k 值 | 1 / 2 / 3 / 5 | **k=2**（平衡修正力度和 student 学习空间） |

### 风险

| 风险 | 缓解措施 |
|---|---|
| Teacher intervention 质量差（teacher 本身不够强） | EMA r=0.05 保证 teacher 是"近期最佳 student"的平滑版 |
| 多轮 iteration 不收敛 | 加 max_rounds=2 限制 |
| Intervention 让 response 变长（teacher 倾向生成更多 token） | 加 response_length penalty + k 上限 |
| 与 v2 同样的 length collapse | 保留 v2 的 min_response_length + linear penalty |

---

## 9. Related Work

- **Counterfactual Rollout (方向 2)**：全序列 counterfactual，但开销 4-11×。本方案通过"只 intervention 分歧点"将开销降至 ~1.65×
- **RLSD (arXiv:2604.03128v2)**：teacher-student log-ratio，相关性归因，无 intervention
- **GRPO (DeepSeek-R1)**：group-level outcome advantage，无 token-level credit
- **DAPO**：filter_groups + clip_ratio，无 teacher 参与
- **TASD**：teacher prob gate 控制 token 流入 loss，相关性归因

---

## 10. 下一步

1. **等 Prior-Shift v2a/v2b 结果**（预计 2026-05-15 21:00）
2. 若 v2 有效 → v3 作为论文 follow-up / ablation
3. 若 v2 仍失败 → **立即启动 v3-lite**（1-2 天跑通）
4. v3-lite 有效 → 补 v3-full 实现（论文 Tier 3 章节）
