# On-Policy DPO + Teacher-Guided Sampling (DPO-TGS)

> **状态**：设计文档（待 user 审）
> **日期**：2026-05-16 23:00
> **作者**：与你对话整理
> **动机**：TCCA-Lite 的 GRPO + ΔR modulation 需要 λ_div tuning，advantage 层面复杂。DPO-TGS 换一条路——同一条 rollout 链路，用 pairwise preference 直接训，不需要超参。

---

## 1. Core Idea

TCCA-Lite 做的事情是：标准 GRPO rollout + 失败样本上 teacher 修 1 token → student 续写 → composite。然后算 ΔR，modulate advantage。

**DPO-TGS 做同样的 rollout，但下游处理完全不同：**

```
y₀ = student 原始 rollout（失败，R=0）
y₁ = teacher 在 t₁ 修 1 token → student 续写 → R=1

→ R(y₁) > R(y₀) → 天然 DPO pair: chosen=y₁, rejected=y₀
→ 不需要 ΔR, 不需要 λ_div, 不需要 advantage modulation
→ 直接用 DPO loss: log σ(β · [logπ(y₁|x) - logπ(y₀|x)])
```

**对比 TCCA-Lite 的复杂度：**

| 维度 | TCCA-Lite (GRPO+ΔR) | DPO-TGS |
|---|---|---|
| rollout | 标准 n=8 + composite | 一样 |
| advantage | GRPO group-relative + λ_div·c_t modulation | **不需要** |
| loss | PPO clipped surrogate + ratio·A_t | **DPO log-sigmoid** |
| 超参 | λ_div, divergence_credit_clip | **β (DPO 标准超参)** |
| pair 来源 | 不需要 pair | 天然 chosen/rejected from chain |

---

## 2. 渐进式链式 Rollout（与 TCCA-Lite 共享）

rollout 部分几乎不改，复用 TCCA-Lite 的 OPSD teacher + student async 续写：

```
For each prompt x with ref_answer r:

  Step 0: y₀ = student.generate(x)                    # 标准 rollout n=1
          R₀ = reward_fn(x, y₀)

  For i in 1..chain_length-1:
    Step A: OPSD teacher context (含 ref answer)
    Step B: teacher forward on y_{i-1} → logp_T_opsd
    Step C: student logp_S (from rollout_log_probs)
    Step D: t_i = argmax |logp_T_opsd - logp_S|, 排尾 8 token
    Step E: z = teacher 在 t_i 写 1 token (OPSD ctx, greedy)
    Step F: prefix = x + y_{i-1}[:t_i] + [z]
    Step G: continuation = student.generate(prefix)    # async 续写到 EOS
    Step H: y_i = y_{i-1}[:t_i] + [z] + continuation
            R_i = reward_fn(x, y_i)

  收集 chain: (y₀, R₀), (y₁, R₁), ..., (y_{n-1}, R_{n-1})
```

这部分代码已经在 `tcca_chain.py` 里实现了，DPO-TGS 直接复用。

---

## 3. DPO Pair 收集

### 3.1 连续比较策略

```
For each prompt's chain (y₀, R₀), (y₁, R₁), ..., (y_{n-1}, R_{n-1}):

  pairs = []
  for i in 1..n-1:
    if R_i > R_{i-1}:
      pairs.append((chosen=y_i, rejected=y_{i-1}))
    # else: skip (teacher 修正没帮到，或者帮了但 reward 没变)
```

**为什么不用 best-so-far？**

连续比较 (yᵢ vs y_{i-1}) 比 best-so-far 更严格——每一步修正必须**相对上一步有改善**才构成 pair。这样 pair 更干净，但也可能更少。论文 ablation 可以对比两种策略。

### 3.2 全局 batch 级 pair 收集

除了链内比较，还可以做**跨样本比较**：

```
For each prompt's n rollouts (standard GRPO n=8):
  按 reward 排序: y_{(1)} ≤ y_{(2)} ≤ ... ≤ y_{(n)}
  
  # 可选: best vs worst
  pairs.append((chosen=y_{(n)}, rejected=y_{(1)}))
  
  # 或相邻比较
  for i in 1..n-1:
    if R_{(i+1)} > R_{(i)}:
      pairs.append((chosen=y_{(i+1)}, rejected=y_{(i)}))
```

### 3.3 混合策略（推荐默认）

```
For each prompt:
  1. 标准 GRPO n=8 rollout → 相邻比较收集 pairs
  2. 失败样本 (R<0.5) 上做 teacher-guided chain (chain_length-1 步)
     → 链内连续比较收集 pairs
  
  total_pairs[prompt] = grpo_pairs + chain_pairs
```

这样 baseline 就有 8 个标准样本的比较信号，加上 teacher-guided 的修正信号。

### 3.4 pair 过滤

```
# 可选: 只保留高置信 pair
pairs = [(chosen, rejected) for chosen, rejected in pairs 
         if R(chosen) - R(rejected) ≥ margin]  # margin=0 即不过滤

# 可选: 去重 (相同 chosen/rejected 对只保留一次)
```

---

## 4. DPO Loss

### 4.1 标准 DPO

```python
# pairs: list of (chosen, rejected) tuples, each with responses and log_probs
# 总 pair 数 N = sum(len(prompt_pairs) for prompt in batch)

for chosen, rejected in pairs:
    # log π_θ(y|x) for chosen and rejected
    logp_chosen = compute_log_prob(model, chosen)      # (T,) per-token
    logp_rejected = compute_log_prob(model, rejected)   # (T,) per-token
    
    # 用 reference model (训练前的模型或 EMA) 计算 ref log_probs
    ref_logp_chosen = compute_log_prob(ref_model, chosen)
    ref_logp_rejected = compute_log_prob(ref_model, rejected)
    
    # DPO loss (per-pair scalar)
    chosen_reward = β * (logp_chosen.sum() - ref_logp_chosen.sum())
    rejected_reward = β * (logp_rejected.sum() - ref_logp_rejected.sum())
    
    loss = -log σ(chosen_reward - rejected_reward)
         = -F.logsigmoid(chosen_reward - rejected_reward)

# batch-level loss
total_loss = loss.mean()  # over all pairs
```

### 4.2 与 PPO 混合

两种方案：

**方案 A：纯 DPO（简单，推荐先试）**

```
# 完全替换 PPO loss
L = L_DPO

# 不需要 advantage, 不需要 reward_fn 的精确值, 只需要 pairwise 比较
```

**方案 B：DPO + PPO 混合（渐进过渡）**

```
# 有 DPO pair 的样本用 DPO loss, 没有的用 PPO
L = α · L_DPO + (1-α) · L_PPO

# α=1.0: 纯 DPO
# α=0.5: 混合
# α=0.0: 纯 PPO (baseline)
```

### 4.3 Reference Model 选择

| 选项 | 说明 | 优劣 |
|---|---|---|
| 初始模型 checkpoint | 训练开始时保存一次 | 标准 DPO 做法，简单 |
| EMA teacher | 同 TCCA-Lite 的 EMA teacher | 动态 reference，可能更好但也更复杂 |
| 无 reference (IPO) | 去掉 ref 项 | Implicit Preference Optimization, 更简单但理论弱 |

推荐先用**初始模型 checkpoint**，最简单。

---

## 5. Integration with ray_trainer.py

### 5.1 新 config

```yaml
# verl/trainer/config/dpo_tgs.yaml
defaults:
  - ppo_trainer
  - user
  - _self_

max_model_len: 18944
seed: 42

actor_rollout_ref:
  actor:
    ppo_mini_batch_size: 32
    self_distillation:
      max_reprompt_len: 10240
      is_clip: 2.0
      teacher_regularization: ema
      teacher_update_rate: 0.05
      include_environment_feedback: false
    optim:
      lr: 1e-5
  rollout:
    n: 8
    calculate_log_probs: True

algorithm:
  adv_estimator: dpo_teacher_guided
  
  dpo:
    beta: 0.1                    # DPO temperature
    ref_model_path: ""           # 初始模型 checkpoint 路径
    ref_update_rate: 0.0         # 0.0 = 固定 reference, >0 = EMA 更新
    
  teacher_guided:
    enable_guided_sampling: true
    chain_length: 4              # 链长度 (含 y₀)
    failed_threshold: 0.5
    max_intervention_per_prompt: 4
    opsd_ref_template: "The correct answer is {r}.\n"
    divergence_metric: argmax_excl_eos
    exclude_tail_tokens: 8
    
  pair_collection:
    strategy: chain_consecutive  # chain_consecutive | chain_best | global_adjacent | hybrid
    margin: 0.0                  # 最小 reward gap, 0=不过滤

data:
  train_batch_size: 32
  gen_batch_size: ${data.train_batch_size}

trainer:
  val_before_train: False
```

### 5.2 ray_trainer.py 改动点

**改动 1：rollout 阶段（~2230 行附近）**

```python
# 现有: generate_sequences → 标准 rollout
# 新增: if adv_estimator == 'dpo_teacher_guided' and enable_guided_sampling:
#         → 调 tcca_chain helpers 做 chain rollout
#         → 收集 (y_i, R_i) chain per prompt
```

复用 `tcca_chain.py` 的 `_build_opsd_teacher_context`, `_compute_divergence_opsd`,
`_select_t_i`, `_teacher_write_one_token`, `_student_continue_async`, `_build_y_i_batch`,
`_rescore_reward`。

**改动 2：pair 收集（rollout 后）**

```python
# 新增: collect_dpo_pairs(batch, chain_data)
# → 按 strategy 收集 (chosen, rejected) pairs
# → 写入 batch.non_tensor_batch['dpo_pairs']
```

**改动 3：advantage 计算阶段（~409 行附近）**

```python
# 现有: elif adv_estimator == "intervention_credit": ...
# 新增: elif adv_estimator == "dpo_teacher_guided":
#         → 跳过 standard advantage computation
#         → 直接用 pairs 计算 DPO loss
```

**改动 4：loss 计算阶段（~450 行附近）**

```python
# 新增 DPO loss:
dpo_loss = compute_dpo_loss(
    actor=actor,
    ref_model=ref_model,
    pairs=batch.non_tensor_batch['dpo_pairs'],
    beta=config.algorithm.dpo.beta,
)
loss = alpha * dpo_loss + (1 - alpha) * ppo_loss
```

### 5.3 新文件

```
verl/trainer/ppo/dpo_tgs/
├── __init__.py
├── pair_collector.py      # DPO pair collection from chain/global rollouts
├── dpo_loss.py            # DPO loss computation (sequence-level)
└── dpo_tgs_trainer.py     # Main integration (optional, can inline into ray_trainer)
```

---

## 6. Compute Cost Analysis

### 6.1 与 TCCA-Lite 对比

| 组件 | TCCA-Lite | DPO-TGS |
|---|---|---|
| rollout | ~40s (+33% baseline) | 一样 |
| teacher fwd | skip (在 intervention_rollout 内) | 一样 |
| **ref model fwd** | 不需要 | **+~5s** (需要 ref model 算 log_probs on pairs) |
| update_actor | ~15s (+50% baseline) | ~15s (DPO loss 计算量类似 PPO) |
| **总** | ~83s (+25%) | **~88s (+33%)** |

DPO-TGS 比 TCCA-Lite 多一次 reference model forward（算 ref log_probs），但 DPO loss 本身比 PPO advantage computation + modulation 简单。总 overhead ~33%，仍可接受。

### 6.2 chain_length 影响

| chain_length | 额外 rollout | ref fwd pairs | 总 overhead |
|---|---|---|---|
| 2 (y₀+y₁) | +10s | +2s | ~20% |
| 4 | +25s | +5s | ~33% |
| 8 | +45s | +8s | ~50% |

推荐默认 **chain_length=4**，balance signal density vs compute。

---

## 7. 实验设计

### 7.1 Main comparison

| Experiment | beta | chain_length | dataset | 用途 |
|---|---|---|---|---|
| GRPO baseline (n=8) | N/A | N/A | biology | 对照 |
| **DPO-TGS (pair_strategy=hybrid)** | 0.1 | 4 | biology | **论文 main** |
| TCCA-Lite (λ_div=1.0) | N/A | N/A | biology | 同 rollout, 不同 loss |
| TCCA-Lite (λ_div=0.0) | N/A | N/A | biology | 纯 GRPO + counterfactual sampling |

### 7.2 Ablation

| Variable | Values | 用途 |
|---|---|---|
| beta | {0.01, 0.1, 0.5} | DPO temperature |
| chain_length | {2, 4, 8} | 链长度对 signal density 的影响 |
| pair_strategy | {chain_consecutive, chain_best, hybrid} | pair 收集策略 |
| ref_model | {initial_ckpt, ema_teacher} | reference model 选择 |
| margin | {0.0, 0.5, 1.0} | pair 过滤阈值 |
| alpha (mix) | {1.0, 0.5, 0.0} | DPO vs PPO 混合比例 |

### 7.3 关键 metric

除了 val acc，DPO-TGS 特有的诊断指标：

| Metric | 含义 |
|---|---|
| `dpo/pairs_per_prompt_mean` | 平均每 prompt 收集到几个 pair |
| `dpo/pair_win_rate` | chain 中 R(yᵢ) > R(yᵢ₋₁) 的比例 |
| `dpo/reward_gap_mean` | 平均 R(chosen) - R(rejected) |
| `dpo/logit_margin_mean` | 平均 β·(logπ_chosen - logπ_rejected) |
| `dpo/loss` | DPO loss 值 |

---

## 8. 与现有工作的关系

### vs DPO (original)

原始 DPO 用人工标注的 chosen/rejected pairs。DPO-TGS 的 pair 来自**自动生成的 teacher-guided chain**，on-policy（student 自己生成的），不需要人工标注。

### vs IPO (Implicit Preference Optimization)

IPO 去掉 reference model。DPO-TGS 保留 ref model 来约束 policy 不偏离太远。论文可以 ablate ref model 的作用。

### vs Online DPO / Iterative DPO

Online DPO 每步用当前 policy 生成 pair 然后更新。DPO-TGS 类似但 pair 不是随机生成的——是 **teacher-guided counterfactual**，有信息量。

### vs TCCA-Lite

同样的 rollout 基础设施，不同的 learning signal：
- TCCA-Lite: GRPO advantage + ΔR modulation → PPO
- DPO-TGS: pairwise preference → DPO loss

---

## 9. 实施风险

| 风险 | 缓解 |
|---|---|
| 🚨 pair 太少 (chain 中 R 不递增) | 降低 margin=0, 或用 best-so-far 策略 |
| 🚨 ref model forward 增加 compute | 用初始 checkpoint 固定, 不每步更新 |
| 🟡 DPO 在 sparse reward (0/1) 下可能信号弱 | 考虑 soft reward (confidence score) |
| 🟡 chain_length 和 GRPO n 的关系 | chain 是额外的, n=8 标准样本仍然在 |
| 🟢 代码改动大? | rollout 复用 tcca_chain.py, 主要改 pair collection + loss |

---

## 10. 实施分阶段

| Phase | 内容 | 时长 |
|---|---|---|
| P0 (now) | 本文档你审 | 已完成 |
| P1 | `pair_collector.py` + `dpo_loss.py` 新文件 | 0.5 天 |
| P2 | 集成到 `ray_trainer.py` (新 adv_estimator 分支) | 0.5 天 |
| P3 | 端到端 smoke (chain_length=2, 纯 DPO, alpha=1.0) | 0.3 天 |
| P4 | Nebula main exp (biology, chain_length=4, beta=0.1) | ~12h |
| P5 | Ablation sweep | ~24h |
| **总计** | | **1.5 天 + 36h GPU** |

---

## 等你 review

请 check 这几个设计是否对齐：

- [ ] §3 pair 收集策略：chain_consecutive vs hybrid？推荐 hybrid（GRPO n=8 + chain pairs）
- [ ] §4 DPO loss：纯 DPO (alpha=1.0) 先试，还是直接混合？
- [ ] §5 ref model：初始 checkpoint (简单) vs EMA teacher (动态)？
- [ ] §6 chain_length=4 默认 OK？还是先 2 smoke？
- [ ] §7 实验设计优先级：先跟 GRPO baseline 对比，还是先跟 TCCA-Lite 对比？
