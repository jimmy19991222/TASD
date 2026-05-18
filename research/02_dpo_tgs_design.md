# 02 — DPO-TGS V2.5 完整设计

> **当前主推方法的 single source of truth**。
> 历史演化见 [01_evolution.md](01_evolution.md);理论锚点见 [03_theory_anchor.md](03_theory_anchor.md);实验状态见 [04_experiments.md](04_experiments.md)。

---

## §0 一句话

> **DPO-TGS** = chain rollout 中 **teacher 单 token 干预**产生的因果反事实样本作为 (chosen, rejected) pair,直接 DPO 训练。在 loss 层面提出 3 个针对 teacher-guided 设定的创新,每个对应 vanilla DPO 没用上的一条独特结构。

## §1 核心 idea (vs vanilla DPO 与 TCCA-Lite)

```
Vanilla DPO:        预收集 (y⁺, y⁻) → DPO loss
                    问题: offline pair, distribution shift, 信号粒度粗

OAIF (online):      π_θ_t 独立采两 response → LLM judge 打偏好 → DPO
                    问题: 依赖 LLM judge,长度偏置,无因果定位

TCCA-Lite (我们的):   GRPO + 失败样本上 teacher 单 token 修 + ΔR modulation
                    问题: λ_div 调参,advantage 层信号粒度

DPO-TGS V2 (当前):  GRPO + 失败样本 chain 干预 + chain 内 R 改善形成 pair
                    + DPO loss (替代 ΔR modulation)
                    + V2.5 3 个 loss innovations 利用 teacher-guided 独特结构
```

vs TCCA-Lite 同 rollout 路径,**完全不同的 learning signal**:从 GRPO advantage modulation 切换到 pairwise preference + DPO。

## §2 V2 Adaptive Rollout (实施版本)

代码:[`adaptive_rollout.py`](../verl/trainer/ppo/dpo_tgs/adaptive_rollout.py)

### Pipeline

```
Phase 1: 标准 rollout n_init per prompt
   y_init = π_θ.rollout(x, n=n_init=2)   # B × n_init samples

Phase 2: per-prompt SDPO teacher ctx
   For each prompt:
     correct_rollouts = [y for y in y_init if R(y) ≥ correct_threshold (default 1.0)]
     if correct_rollouts:
        teacher_ref_text = decode(correct_rollouts[0])
        ctx = "Refer to this correct answer: {teacher_ref_text}\n" + original_prompt
     else:
        if all_failed_strategy == "skip":
           # 默认: 跳过该 prompt 的 intervention (此 prompt 不参与 DPO pair, 走 GRPO fallback)
        elif all_failed_strategy == "gt_fallback":
           # ablation: 用 dataset GT 作 ref

Phase 3: 对每个 failed sample (R < correct_threshold) 迭代干预
   For each failed_y in y_init where uid has valid sdpo_ctx:
     chain = [failed_y]
     used_positions = set()
     For attempt in 1..n_attempts (default 2):
        a. teacher fwd (SDPO ctx) on chain[-1] → logp_T_opsd
        b. divergence = |logp_T_opsd - logp_S| · response_mask
        c. mask out used_positions + tail-8
        d. t* = argmax(divergence)
        e. teacher decode 1 token z_T at t* (OPSD ctx)
           ★ if z_T == y_S[t*]: reselect t* (next argmax),max 3 次
        f. student async continue from prefix
        g. y_attempt = chain[-1][:t*] + z_T + continuation
        h. used_positions.add(t*)
        i. chain.append(y_attempt)

Phase 4: concat y_init + 所有 chain attempts
   tag dpo_lineage_id (init 索引或 derive from)
   tag dpo_attempt_idx (0=init, 1..n_attempts=chain attempt)
   tag dpo_t_star (t_i for attempts, -1 for init)

Phase 4b (optional, 当 use_teacher_anchored_ref=True):
   1 次 post-hoc OPSD teacher fwd over augmented batch
   → batch.batch['teacher_log_prob_opsd'] (B*n, T) float32
   NaN rows for prompts without valid SDPO ctx
```

### 关键 design 决策

| 维度 | 选择 | 理由 |
|---|---|---|
| n_init 默认 | 2 | smoke; 节省 compute 同时保证有 sibling correct 概率 |
| n_attempts 默认 | 2 | 每个 failed sample 最多 2 次干预 (avoid 过度干预) |
| correct_threshold | 1.0 (sciknoweval verifier) | 根据数据集 reward 范围 |
| sdpo_ctx_source | sibling_correct (默认) | self-distillation 纯粹,无 GT 依赖 |
| all_failed_strategy | skip (默认) | Meta paper 推荐,避免 noisy pair |
| Token mismatch | reselect_t (max 3 次) | argmax 后查 z_T ≠ y_S[t*],避免 silent ΔR=0 |
| 排尾 token | 8 | 避免选 EOS / stylistic |

## §3 V1 线性化 DPO Loss

代码:[`dpo_loss.py`](../verl/trainer/ppo/dpo_tgs/dpo_loss.py)

为避免 actor surgery (proper DPO 需 pair-aware micro-batching),把 DPO 梯度方向**编码为 per-token advantage**,复用 PPO clipped surrogate:

### 公式

For each (chosen, rejected) pair:
$$\text{margin} = \beta \cdot \sum_t \text{mask}_t \cdot (\log \pi_{\text{old}}(t) - \log \pi_{\text{ref}}(t)) \Big|_{y^+} - \beta \cdot \sum_t \text{mask}_t \cdot (\log \pi_{\text{old}}(t) - \log \pi_{\text{ref}}(t)) \Big|_{y^-}$$
$$g = \beta \cdot \sigma(-\text{margin}) \quad \text{(per-pair gradient magnitude)}$$
$$A_t[y^+] = +g / L_{y^+} \cdot \text{mask}, \quad A_t[y^-] = -g / L_{y^-} \cdot \text{mask}$$

非 paired sample fallback 到 GRPO group-relative,**α-mix** 控制权重:
$$A_t = \mathbb{1}[\text{paired}] \cdot (\alpha \cdot A_t^{\text{DPO}} + (1-\alpha) \cdot A_t^{\text{GRPO}}) + \mathbb{1}[\text{unpaired}] \cdot A_t^{\text{GRPO}}$$

### 性质

- ✅ 在 $\pi_\theta = \pi_{\text{old}}$ 处**第一阶精确** (PPO mini-epoch 起点)
- ✅ PPO ratio clip 控制 $\pi_\theta$ 偏离 $\pi_{\text{old}}$ 的程度
- ✅ α-mix GRPO fallback 防梯度消失 (OFS-DPO Prop 3.3.1 警告)
- ✅ 零 actor 改动,直接复用 verl PPO 框架

## §4 V2.5 三个 Loss Innovations

针对 teacher-guided pair 的 4 个独特结构 (vanilla DPO 没用上),3 个独立可 toggle 的 knob:

### 利用的 4 个结构

```
Teacher-guided pair 比普通 DPO pair 多 4 个信号:
1. 共享 prefix: y⁺ 和 y⁻ 在 t* 之前完全相同 → log-ratio 在前缀上严格 = 0
2. 因果定位: 唯一确定性差异是 t* 一个 token (continuation 是 student 自由生成)
3. Teacher 信号: OPSD ctx 下 teacher 给了 z_T,且我们知道 P_T(·|y_<t*) 整个分布
4. 连续 ΔR: 不只是偏好符号,有真实 reward 差值 R(y⁺) - R(y⁻)
```

### ① Causal-Localized DPO (`causal_localize: true`)

利用结构 (1)(2). Log-ratio 自然分解:
$$\log \frac{\pi_\theta(y^+)}{\pi_\theta(y^-)} = \underbrace{\log \frac{\pi_\theta(z_T|y_{<t^*})}{\pi_\theta(y_S|y_{<t^*})}}_{\text{token-level (decisive)}} + \underbrace{\log \frac{\pi_\theta(\text{cont}^+|y_{<t^*}, z_T)}{\pi_\theta(\text{cont}^-|y_{<t^*}, y_S)}}_{\text{continuation (derivative)}}$$

把 margin 拆成 2 路 β:
$$\text{margin}_{\text{combined}} = \beta_{\text{tok}} \cdot \Delta\text{logp}_{t^*} + \beta_{\text{cont}} \cdot \Delta\text{logp}_{>t^*}$$

**Plumbing**: `dpo_t_star` per sample (adaptive_rollout 写入) → `_build_token_localize_masks` 拆分 mask → 双路 β。

**Config**:
- `beta_token` (默认 null = use beta;推荐 2β = 0.2)
- `beta_continuation` (默认 null = 0.5β = 0.05)

**论文卖点**: DPO 一直 sequence-level,即使因果差异只有 1 token。**首次系统问 "DPO 信号住哪"**。

### ② Teacher-Anchored DPO (`use_teacher_anchored_ref: true`)

利用结构 (3). 用 OPSD teacher (含 ref answer 的 privileged-context teacher) 替换 $\pi_{\text{ref}}$:
$$\mathcal{L}_{\text{TA-DPO}} = -\log \sigma\left(\beta \log \frac{\pi_\theta(y^+) \cdot \pi_T^{\text{OPSD}}(y^-)}{\pi_\theta(y^-) \cdot \pi_T^{\text{OPSD}}(y^+)}\right)$$

**Plumbing**: Phase 4b post-hoc OPSD teacher fwd → `teacher_log_prob_opsd` (B*n, T) float32,NaN 时 per-row fallback 到 $\pi_{\text{ref}}$。

**理论支撑**: **Samplers-DPO ICLR-25 Theorem 4** — reward-aware ref ⇒ **quadratic 收敛**;OPSD teacher 是 reward-aware 的 (privileged ref answer 隐式编码 reward 方向)。

**论文卖点**: **首次用 privileged-context teacher 作 DPO ref**。OPSD teacher 是 DPO-Mix-R 的 verifiable-reward 工程实现 → 直接获得 ICLR-25 收敛理论支撑。

**额外开销**: +1 teacher fwd per chain rollout (~5%)。

### ③ ΔR-Weighted DPO (`delta_r_weight_mode: linear|sqrt|squared|sigmoid`)

利用结构 (4). 用 verifiable env reward 的真实差值 ΔR 给 pair 加权:
$$\mathcal{L}_{\Delta R\text{-DPO}} = -W(\Delta R) \cdot \log \sigma(\beta \cdot \text{margin})$$

权重函数选项:
| Mode | $W(\Delta R)$ | 用例 |
|---|---|---|
| `none` | 1 | vanilla (默认) |
| `linear` | $|\Delta R|$ | 推荐,与 reward 量级线性 |
| `sqrt` | $\sqrt{|\Delta R|}$ | 抑制极端 pair |
| `squared` | $\Delta R^2$ | 强放大大 gap |
| `sigmoid` | $\sigma(\Delta R / \tau)$ | 饱和 (`delta_r_weight_tau` 控制) |

**Plumbing**: `_delta_r_weight()` per-pair scatter,乘到 σ-gate 输出 g。

**论文卖点**: RPO 用 $W_{\text{hal}}$ 启发式权重 (人工定义 hallucination severity);**我们用 verifiable env reward 严格化** — model-free,无人工启发式。

**额外开销**: 几乎为 0 (~10 行代码)。

### 3 innovations 关系表

| Innovation | 利用结构 | 独特性 | 额外开销 | Config knob |
|---|---|---|---|---|
| ① Causal-Localized | (1)(2) 共享 prefix + 因果定位 | 首次问 "DPO 信号住哪" | ~0 | `causal_localize`, `beta_token`, `beta_continuation` |
| ② Teacher-Anchored | (3) OPSD teacher 完整分布 | 首次用 privileged-ctx teacher 作 ref → 接 ICLR-25 quadratic 收敛 | +1 teacher fwd | `use_teacher_anchored_ref` |
| ③ ΔR-Weighted | (4) 连续 ΔR magnitude | RPO 启发式权重的严格化 | ~0 | `delta_r_weight_mode`, `delta_r_weight_tau` |

3 个独立可 toggle,$2^3 = 8$ 种组合。**主推: ① + ② + ③ 全开** (论文 main result),其余作 ablation。

## §5 Pair Collection 策略

代码:[`pair_collector.py`](../verl/trainer/ppo/dpo_tgs/pair_collector.py)

### chain_consecutive (默认)

Per (uid, lineage_id) chain,sort by `attempt_idx`,`chain_consecutive`:
- For $i$ in $1..|\text{chain}|-1$: if $R(\text{chain}[i]) > R(\text{chain}[i-1]) + \text{margin}$: form pair (chosen=chain[i], rejected=chain[i-1])
- 每个 sample 至多参与 1 个 pair (chosen 与 rejected 不重叠)

### hybrid_init_chain (Meta-inspired)

= chain_consecutive **+** init pool best-vs-worst pair:
- For each prompt, in y_init samples (attempt_idx == 0):
  - if both correct (R ≥ threshold) and failed exists, form pair (chosen=best_correct, rejected=worst_failed) with R-gap > margin

**Meta paper 的实证依据**: best-vs-worst pool pair + skip all-correct/all-failed prompt 是有效设计。

## §6 Diagnostic Metrics (16 个)

代码:`pair_collector.py:compute_dpo_metrics`

### 基础 (10 个)
- `dpo/pairs_total`, `dpo/pair_win_rate`, `dpo/grpo_fallback_rate`
- `dpo/avg_chosen_length`, `dpo/avg_rejected_length`, `dpo/length_ratio_chosen_over_rejected`
- `dpo/avg_chosen_reward`, `dpo/avg_rejected_reward`, `dpo/reward_gap_mean`
- `dpo/margin_mean`, `dpo/margin_std`, `dpo/margin_pos_rate`
- `dpo/sigma_neg_margin_mean`, `dpo/sigma_neg_margin_min` (OFS-DPO 梯度消失警报)
- `dpo/implicit_reward_accuracy` (DPO field 标准)
- `dpo/kl_to_ref_chosen_mean`, `dpo/kl_to_ref_rejected_mean`

### V2 chain 健康 (4 个)
- `dpo/prompts_with_no_correct_pct`
- `dpo/correct_per_prompt_init_mean`
- `dpo/chain_attempt_success_rate@k` (per k in 1..n_attempts)
- `dpo/chain_edges_total`, `dpo/init_edges_total`, `dpo/chain_vs_init_pair_ratio` (hybrid 模式)

### V2.5 innovation 诊断 (6 个)
- ② `dpo/teacher_anchored_coverage`, `dpo/teacher_anchored_kl_chosen_mean`
- ③ `dpo/delta_r_max`, `dpo/delta_r_std`
- ① `dpo/t_star_mean_position`, `dpo/t_star_std_position`

## §7 Validation: DPO Implicit Reward Accuracy

代码:`ray_trainer.py:_compute_dpo_val_implicit_accuracy`

End-of-val one-shot:
1. Concat all val batches
2. Run actor.compute_log_prob (π_θ) + ref_policy.compute_ref_log_prob (π_ref) over full val batch
3. Group by uid; split correct (R ≥ threshold) vs incorrect
4. For each prompt with both: form (best_correct, worst_incorrect) pair; compute β·Σ(logπ_θ - logπ_ref); count chosen-wins

**Output**:
- `val-core/dpo_implicit_reward_accuracy` (主 val 指标)
- `val-aux/dpo_val_pairs_total`
- `val-aux/dpo_val_margin_mean`
- `val-aux/dpo_val_sigma_neg_margin_mean`

额外开销: +2 forwards per val (~5% val time)。

## §8 配置 (Hydra YAML)

[`verl/trainer/config/dpo_tgs.yaml`](../verl/trainer/config/dpo_tgs.yaml) 主要 knob 一览:

```yaml
algorithm:
  adv_estimator: dpo_teacher_guided

  dpo:
    # === V2 adaptive rollout ===
    n_init: 2
    n_attempts: 2
    correct_threshold: 1.0
    sdpo_ctx_source: sibling_correct       # | gt
    sdpo_ref_template: "Refer to this correct answer: {r}\n"
    all_failed_strategy: skip              # | gt_fallback
    exclude_tail_tokens: 8
    max_reselect_attempts: 3

    # === V1 linearized DPO ===
    beta: 0.1
    alpha: 1.0                              # 1.0=pure DPO, 0.5=mix, 0.0=pure GRPO

    # === V2.5 ① Causal-Localized ===
    causal_localize: false
    beta_token: null                        # null = use beta (推荐 2β)
    beta_continuation: null                 # null = use beta * 0.5

    # === V2.5 ② Teacher-Anchored ===
    use_teacher_anchored_ref: false

    # === V2.5 ③ ΔR-Weighted ===
    delta_r_weight_mode: none               # | linear | sqrt | squared | sigmoid
    delta_r_weight_tau: 1.0

    # === pair collection ===
    pair_strategy: chain_consecutive        # | hybrid_init_chain
    pair_margin: 0.0

    # === GRPO fallback / length protection ===
    norm_adv_by_std_in_grpo: false
    min_response_length: 50
    length_penalty_type: linear

    # === Validation ===
    val_implicit_reward_accuracy: true
```

## §9 启用方式

### 命令行 env 覆盖 (不需要改 yaml)

```bash
# 全 3 innovations 开 (论文 main):
DPO_CAUSAL_LOCALIZE=True DPO_BETA_TOKEN=0.2 DPO_BETA_CONTINUATION=0.05 \
DPO_USE_TEACHER_ANCHORED_REF=True \
DPO_DELTA_R_WEIGHT_MODE=linear \
N_INIT_LIST="2" N_ATTEMPTS_LIST="2" \
TOTAL_STEPS=250 \
bash nebula_scripts/submit_dpo_tgs_sweep.sh
```

### 单独 toggle (ablation)

```bash
DPO_USE_TEACHER_ANCHORED_REF=True ./run_notebook_dpo_tgs.sh smoke   # 仅 ②
DPO_DELTA_R_WEIGHT_MODE=linear ./run_notebook_dpo_tgs.sh smoke      # 仅 ③
DPO_CAUSAL_LOCALIZE=True DPO_BETA_TOKEN=0.2 DPO_BETA_CONTINUATION=0.05 \
  ./run_notebook_dpo_tgs.sh smoke                                    # 仅 ①
```

## §10 实施完成度 (V2.5 spec freeze: `299eec7` · 现役 HEAD: `560c761`)

| 组件 | 状态 | 文件 |
|---|---|---|
| V2 adaptive rollout | ✅ | `dpo_tgs/adaptive_rollout.py` |
| V1 线性化 DPO loss + α-mix | ✅ | `dpo_tgs/dpo_loss.py` |
| V2.5 ① Causal-Localized | ✅ | `dpo_tgs/dpo_loss.py:_build_token_localize_masks` |
| V2.5 ② Teacher-Anchored | ✅ | `dpo_tgs/adaptive_rollout.py:_post_hoc_opsd_teacher_logp` |
| V2.5 ③ ΔR-Weighted | ✅ | `dpo_tgs/dpo_loss.py:_delta_r_weight` |
| Pair collector (chain + hybrid) | ✅ | `dpo_tgs/pair_collector.py` |
| 16 SwanLab metrics | ✅ | `dpo_tgs/pair_collector.py:compute_dpo_metrics` |
| DPO val mode | ✅ | `ray_trainer.py:_compute_dpo_val_implicit_accuracy` |
| ray_trainer dispatch + bug fixes | ✅ | `ray_trainer.py` |
| Hydra config (16 knob,默认 v1 backwards-compat) | ✅ | `dpo_tgs.yaml` |
| Nebula 提交脚本 (sweep + parametric) | ✅ | `nebula_scripts/dpo_tgs/`, `submit_dpo_tgs_sweep.sh` |
| Notebook 4-GPU local smoke (smoke/innov/pair/full) | ✅ | `run_notebook_dpo_tgs.sh` |

## §11 与 OPD/Online DPO 现有工作的差异化 (论文 framing)

| vs Method | 我们的差异 |
|---|---|
| **OAIF** (online DPO baseline) | annotator 换成 verifiable env reward;独立采样换成 teacher counterfactual |
| **OFS-DPO** (Fast-Slow Chasing) | α-mix GRPO fallback 防梯度消失;v2 EMA teacher 作 slow module 可加 chase |
| **Samplers-DPO** (ICLR-25) | OPSD teacher = privileged-info-aware sampler 工程实现 → quadratic 收敛理论支撑 |
| **RPO** (最相近先前工作) | token-level vs sequence-level 信号粒度;同模型 OPSD 替代付费 GPT-4V;verifiable reward 替代 hint 质量 |
| **TCCA-Lite** (我们之前的方法) | pairwise preference 替代 advantage modulation,消除 λ_div tuning |
| **AlignDistil** | pair 来自因果反事实而非启发式 |

详见 [03_theory_anchor.md](03_theory_anchor.md) 和 [`OPD_Deep_Analysis.html`](file:///Users/ljm/lazada/papers/raw/opd_papers/OPD_Deep_Analysis.html)。
