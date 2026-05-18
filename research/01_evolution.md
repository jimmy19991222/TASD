# 01 — 来龙去脉: 从 Self-Teacher 到 DPO-TGS V2.5

> **本文是历史叙事**: 完整记录从早期 Self-Teacher Advantage 到当前 DPO-TGS V2.5 的演化路径,每一次 pivot 的动机、KILLED/弃用原因、后续修正。
> **目的**: 让读者(包括论文 reviewer)理解为什么我们最终选择当前方法。
> **当前主推**: [02_dpo_tgs_design.md](02_dpo_tgs_design.md) — DPO-TGS V2.5

---

## 演化时间线 (2026-05)

```
2026-05-07   Self-Teacher Advantage (fix_main)    ❌ KILLED — mode collapse
                       ↓ "学生自蒸馏 advantage 不稳"
2026-05-14   Prior-Shift Tier 1 (g_t Bayes surprise)  ⚠️ length collapse / val < GRPO
                       ↓ "相关性归因不够,换因果"
2026-05-15   TGDI v1 (single-step single-token replacement, share tail)  ⚠️ ΔR≡0 bug
                       ↓ "tail 复用导致 reward 不变"
2026-05-16   TCCA (top-K interventions per failed sample, chain)  ⚠️ 工程复杂度高,~2× compute
                       ↓ "复杂度跟收益不匹配"
2026-05-16   TCCA-Lite (single-step + real student tail + OPSD ctx)  ⚠️ smoke step=0 卡住
                       ↓ "advantage modulation 需要 λ_div tuning,不如换 pair learning"
2026-05-17   DPO-TGS V1 (linearized DPO + chain rollout)  ✅ 已实现
                       ↓ "n_init/n_attempts 自适应 + SDPO ctx 更可靠"
2026-05-17   DPO-TGS V2 (adaptive rollout)  ✅ 已实现
                       ↓ "vanilla DPO 没用上 teacher-guided 4 个独特结构"
2026-05-17   DPO-TGS V2.5 (3 loss innovations: ① ② ③)  ✅ 当前主推 — 4 nebula tasks 排队中
```

---

## §1 阶段 0: Self-Teacher Advantage (2026-05-07) — KILLED

### Idea
让 student 自己作为 teacher,用 EMA shadow 模型作 advantage estimator。`adv_mode=self_teacher`, `use_vce=true`。

### 失败
- **CRASHED at step 89/250**, val best 0.40 (s75)
- entropy -88%, length 346→221 (经典 mode collapse 三件套)
- EMA 几乎不工作 (shadow 立即追上 student → V_CE 退化)

### 启示
- ❌ **隐式 V_CE 退化** 是真实的: 同一个 model 既学习又评分,总会塌陷
- → 必须有**外部信号**作为 advantage 估计的锚点 (后来变成 RLSD/Prior-Shift 的 teacher forward + env reward)

---

## §2 阶段 1: Prior-Shift Tier 1 (2026-05-14)

### Idea
用 teacher forward 的 **Bayes surprise** $g_t = \text{KL}(P_T(\cdot|y_{\le t}) \| P_T(\cdot|y_{<t}))$ 作 token 级权重。
$$A_t = A_{\text{seq}} \cdot g_t / \text{mean}_t(g_t)$$

### 实验
- **v1 smoke**: `max_ratio=10`, `EPS_NORM=1e-6` → val best **0.5225** (s130) → length **372→18 collapse**
- **v2a** (max_ratio 10→3): KILLED s33,length **暴涨到 1442** (renormalize_after_clip=False)
- **v2b** (renormalize=True): val 0.5525 (s153),持续爬升,但仍未超 GRPO baseline

### 失败模式
| 失败模式 | 案例 | length | entropy |
|---|---|---|---|
| Length collapse | PS v1 (372→18), GRPO biology v2 (330→17) | 缩到极短 | -90~99% |
| Length explosion | PS v2a (1442), TGDI-gtarg (1686) | 暴涨 | -62~97% |
| Both 异常 + val 仍可高 | GRPO biology v2 val=0.66 length=17 | 4 选 1 蒙对 | -90% |

### 启示 (Why-SD-Degrades 论文统一解释)
- 三种失败都是 **epistemic verbalization** 被压制:
  - 缩短 = "Wait/Hmm" 等纠错通道被压
  - 暴涨 = procedural step ("Therefore/Step 2:") 被过度强化
  - 4 选 1 蒙对 = in-domain 短答信号噪声大,与 OOD 真实能力脱钩
- → 必须有 **length floor/ceiling + renormalize_after_clip** 双重防护
- → **相关性 g_t 不够**,需要因果信号

---

## §3 阶段 2: TGDI v1 (2026-05-15) — Single-step Intervention 数据增强

### Idea
找 token 级 **divergence 最大位置** $t^*$,teacher 接管写 k=2 token,student 续写,append 到 batch 当 contrastive sample。
$$A_{\text{seq}} \mathrel{+}= \lambda \cdot \Delta R$$ (sequence-level injection)

### 实验
3 个 t* 选择策略:
| 实验 | t* metric | 结果 | 解读 |
|---|---|---|---|
| TGDI-v3p1-aexEOS | argmax \|logp_T − logp_S\|, 排尾 8 | val 0.5737 ⭐ | 排尾 stylistic token 有效 |
| TGDI-v3p1-araw | argmax 不排尾 | val 0.4238 | 选到 EOS/punct,差 0.15 acc |
| TGDI-v3p1-gtarg | g_t argmax | KILLED s28 | g_t 与 student 错位无关 |

### Bug 暴露
- **ΔR ≡ 0**: composite y' 复用了 student 原 tail,reward 基于尾部 answer (没变),所以 ΔR 永远 0
- 这意味着 sequence-level injection $A_{\text{seq}} += \lambda \cdot \Delta R$ 完全不工作

### 启示
- ❌ 复用 tail 不行,**必须 student 真续写到 EOS** 才能产生非零 ΔR
- ❌ Sequence-level injection 信号粒度太粗,不算 "token-level credit assignment"
- → 升级到 token-level credit (TCCA)

---

## §4 阶段 3: TCCA (2026-05-16) — Token-level Causal Credit (chain rollout)

### Idea
对每个 failed sample 在 **top-K positions** $\{t_1, ..., t_K\}$ (K=3) 做 K 次独立 intervention,得 K 个 $\Delta R_{t_k}$,作 **per-token causal credit vector** $c_t$:
$$c_t[y, t_k] = -\Delta R_{t_k}, \quad c_t[y'_k, t_k] = +\Delta R_{t_k}$$
$$A_t = A_{\text{seq}} \cdot (1 + \lambda \cdot c_t) \cdot \text{length\_scale}$$

设计还包括 **chain rollout**: 8 个 sample 不再独立采样,而是串行迭代 (y_0 → y_1 → ... → y_7),每步 teacher 修 1 token + student 续写到 EOS。

### 问题
- **工程复杂度**: chain 串行 8 步,每步都需要 teacher fwd + student async 续写,代码改动 ~500 行
- **Compute**: ~2.2× baseline (chain_length=8 默认 → 15-16h/job vs 7h GRPO baseline)
- **Multiplicative formula 丢符号**: $(1 + \lambda c_t)$ 当 $c_t < 0$ 时仍保留 $A_{\text{seq}}$ 正号方向,无法真正 push down

### 启示
- → 工程复杂度跟边际收益不匹配,需要轻量级版本
- → 改 additive formula `A_seq + λ·c_t` 保符号语义

---

## §5 阶段 4: TCCA-Lite (2026-05-16 21:00) — 简化 + 修复 ΔR≡0

### Pivot 决策
| 维度 | TCCA (chain) | **TCCA-Lite** |
|---|---|---|
| rollout | 链式 8 iter 串行 | 标准 GRPO n=8 |
| intervention 时机 | 每 iter 都做 | **仅 failed samples** |
| 位置数 | 每 iter 1 个 (共 8) | **每 failed 1 个** |
| teacher 写多少 | k=1 token | k=1 token |
| student tail | 真续写到 EOS | **真续写到 EOS** (修复 ΔR≡0) |
| teacher context | OPSD (含 ref answer) | OPSD (含 ref answer) |
| 公式 | multiplicative | **additive** $A_t = (A_{\text{seq}} + \lambda \cdot c_t) \cdot \text{mask} \cdot \text{length\_scale}$ |
| compute overhead | ~2.2× | **~1.25×** |

### 实施 + 问题
- ✅ 完整代码: `intervention_credit.py`, `intervention_rollout.py`, `tcca_chain.py` (commit `4f85bde`)
- ⚠️ Local smoke 卡在 step=0 (`async_rollout_manager.server_manager.generate` API 没用过,可能是 init 问题)
- ⚠️ 需要 tune `λ_div ∈ {0, 0.5, 1.0, 2.0}` ablation,advantage 层面调参

### 启示
- TCCA-Lite 在 advantage 层面注入 $\Delta R$,**仍需 hyperparameter (λ_div) tuning**
- → 思考: 同样的 rollout 数据是否可以走 **不同的 learning signal** (DPO 替代 advantage modulation)?

---

## §6 阶段 5: DPO-TGS V1 (2026-05-17 早期) — Pairwise preference

### 关键洞察
TCCA-Lite 的 chain rollout (或 single-step counterfactual) **天然产生 (chosen, rejected) pairs**:
- $y^+ = y[:t^*] + z_T + \text{student\_continue}$ (teacher 修后 R 改善)
- $y^- = y$ (原 failed)
- $R(y^+) > R(y^-)$ → 直接形成 chosen/rejected pair

→ 不需要 ΔR modulation,不需要 λ_div 调参,**直接用 DPO loss**:
$$\mathcal{L}_{\text{DPO}} = -\log \sigma\left(\beta \log \frac{\pi_\theta(y^+|x)}{\pi_{\text{ref}}(y^+|x)} - \beta \log \frac{\pi_\theta(y^-|x)}{\pi_{\text{ref}}(y^-|x)}\right)$$

### V1 实施 (线性化 DPO)
为避免 actor surgery (proper DPO 需 pair-aware micro-batching),把 DPO 梯度方向**编码为 per-token advantage**,复用 PPO surrogate:
- $\text{margin} = \beta \sum_t \text{mask}_t (\log \pi_{\text{old}} - \log \pi_{\text{ref}})$ for (chosen, rejected)
- $g = \beta \cdot \sigma(-\text{margin})$
- $A_t[\text{chosen}] = +g/L_{\text{chosen}} \cdot \text{mask}$, $A_t[\text{rejected}] = -g/L_{\text{rejected}} \cdot \text{mask}$
- pair-free 样本 fallback 到 GRPO group-relative (α-mix 防梯度消失)

### 启示
- ✅ **零 actor 改动**,可立即复用 verl PPO 框架
- ✅ 第一阶 DPO 近似 (在 $\pi_\theta = \pi_{\text{old}}$ 处精确,PPO ratio clip 控制偏离)
- 但: chain rollout 仍是固定 chain_length 串行,效率不高 → 进入 V2 adaptive rollout

---

## §7 阶段 6: DPO-TGS V2 (2026-05-17 中期) — Adaptive Rollout

### 核心改动
**3 个用户驱动的 design 升级** ([见对话历史](archive/dpo_tgs_v1_design.md) "V2.5 Update" 章节):

| Pivot | v1 | **V2** |
|---|---|---|
| **采样基底** | chain 串行 (chain_length 个 sample) | **GRPO n_init=2/4 i.i.d. + 失败才干预** |
| **Teacher ctx** | OPSD with dataset GT (`"The correct answer is {GT}.\n"`) | **SDPO with sibling correct rollout** (`"Refer to this correct answer: {sibling_correct}\n"`) |
| **Token mismatch** | argmax 后不查 z_T == y_S[t*] | **强制 z_T ≠ y_S[t*],等于就 reselect_t** (max 3 次) |
| **停止条件** | 固定 chain_length-1 步 | **每 failed sample 跑 n_attempts 次干预,带 lineage tagging** |

### Pipeline
```
Phase 1: standard rollout n_init=2 per prompt → y_init (B × n_init samples)
Phase 2: per-prompt SDPO teacher ctx
   - sibling correct rollout 作 ref (无 GT 依赖)
   - 全 failed → 默认 skip (ablation 可 gt_fallback)
Phase 3: 对每个 failed sample 迭代干预
   For attempt 1..n_attempts=2:
     a. teacher fwd (SDPO ctx) → divergence
     b. argmax divergence,排尾 8 + 排 used_positions
     c. teacher decode 1 token z_T,enforce z_T ≠ y_S[t*]
     d. student async continue
     e. y_attempt = chain[i-1][:t*] + z_T + continuation
Phase 4: concat y_init + 所有 chain attempts
   tag dpo_lineage_id + dpo_attempt_idx + dpo_t_star
```

### 启示
- ✅ 选择性干预 → 节省 compute (correct samples 不浪费)
- ✅ self-distillation 更纯粹 (无 GT 依赖)
- ✅ Token mismatch 避免 degenerate intervention (silent ΔR=0)
- ✅ Lineage tagging 让 pair_collector 在共享 prefix 的 chain 里正确分组

---

## §8 阶段 7: DPO-TGS V2.5 (2026-05-17 晚期) — Loss Innovations 当前主推

### 起源
回答用户的提问:**"DPO loss 在 teacher-guided 设定下有没有创新空间?"**

观察: vanilla DPO 没用上 teacher-guided pair 的 4 个独特结构:
1. **共享 prefix**: $y^+$ 和 $y^-$ 在 $t^*$ 之前完全相同 → log-ratio 在前缀上严格 = 0
2. **因果定位**: 唯一确定性差异是 $t^*$ 一个 token (continuation 是 student 自由生成)
3. **OPSD teacher 完整分布**: 我们知道 $P_T^{\text{OPSD}}(\cdot|y_{<t^*})$,vanilla DPO 没用
4. **连续 ΔR**: 不只是偏好符号,有真实 reward 差值

→ 每条结构对应一个 loss innovation。3 个独立可 toggle 的 knob (默认 OFF,backwards compat 保留 v1):

### ① Causal-Localized DPO (`causal_localize: true`)
利用结构 (1)(2). 把 margin 拆成 token-level (only t*) + continuation-level:
$$\text{margin} = \beta_{\text{tok}} \cdot \Delta\text{logp}_{t^*} + \beta_{\text{cont}} \cdot \Delta\text{logp}_{>t^*}$$

### ② Teacher-Anchored DPO (`use_teacher_anchored_ref: true`)
利用结构 (3). 把 OPSD teacher 当第三个 reference 替代 $\pi_{\text{ref}}$:
$$\mathcal{L} = -\log \sigma\left(\beta \log \frac{\pi_\theta(y^+) \cdot \pi_T^{\text{OPSD}}(y^-)}{\pi_\theta(y^-) \cdot \pi_T^{\text{OPSD}}(y^+)}\right)$$
理论支撑: **Samplers-DPO ICLR-25 Theorem 4** — reward-aware ref ⇒ quadratic 收敛。

### ③ ΔR-Weighted DPO (`delta_r_weight_mode: linear`)
利用结构 (4). 用 verifiable reward 差值 $\Delta R$ 给 pair 加权:
$$\mathcal{L} = -W(\Delta R) \cdot \log \sigma(\beta \cdot \text{margin}),\ W \in \{|\Delta R|, \sqrt{|\Delta R|}, \Delta R^2, \sigma(\Delta R/\tau)\}$$
RPO 的 $W_{\text{hal}}$ 启发式权重的 verifiable-reward 严格化。

### 当前状态
- ✅ 全部实现 (commit `7afa10f`)
- ✅ 3 个独立 toggle,8 种组合 (2³),默认 v1 backwards compat
- ✅ V2.5 spec freeze: `299eec7`
- 📋 完整 spec → [02_dpo_tgs_design.md](02_dpo_tgs_design.md)

---

## §8.5 阶段 8: 稳定化 (2026-05-18) — 工程鲁棒化,无方法变更

V2.5 spec freeze 后,进入 Nebula + 本地 notebook 实跑。第 3 批跑起来命中**新的 FSDP chunk divisibility bug** (teacher forward 时 batch size 不能被 fsdp world size 整除直接 crash),`8512de9` 修了 chunk padding。本地 notebook 启动又踩到一连串 ray init / NCCL bootstrap 坑。整个稳定化 9 个 commit (`0891353` → `560c761`),**没有任何方法学改动**,只是让代码在真实部署环境跑得起来。

| Commit | 修复 |
|---|---|
| `0891353` | non_tensor merge 替换 batch.union(gen_batch) |
| `b06634d` | 加 tiny 模式 (1-GPU 极简 smoke,3 step bs=2) |
| `c4b7f46` | re-attach prompt_batch non_tensor to y_init after _standard_rollout |
| `a35d580` → `b3b4d65` | ray init 鲁棒化 (timeout / log buffering / loopback IP / 后台启动) |
| `8512de9` | **FSDP chunk divisibility padding** (Nebula main blocker) |
| `560c761` | NCCL bootstrap 走 lo + ray IP detection 容错 (notebook 单机) |

第 4 批 nebula 重交 (commit `560c761`) 见 [04_experiments.md §2 第 4 批](04_experiments.md#§2-已提交-nebula-任务时间线)。

### 当前状态 (2026-05-18 23:12)
- 🔄 第 4 批 3 nebula tasks 排队中 (commit `560c761`)
- ⏳ 本地 notebook tiny 模式待用户验证 NCCL fix (commit `560c761` 是首次端到端跑通的版本)

---

## §9 横切关注: bugs 时间线

| 阶段 | Bug | 发现于 | 修复 |
|---|---|---|---|
| TGDI v1 | tail 复用 → ΔR ≡ 0 | smoke metric 全 0 | TCCA-Lite student 真续写到 EOS |
| Prior-Shift v1 | length collapse 372→18 | 训练 trajectory | v2 max_ratio=3 + length_floor + renormalize |
| TCCA-Lite | smoke step=0 卡住 | local smoke | 未完全诊断,DPO-TGS pivot 后绕过 |
| DPO-TGS V2.5 (commit `2ea25d9`) | sdpo_ref_template 含 `{r}`,Hydra 解析失败 | nebula log | commit `5d5bfdf` 移除 cli 传值 |
| DPO-TGS V2.5 (commit `5d5bfdf`) | gen_batch 缺 reward_model → KeyError | nebula log | commit `299eec7` pass full batch |
| 同上 | `logp_actor` undefined (NameError) | bug 审计 | commit `299eec7` 改 `logp_old` |
| 同上 | B=1 edge case `t_i.numpy()` 0-d | bug 审计 | commit `299eec7` `np.atleast_1d` |
| 同上 | `_vectorized_pair_advantage` 用 `mask.dtype` (long),integer 截断 | bug 审计 | commit `299eec7` 强制 `torch.float32` |
| DPO-TGS V2.5 (commit `299eec7`) | teacher forward batch 不能被 FSDP world_size 整除 → crash | 第 3 批 nebula log | commit `8512de9` 加 chunk padding |
| 同上 (本地 notebook) | ray.init() in-process 5 min timeout,/tmp/ray_start.log 空 | 用户首次跑 tiny | commit `ada178b` shell `ray start --head` 后 attach |
| 同上 | NCCL bootstrap 卡 `connect to eth1:port` 30s × 34 retry | 用户日志 | commit `560c761` 强制 `NCCL_SOCKET_IFNAME=lo` |
| 同上 | 脚本静默 exit (无 cluster up 行) | 用户日志 | commit `560c761` ray IP detection grep `\|\| true` |

---

## §10 关键学习

| Lesson | 当前方法如何 honor |
|---|---|
| 不能让 student 自评 (Self-Teacher 死) | DPO-TGS 的 reference 是 frozen π_ref / EMA teacher / OPSD teacher,不是 self |
| 必须 length floor + renormalize | DPO-TGS 保留 `min_response_length` + `length_penalty_type` |
| Tail 复用导致 ΔR=0 | adaptive_rollout 的 student 真续写到 EOS |
| Multiplicative 丢符号 | DPO loss 本身 sigmoid-σ-gating 保符号 |
| 工程复杂度跟收益要匹配 | V1 线性化 DPO 复用 PPO surrogate,零 actor 改动 |
| advantage 层调参 (λ_div) 复杂 | DPO-TGS 用标准 β,无 λ_div |
| Vanilla DPO 没用上 teacher-guided 结构 | V2.5 三个独立 innovation 各对应一条结构 |

---

## 当前结论

**DPO-TGS V2.5 是 OPD literature 上的下一步**:
- 用 token-level causal counterfactual 升级 OPD 的 token-level signal (从 distribution matching → 启发式 → 统计 evidence ratio → **真因果反事实 pair**)
- 用 pairwise preference learning 替代 advantage modulation,消除 λ_div 调参
- 用 5 篇 Online DPO 论文 (OAIF + OFS-DPO + Samplers + RPO + Meta) 提供完整理论锚点
- 用 3 个 loss innovation (Causal-Localized + Teacher-Anchored + ΔR-Weighted) 利用 teacher-guided 设定的独特结构

下一步: 等 nebula 4 个 task 跑完出主结果。详见 [04_experiments.md](04_experiments.md)。
