# Teacher-Guided Dynamic Intervention Rollout → **TCCA (Token-level Causal Credit Assignment)**

> **状态**：2026-05-16 升级为 TCCA (top-K interventions per failed sample → per-token causal credit)
> **关联**：[Prior-Shift](file:///Users/awesome_jimmy/lazada/SDPO/verl/trainer/ppo/bayesian_credit/prior_shift.py) (ablation) / [实验进展报告](file:///Users/awesome_jimmy/lazada/SDPO/实验进展报告.md)
> **更新**：2026-05-16 19:00

---

## 0. TCCA 升级说明 (2026-05-16)

旧 TGDI (Phase 2 V1) 只在单点 t\* 做 intervention，本质是"数据增强 + 微弱 seq-level advantage tweak"，**不算 token-level credit assignment 创新**。

升级为 TCCA：对每个失败 sample 在 **top-K 位置**分别做 intervention，得到 K 个独立 ΔR_{t_k}，作为 **per-token causal credit**：

```
对失败 sample y:
  1. 选 top-K divergence 位置 {t_1, ..., t_K} (典型 K=3)
  2. for k in 1..K:
       teacher 在 t_k 改写 intervention_length_k 个 token → composite y'_{t_k}
       ΔR_{t_k} = R(y'_{t_k}) - R(y)
  3. 构造 per-token causal credit c_t:
       For 原 sample y: c_t = -ΔR_{t_k} at t = t_k, 0 elsewhere
       For composite y'_{t_k}: c_t = +ΔR_{t_k} at t = t_k, 0 elsewhere
  4. token-level advantage:
       A_t = A_seq · base_reweight · (1 + λ_token · c_t_normalized) · length_scale
```

K 个 composite 全部 append 到 batch (Mode B append × K)，GRPO group advantage 自然处理。

**与 TGDI 相比的核心区别**：
| | TGDI Phase 2 V1 | TCCA |
|---|---|---|
| 干预位置数 | 1 (单 t\*) | **K (top-K)** |
| 信号粒度 | seq 级 scalar ΔR | **token 级 vector c_t** |
| credit 类型 | 数据增强 + seq 注入 | **真 token 级 causal credit** |
| 论文 contribution | 数据增强 | **新 credit assignment 方法** |

详见本文末"附录: TCCA 实现 spec"。

---

---

## 1. 演化路径（历史叙事）

```
2026-05 早期         Self-Teacher Advantage (fix_main) — KILLED (mode collapse)
        ↓
2026-05-14           Prior-Shift Tier 1 (g_t forward Bayesian surprise)
                     A_t = A_seq · g_t / mean_t(g_t)
                     ❌ v1 length collapse (372→18); v2b val 0.55 < GRPO 0.66
        ↓
2026-05-15           TGDI 设计 (本文档原内容):
                     单点 t* + teacher 接管 k token + Mode B append
                     A_seq += λ · ΔR (seq 级注入)
                     ⚠️ 本质是数据增强, 不是 token-level credit 创新
        ↓
2026-05-16 19:00     **TCCA 升级 (当前)**:
                     top-K positions per failed → K 个独立 ΔR_{t_k}
                     c_t = ±ΔR_k 写到 token 级
                     A_t = A_seq · base_reweight · (1 + λ · c_t)
                     ✅ 真 token-level causal credit assignment
```

完整方法 + 公式 + pipeline → [paper_idea.md](paper_idea.md)
当前实验状态 + baseline 对比 → [experiment_progress.md](experiment_progress.md)

---

## 2. TGDI 原始 motivation（保留作历史参考）

### 问题

Prior-Shift (Tier 1) 的 credit 是**纯相关性**的：
```
A_t = A_seq · ĝ_t   (ĝ_t 来自 teacher 的 forward Bayes surprise)
```

它回答"teacher 在哪个位置感到惊讶"，**而不是** "student 在哪个 token 犯了错"。
v1 smoke 实验暴露了这种相关性归因的脆弱性——max_ratio 过大时，credit 分配失准导致 length collapse (372→18 token)。

### 核心洞察

**Student 的错误本质上是某个 token 的选择偏离了 teacher 的认知**。如果能精确找到分歧最大的位置，让 teacher 在那个点上接管生成，student 就能看到：

> "如果我在 t\* 处采用了 teacher 的建议，后续会怎样？"

这就是 **causal intervention** — 不是猜测哪个 token 重要，而是直接做实验验证。

### TGDI vs TCCA 的关键差异

| | TGDI (legacy) | TCCA (current) |
|---|---|---|
| 干预位置数 | 1 (单 t\*) | K (top-K) |
| 信号粒度 | seq 级 scalar ΔR | token 级 vector c_t |
| 公式 | A_seq += λ · ΔR | A_t · (1 + λ · c_t) |
| credit 类型 | 数据增强 + seq 注入 | 真 token 级 causal credit |

---

## 3. 与既有方法的关系（精简版）

详细对比表 → [paper_idea.md §6](paper_idea.md)。简版：

- **vs OPSD / SDPO**：他们用 distribution matching (有 leakage, 见 RLSD Theorem 1)；我们用环境 reward 锚定方向、ΔR 作为 magnitude
- **vs RLSD**：他们用统计学 log-ratio 估计 token 重要性；我们用真实因果 ΔR_t
- **vs TIP**：他们用启发式 (entropy, divergence) 选 token；我们用 ΔR_t 直接测量
- **vs Prior-Shift (我们的 ablation)**：correlational g_t vs causal ΔR，论文展示后者优

---

## 附录：原 TGDI 文档主要章节归档

原文档 506 行的详细内容已被以下文件取代：

| 原章节 | 现位置 |
|---|---|
| §2 核心算法 | [paper_idea.md §4 完整 pipeline](paper_idea.md) |
| §3 数学公式 | [paper_idea.md §3.2](paper_idea.md) + 代码 [intervention_credit.py](../verl/trainer/ppo/bayesian_credit/intervention_credit.py) |
| §4 开销分析 | TCCA 升级后已变化，新数据见 [experiment_progress.md](experiment_progress.md) |
| §5 工程实现 | 直接看代码：[intervention_rollout.py](../verl/trainer/ppo/bayesian_credit/intervention_rollout.py) + [intervention_credit.py](../verl/trainer/ppo/bayesian_credit/intervention_credit.py) + [dp_actor.py](../verl/workers/actor/dp_actor.py#L887) |
| §6 实验设计 | [experiment_progress.md §7 待跑实验](experiment_progress.md) |
| §7 Bayesian Credit Assignment 叙事 | 已弃用（替换为 TCCA framing）, 见 [paper_idea.md](paper_idea.md) |
| §8 关键设计决策 | 已全部落地，见代码 + [paper_idea.md §5](paper_idea.md) |
| §9 Related Work | [paper_idea.md §2 OPD 演化表](paper_idea.md) |

如需查看原 TGDI 完整设计（500+ 行细节），可 `git log -- research/design_history.md` 找历史版本。
