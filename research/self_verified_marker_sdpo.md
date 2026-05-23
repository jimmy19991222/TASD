# Self-Verified Marker SDPO

> **TL;DR** — 在 OPSD 框架下,用 student 自己的"我已经做对了"自我标注(一个固定 marker token,如 `"This answer is verified correct."`)代替 privileged reference solution,作为 teacher 的额外 conditioning。理论上这一改动把 SDPO 的 ΔW 信号从 "answer-dependent + reference-conditioned" 变为 **"answer-independent + self-conditioned"**,从而(1)解除对 sibling rollout success 的依赖,(2)缓解 reference 风格漂移,(3)恢复 token-level ΔW 的纯 value 解释。本文给出 motivation、Bayes 推导、与 vanilla SDPO / VC-OPSD 的等价性边界,以及 3 个 marker 实验的 hypothesis 与判读标准。

- **作者**: 月丘 + Claude (Opus 4.7)
- **日期**: 2026-05-23
- **状态**: 已落地实现,sweep 待回收
- **相关代码**:
  - 配置: [`verl/workers/config/actor.py`](../verl/workers/config/actor.py) (`SelfDistillationConfig.teacher_context_mode / self_verified_marker / overconfidence_damping`)
  - 数据装配: [`verl/trainer/ppo/ray_trainer.py:694-857`](../verl/trainer/ppo/ray_trainer.py#L694-L857) (`_maybe_build_self_distillation_batch`)
  - Loss: [`verl/trainer/ppo/core_algos.py`](../verl/trainer/ppo/core_algos.py) (`compute_self_distillation_loss`,ΔH damping 块)
  - Sweep: [`nebula_scripts/submit_sdpo_marker_sweep.sh`](../nebula_scripts/submit_sdpo_marker_sweep.sh)
- **前置阅读**: [`research/verdict_credit_assignment.md`](verdict_credit_assignment.md), [`research/reward_bayes_distillation_v2.md`](reward_bayes_distillation_v2.md), [`research/sdpo_loss_bayes_reading.md`](sdpo_loss_bayes_reading.md)

---

## 0. Motivation:为什么要 marker 而不是 ref

Vanilla SDPO 在 OPSD 框架下的 teacher conditioning 是

```
teacher = π_θ(· | system, question, "Correct solution: " + ref_solution_y*, prompt)
```

即把 verifier 认可的 sibling rollout(或 ground-truth 答案)拼到 user turn 上重 prompt。这是工程上行之有效的 strong baseline,但在 OPSD 框架下(teacher 和 student 共享 weight)有三个理论与实操上的痛点:

### 0.1 痛点 1:Sibling 依赖导致信号稀疏

`_collect_solutions_by_uid` 要求 group 内**至少一条** rollout 满足 `reward >= success_threshold`。这就是当前训练日志里 `self_distillation/success_group_fraction` 这个指标的来源——任务越难,这个 fraction 越低,有效 distill batch 越小。当前 8B 模型在 sciknoweval/biology 上 success_group_fraction 普遍 < 0.5,意味着一半 group 拿不到 distill 信号。要么(a)涨 rollout_n 兜底,(b)接受信号稀疏。

### 0.2 痛点 2:Reference 风格漂移污染 ΔW

`ref_solution` 是从某条 sibling rollout 取的(或 ground-truth),它的**写作风格**(中英混杂、$\LaTeX$ 习惯、是否出现 `<think>` block)与当前 student rollout 不同。在 OPSD `p_T` 与 `p_s` 共享 weight 的前提下,teacher 看到 ref 之后 forward 当前 response,前缀风格 mismatch 会让 `log p_T(y_t)` 偏低**不是因为内容错**而是**因为前缀风格不匹配**。这部分噪声直接进入 `ΔW_t = log p_T - log p_s`,污染 token-level credit。

### 0.3 痛点 3:答案依赖让 ΔW 失去 value 解释

[`research/verdict_credit_assignment.md`](verdict_credit_assignment.md) §3.4 的关键结论是:**OPSD 下 `ΔW_t` 之所以能折叠成"纯 value 信号"是因为 teacher 和 student 都看到了某个 answer-independent 的条件**。一旦 ref 是 `argmax_y* p_T(y* | s_<t)` 的相关量(比如来自 sibling rollout 的 best-of-N),ref 就变成了 answer-dependent,ΔW 里就会重新混入 likelihood 项,Bayes 等价性被打破。Marker 是一个**fixed string**,与具体 answer 无关,因此 ΔW 恢复 pure value 解释。

---

## 1. Marker Mode 定义

把 teacher 的 conditioning 从 "ref" 换成:

```
teacher_prompt =
  system + user(question) + assistant_header + "This answer is verified correct.\n\n"
```

然后把 student response token 接在 marker 之后做 teacher forward。这就是 [`ray_trainer.py:795-809`](../verl/trainer/ppo/ray_trainer.py#L795-L809) 的实现:

```python
marker_ids_1d = self.tokenizer.encode(marker_text, add_special_tokens=False)
teacher_input_ids = torch.cat([teacher_prompt_ids, marker_ids, responses], dim=1)
teacher_attention_mask = torch.cat([teacher_prompt_mask, marker_mask, response_mask], dim=1)
```

slicing 仍然走标准 `[:, -response_length - 1 : -1]`(由 `_forward_micro_batch` 用 `responses.size(-1)` 推断),marker 自动被吸收进 "prompt-like prefix",不污染 response position。

三种 mode(`actor_rollout_ref.actor.self_distillation.teacher_context_mode`):

| Mode | Teacher sees | 用途 |
| --- | --- | --- |
| `ref` | system + question + "Correct solution: <ref>" + prompt | 当前 baseline |
| `marker` | system + question + assistant_header + marker + responses | 纯 self-distill,无 ref 依赖 |
| `ref_and_marker` | ref 拼到 user turn,marker 拼到 assistant turn | 双信号叠加,验证可加性 |

---

## 2. Bayes 推导:marker 下 ΔW 的真实含义

记 student 的 generation 分布为 `p_s(y_t | s_<t) = π_θ(y_t | x, y_<t)`,其中 `x = system + question`。marker mode 下 teacher 是

```
p_T(y_t | s_<t, m) = π_θ(y_t | x, m, y_<t)
```

其中 `m = "This answer is verified correct."` 是一个 verdict-style 条件。

按 Bayes:

$$
\log \frac{p_T(y_t \mid s_{<t}, m)}{p_s(y_t \mid s_{<t})}
= \log \frac{\pi_\theta(y_t \mid x, m, y_{<t})}{\pi_\theta(y_t \mid x, y_{<t})}
= \log \frac{p(m \mid x, y_{\le t})}{p(m \mid x, y_{<t})}
$$

(用 conditional Bayes,记 `p` 为模型隐式 joint。)

定义 `c_t := \log p(m \mid x, y_{\le t})`(模型在"看到前 t 个 student token 后,主观相信 marker 成立"的对数信念),则:

$$
\Delta W_t = c_t - c_{t-1}
$$

— **`ΔW_t` 等于 token t 对 "marker 成立信念" 的边际贡献**。当 student 写出一个让模型自己更相信"我做对了"的 token,ΔW_t > 0,SDPO 把 student 往这个方向拉。这正是我们想要的 retrospective credit signal,**且完全不依赖 sibling/ref**。

⚠️ 注意:这个 calibration 只在 `p_T` 和 `p_s` 共享 weight(OPSD)时严格成立。传统 OPD(独立 teacher)下 `c_t` 是 teacher 网络的信念,与 student 行为脱钩,等价性不再 trivially 成立。这一点延续 [`verdict_credit_assignment.md` §3.4](verdict_credit_assignment.md#34) 的 framing。

### 2.1 严格性 caveat:两条隐含假设

上述推导依赖两条没有显式说出来的假设,有必要单独标出来:

**(A) Shared-prefix 约定**。本节里 `s_<t` 指 teacher 和 student **共享**的 prefix `(x, y_<t)`,$m$ 是 teacher 独占的额外 conditioning。
- 在 **marker mode** 下,这成立:student 看到 `(x, y_<t)`,teacher 看到 `(x, m, y_<t)`,差集恰好是 fixed marker $m$。
- 在 **ref mode** 下,teacher prefix 实际是 `(x, "Correct solution:", ref, y_<t)`,直接套 §2 的拆解会把 $m := \text{ref}$。此时 $c_t = \log p(\text{ref} \mid x, y_{\le t})$ **不再是 "verdict 信念"**,而是 "在见过 student 的前 t 个 token 后,模型相信会出现这一整段 ref 文本的对数似然"。ref 是 answer-dependent(它本身就是另一条解答),pure value 解释**直接打破** —— 这正是 §0.3 论的事。

**(B) Prefix-order invariance**。Bayes 翻转

$$
\pi_\theta(y_t \mid x, m, y_{<t}) = \pi_\theta(y_t \mid x, y_{<t}) \cdot \frac{p(m \mid x, y_{\le t})}{p(m \mid x, y_{<t})}
$$

要求 $\pi_\theta$ 隐式定义的 joint 对 $m$ 和 $y_<$ 的**出现顺序无关**。Left-to-right transformer + 位置编码让 `[x, m, y<]` 和 `[x, y<, m]` 严格不是同一个输入,所以这条**严格不成立**,只是经验近似。
- $m$ 是 **短 fixed string**(marker)时,position offset 是常数,近似良好。
- $m = \text{ref}$(上百 token 的另一段 answer)时,offset 大、内容长,近似很弱 —— 这是 ref mode 下 ΔW 会有 "风格漂移噪声" 的根本来源。

结论:§2 的"ΔW = verdict 信念边际贡献"这条 clean 解释**只对 marker mode 严格**,对 ref mode 是 degrade 的近似。这恰好支撑 §0 的三条 motivation。

---

## 3. 与 VC-OPSD / OPD-Bayes 的关系

[`reward_bayes_distillation_v2.md`](reward_bayes_distillation_v2.md) 和 [`vc_opsd_paper_draft.md`](vc_opsd_paper_draft.md) 提出的 verdict-conditioned 方案(`reward_bayes_v2`)是:**两次** teacher forward,分别在 prompt 头部加 `right`/`wrong` marker,然后取 log-prob 差值 `Δ_t = log p_T^+ - log p_T^-` 作为 contrastive credit。本文的 marker SDPO 是这个想法的**单边 + 后置**版本:

| 维度 | VC-OPSD (`reward_bayes_v2`) | Marker SDPO (本文) |
| --- | --- | --- |
| 信号方向 | contrastive (right − wrong) | one-sided (verified correct only) |
| 信号位置 | prepended to **user turn** | inserted at **assistant turn** start |
| Teacher forwards | 2× per batch | 1× per batch (与 vanilla SDPO 同) |
| 训练对象 | weighted PG / OPD-Bayes loss | 标准 SDPO KL(`compute_self_distillation_loss`) |
| 依赖 R | 是(decide sign) | **否**(仅用 student rollout) |
| 算力开销 | +1× teacher forward | 0 额外开销 |

观察:**marker SDPO 是 VC-OPSD 单边化后并入 vanilla SDPO loss pipeline 的最小改动版本**。它的优势是工程成本接近 0(只改 conditioning,loss 完全复用),劣势是丢失了 contrastive 的对称性,可能受 marker 本身的 prior bias 影响。两条路径互补,不冲突。

### 3.1 vs Zhao et al. 2026 (OPSD paper, arXiv:2601.18734)

发现一篇直接同名的工作:**Self-Distilled Reasoner: On-Policy Self-Distillation for Large Language Models** (Zhao, Xie, Liu, Huang, Pang, Chen, Grover; UCLA + HKU + Meta; 2026年3月)。论文的 OPSD 框架与本 codebase 基本是同一回事:

| 维度 | Zhao et al. 2026 OPSD | 本 codebase |
| --- | --- | --- |
| 同一 LLM 当 teacher/student | ✅ | ✅ |
| Teacher 看到 y\*(GT 解答)作为 privileged info | ✅ | ✅ (`ref` mode) |
| Student 只看 problem | ✅ | ✅ |
| 在 student rollout 上算 per-token divergence | ✅ | ✅ |
| 梯度只回流 student logits | ✅ | ✅ |
| Full-vocab JSD/KL + 逐 token clipping | ✅ (Eq. 6-7, "Per-Token Pointwise Divergence Clipping") | ✅ (`full_logit + is_clip=2`) |
| Sampled-token PG fallback $A_n = \log p_T - \log p_s$ | ✅ (Eq. 9) | ✅ (ΔW token-PG 分支) |
| Teacher prompt 把 ref 放 user turn,要求"先理解再 rewrite" | ✅ (Figure 2) | ✅ (`"Correct solution: " + ref + prompt`) |

**结论:本 codebase 的 `ref` mode 实质上是 OPSD paper 的实现。**

论文**未覆盖**(本工作的增量)的部分:

1. **`marker` mode(answer-independent 自验证标记)** — 论文必须有 y\* 才能 condition teacher,我们的 marker 模式**完全不用 GT**,只靠 fixed verdict string,把 ΔW 的 "verdict 信念边际贡献"解释推到最纯粹。这是论文框架直接缺失的一种 conditioning。
2. **ΔW = c_t − c_{t-1} 的 Bayes 解释** — 论文把 OPSD 当 distillation 训练目标讲,**没有**把 per-token divergence 拆解成 "marker 信念的边际贡献" 这种 retrospective credit 视角。本文 §2 是新框架。
3. **ΔH overconfidence damping** — 论文不涉及。
4. **VC-OPSD sign-flip / verdict contrastive** — 论文不涉及(见 [`reward_bayes_distillation_v2.md`](reward_bayes_distillation_v2.md))。
5. **gt_marker(assistant-turn 前置 GT)** — 论文精神上覆盖(都把 GT 喂给 teacher),但 prompt 位置不同(论文 user turn,我们 assistant turn 前缀)。这是工程层面变体,理论 novelty 弱,留作可选 ablation。

实操意义:
- **`ref` baseline 必须保留** — 这是 Zhao et al. 同等方法的基准,任何 claim 都要先回到这上面 anchor。
- **`marker` / `marker_damp` / `refmarker` 三个变体都是论文未覆盖的方向**,sweep 数据出来可以直接 vs OPSD 作论文级 baseline 比较。
- Research note 与论文的 framing 切口不同:论文从 "rationalization 角度" 讲(给 teacher 看 GT 是为了让它理解后 rewrite),本文从 "Bayes verdict-belief 角度" 讲(condition 是为了让 ΔW 折叠成 c_t − c_{t-1})。互补,不冲突。

---

## 4. 实验设计 (3 variants)

所有变体共享 baseline:`JSD (alpha=0.5) + full_logit + distillation_topk=100 + tail bucket + EMA teacher + is_clip=2`,与 [`submit_loss_variants_sweep.sh`](../nebula_scripts/submit_loss_variants_sweep.sh) 的 JSD baseline 完全对齐。只改 `teacher_context_mode` 和 `overconfidence_damping`。

| Tag | `teacher_context_mode` | `overconfidence_damping` | 目的 |
| --- | --- | --- | --- |
| `marker` | `marker` | 0.0 | 纯 marker self-distill,**直接 vs vanilla SDPO** 验证 §0 三个痛点是否被解决 |
| `marker_damp` | `marker` | 1.0 | marker + ΔH 过自信抑制,验证 entropy-collapse 修正 |
| `refmarker` | `ref_and_marker` | 0.0 | 双信号叠加,**消融**:marker 是否对 ref 有 incremental gain |

### 4.1 ΔH 过自信抑制 (`overconfidence_damping`)

Marker 是一个非常 confident 的 prior(模型被告知"答对了"),会让 `H(p_T)` 显著低于 `H(p_s)`。在 reverse-KL / JSD 下,这种 entropy gap 会被 KL 过度放大,导致 student 被推向 sharp 的分布,加剧 mode collapse 风险。Damping 公式:

$$
w_t = \exp\!\left( -\frac{\max(0, H_s - H_T)}{\tau} \right), \quad \tau = \text{overconfidence\_damping}
$$

- 当 `H_s ≈ H_T`(student 已经足够自信),`w_t ≈ 1`,正常更新。
- 当 `H_s ≫ H_T`(student 还在探索但 teacher 因 marker 已锁定一个 token),`w_t → 0`,跳过这个 token,避免 marker 强行让 student 模仿过于 sharp 的 teacher。

实现见 [`core_algos.py` ΔH damping 块](../verl/trainer/ppo/core_algos.py),仅在 `full_logit_distillation=True` 时生效,因为需要完整 vocab 上的 entropy。

诊断指标:`actor/damp_weight_mean / damp_weight_min / damp_gap_mean / damp_H_student_mean / damp_H_teacher_mean / damp_tau`。

### 4.2 判读标准 (优先级降序)

1. **`marker` vs vanilla SDPO baseline 的 `val-core/sciknoweval/acc/mean@16`**:这是核心 hypothesis 的 go/no-go。
   - 若 `marker` 持平或优于 vanilla SDPO,则证明 ref 信号可以被 marker 替代,SDPO 对 sibling success 的依赖可解除。
   - 若 `marker` 明显劣化(> 1pp),则 ref 的具体内容确实贡献了 distill signal,marker 不足以替代。
2. **`marker_damp` vs `marker`**:验证 ΔH damping 是否缓解 marker 引入的 entropy collapse。看 `actor/entropy`、`response_length/mean`、`actor/grad_norm`。
3. **`refmarker` vs `marker` 和 `ref`**:加性消融。若 `refmarker > max(marker, ref)`,marker 提供了 ref 没覆盖的正交信号;若 `refmarker ≈ ref`,marker 信号被 ref 完全 dominate。
4. **`actor/delta_w_snr_mean` 跨 variant 比较**:marker 应当让 ΔW SNR 更高(reference 风格漂移噪声被消除)。SNR ↑ 是理论 prediction 的直接 check。
5. **`self_distillation/marker_active_fraction`**:常驻为 1.0(sanity check,确认 marker 模式被正确激活)。

### 4.3 Risk & 退路

| Risk | 触发条件 | Mitigation |
| --- | --- | --- |
| marker 让 student 学会"自我洗白"(写一些让模型相信自己对、但实际错的话术) | val acc 不涨甚至下降,但 train 端 `c_t` 系数变大 | 加 calibration penalty `(p_T(m|y) - R)^2`,或回退 `refmarker` |
| ΔH gap 过大 token 比例 ≫ 50%,damping 把信号几乎清空 | `damp_weight_mean < 0.3` | 加大 `τ`(2.0~4.0)或回退 `marker` no-damp |
| Tokenizer 把 marker 切成奇怪的 subword,导致 position alignment 错位 | early step `actor/pg_loss` 出现 NaN | log marker token ids 第一次 batch,人工 sanity check |

---

## 5. 与 rollout_n 解耦的副产品

ref 模式需要 group 内有 success rollout 才能 distill,所以 `rollout_n` 必须够大保证 success 至少出现一次。Marker 模式没有这个 constraint:每条 rollout 都能独立产生 ΔW signal。因此**marker 模式天然支持 rollout_n=1 训练**,大幅降低 PPO step 的 sample cost。

本次 sweep 保持 `rollout_n=8` 是为了和 baseline 严格可比;一旦 marker 在 acc 上 validate,后续可以单独跑一组 `marker rollout_n=1 vs marker rollout_n=8`,验证 sample efficiency。

---

## 6. 未做的事 / Open questions

- **Marker 文本选择敏感性**:本文用 `"This answer is verified correct."`。是否 `"The following solution is correct:"` / `"Final answer (verified):"` 等不同 phrasing 会显著影响 ΔW 信号?需要一个 mini ablation。
- **Marker 是否应 trainable(soft prompt)**:固定 string 是 phase-1 选择,理论上 soft-prompt 化能让 ΔW 在 student 当前能力下"自适应"。但会引入额外 hyperparameter,phase-2 再考虑。
- **与 PrefixGrouper 的交互**:marker prefix 完全 batch-shared,理论上可以走 prefix-cache 节省 teacher forward FLOPs,后续工程优化点。
- **ΔH damping 与 KL direction(`alpha`)的耦合**:reverse KL 比 forward KL 对 entropy gap 更敏感,damping 可能对 `alpha=1.0` (reverse) 更有用,对 `alpha=0.0` (forward) 几乎无效。本次 sweep 固定 `alpha=0.5` (JSD),后续可单独跑 `alpha × damping` grid。

---

## 7. Summary

Marker SDPO 是 OPSD 框架下对 vanilla SDPO 的最小理论修正:把 "ref-conditioned, answer-dependent" 的 teacher 替换为 "marker-conditioned, answer-independent" 的 teacher,从而(1)让 ΔW 恢复 pure value 解释,(2)消除 sibling-success 依赖与 reference 风格漂移噪声,(3)将额外算力成本降为零。3 个 variant 的 sweep 旨在 isolate 三件事:**信号源切换可行性**(marker vs ref)、**marker 引入的 entropy collapse 是否需要 damping**、**marker 与 ref 是否正交**。
