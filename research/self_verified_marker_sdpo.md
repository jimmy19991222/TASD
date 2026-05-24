# Self-Verified Marker SDPO

> **TL;DR (v2, 2026-05-24)** — 在 OPSD(teacher ≡ student 共享参数)下,用 fixed verdict marker 替代 privileged ref 作为 teacher 的额外 conditioning,使 ΔW 折叠成 marker 信念的边际贡献(§2)。第一轮 sweep 暴露 `gt_marker` 模式 +7.4pt 峰值但 response length 崩到 12 tok 的 **GT-shortcut** 问题(§4 实测)。
>
> **当前主推方向 — §3.7 Counterfactual Discriminative Markers (CDM)**:在 assistant 头部同时构造 `m^{+gt}="verified correct, reference is {gt}"` 和 `m^{-gt}="verified incorrect, reference is {gt}"` 两个 teacher 上下文,定义 token-level 贝叶斯证据
> $$\Delta W^{\text{discr}}_t = \log p_T(y_t \mid x, m^{+gt}, y_{<t}) - \log p_T(y_t \mid x, m^{-gt}, y_{<t})$$
> trajectory-sum 严格 telescope 到 verdict 后验 log-Bayes-factor。**两个 context 都注入 GT,所以 GT-copy shortcut 在减法里结构性抵消**;只有真正承载"正确性判别"的 token 信号非零。在 reverse-KL SDPO 下等价于"marker posterior 的变分推断 ELBO",纯 loss 替换,不和 R 复合 → 不踩 §3.5 sign-trap。
>
> 旧版 TL;DR(单 marker / gt_marker 三 variant sweep)保留作历史 anchor;实验设计与判读见 §8。

## Reading Guide (建议阅读顺序)

| 优先级 | 章节 | 内容 |
|---|---|---|
| **★ 主线** | §0 → §1 → §2 → **§3.7 (CDM)** → §8 (实验) | 当前推荐方案的最短路径 |
| 背景 | §3 (vs Zhao 2026) → §4 (sweep 实测) | 文献定位 + 第一轮经验事实 |
| 早期尝试 | §3.6 (ΔW^shortcut decomposition) | CDM 的前身;两次 marker forward + topk mask,被 CDM 用更干净的减法对称构造取代 |
| Negative anchor | §3.5 (R-conditioning sign-trap) | A1/A2 双层失败推演,后续再触碰 R-conditioning 必读 |
| 杂项 | §9 / §10 / §11 | 副产品 / open questions / summary |

- **作者**: 月丘 + Claude (Opus 4.7)
- **日期**: 2026-05-23 初稿,2026-05-24 加入 §3.7 CDM 并重组导航
- **状态**: §1-§3.6 已实现并完成第一轮 sweep;§3.7 (CDM) 设计完成,实现中
- **相关代码**:
  - 配置: [`verl/workers/config/actor.py`](../verl/workers/config/actor.py) (`SelfDistillationConfig.teacher_context_mode / self_verified_marker / overconfidence_damping / cdm_*`)
  - 数据装配: [`verl/trainer/ppo/ray_trainer.py:694-857`](../verl/trainer/ppo/ray_trainer.py#L694-L857) (`_maybe_build_self_distillation_batch`,CDM 走 `_build_cdm_self_distillation_batch`)
  - Loss: [`verl/trainer/ppo/core_algos.py`](../verl/trainer/ppo/core_algos.py) (`compute_self_distillation_loss`,CDM 分支在 `_compute_verdict_distillation_loss`)
  - Sweep: [`nebula_scripts/submit_sdpo_marker_sweep.sh`](../nebula_scripts/submit_sdpo_marker_sweep.sh)(v1),[`nebula_scripts/submit_sdpo_cdm_sweep.sh`](../nebula_scripts/submit_sdpo_cdm_sweep.sh)(CDM v2)
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

## 3.5 把 seq-level reward R 接入 marker SDPO —— 两层 negative result + 两条 sign-correct 候选路径

> **结论 (TL;DR)**:试图把 sample-level R 加权与 token-level marker credit `ΔW_t = c_t^+ - c_{t-1}^+` 复合,踩到两层嵌套的坑。
> - **第一层(sign 层)**:naive 乘积 `A(R) · Δc^+`(单向 marker × R 权重)在 R=0 + marker 校准好的子集上 trajectory-sum 反号 —— **model 越能自验证,gradient 方向越错**。详见 §3.5.2。
> - **第二层(objective 层)**:用双向 marker `R·Δc^+ - (1-R)·Δc^-` 修掉 sign 后,优化目标在 R=0 trajectory 上被悄悄偷换成"最小化模型对 wrongness 的主观信念"。在 OPSD teacher=student 下,这是显式训 policy 学会写让自己 verifier 放松警惕的"体面瞎话"(verifier deception / reward hacking)。详见 §3.5.4-A2。
>
> 两层后,sign-correct 的复合方案剩下两条 —— **A1 switching**(R=1-only marker shaper,R=0 退回 GRPO 均权;零额外算力、无 deception 风险但失去 R=0 token credit)与 **A2 bidirectional**(`R·Δc^+ - (1-R)·Δc^-`;能拿到 R=0 token credit 但需要 +1× teacher forward + deception monitoring + calibration anchor)。两者都**不在当前 sweep 范围**。
>
> 本节既记录失败推演当 anchor,也把两条退路写清当后续 implement 时的 spec。

### 3.5.1 动机:为什么想把 R 直接接进 ΔW

§2 给出的 `ΔW_t = c_t^+ - c_{t-1}^+` 是一个**只看 marker 信念变化**的 token credit,完全不依赖 R。这带来一个让人不满的后果(§0.1 痛点 1 的另一面):

- 它在 R=1 trajectory 上 well-defined(model 看到对答案后变更信,各 token 有差异化贡献)。
- 它对 R=0 trajectory **没有合适的解释**:marker mode 在 R=0 trajectory 上仍然会算 `ΔW_t`,但 student 这条 trajectory 本身就不该被 reinforce,marker credit 在这里到底想推哪边并不显然。

直观上希望有一个"统一公式",形如

$$
A_{i,t} = A(R_i) \cdot \Delta W_{i,t}, \qquad A(R_i) \in \{\text{R 加权 / } (2R-1) \text{ / } A_{\text{GRPO}}\}
$$

让 sample-level R 决定方向、token-level marker 决定形状。如果这能成立,**marker SDPO 就解锁了 R=0 trajectory 的学习,sibling-success 依赖被解除**。下面看 sign 怎么走。

### 3.5.2 Sign 分析:trajectory-sum 在 R=0 上反号

每条 trajectory 贡献的总 gradient 正比于

$$
A(R_i) \cdot \sum_t \Delta W_{i,t} = A(R_i) \cdot \big(c_T^+ - c_0^+\big) =: A(R_i) \cdot S_i
$$

其中 $S_i$ 是这条 trajectory 上 marker 信念的净增量(P1 telescoping)。讨论四种情形:

| 情形 | $A(R_i)$ | $S_i$ 的"自然" sign | 乘积 | PG 物理含义 |
|---|---|---|---|---|
| R=1, marker 校准好 | $+$ | $+$ (model 看到对答案变更信) | $+$ | reinforce ✓ |
| R=1, marker 失校 (overconfident) | $+$ | $\approx 0$ | $\approx 0$ | 信号被 wash 掉,但 sign 没错 |
| R=0, marker 校准好 | $-$ | $-$ (model 看到错答案不信) | $+$ | **reinforce 错轨迹 ✗** |
| R=0, marker 失校 (overconfident) | $-$ | $+$ (model 误信错答案) | $-$ | down-weight 错轨迹 ✓ |

关键观察:**marker estimator 越准,R=0 trajectory 的 trajectory-sum sign 越错**。我们想把 marker 训得越来越校准(这本来就是 §2 想要的 Bayes 干净性),结果是让这条"统一公式"在 R=0 上越用越坏。**训练越成功,这部分 loss 越破坏方向**。

### 3.5.3 为什么 sign 会反:credit ≠ advantage

根因是把两个**不同物理意义**的量当成同一种东西乘起来了:

- $\Delta W_t = c_t^+ - c_{t-1}^+$ 描述的是**"token t 让 model 多/少相信 marker"**。它的"自然方向"是跟 model 自评的对齐度走。
- PG advantage 描述的是**"token t 让真实回报增加多少"**,方向应该跟 verifier 的真实 R 走。

R=1 trajectory 上两者方向一致(model 自评准 ↔ trajectory 是对的),所以 sign 巧合对得上。R=0 trajectory 上两者方向**相反**:一个"诚实"的 R=0 trajectory(model 自评出"这是错的",$\Delta W_t < 0$)恰好是 verifier 也判错的 trajectory,我们想 down-weight;但 $A(R) \cdot \Delta W = (-)\cdot(-) > 0$ 反而 reinforce。

**marker credit 是"appearance-of-correctness"信号,不是"is-correct"信号**。只有当 trajectory 的 appearance 跟 reality 在同一侧(R=1)时,它才能当 PG advantage 用。

### 3.5.4 方案对比与权衡

§3.5.2 的 sign 分析关掉了"单纯乘起来"(`A(R) · ΔW_t` 或 §3.5.2 P3 那种 `R·Δc^+ + (1-R)·Δc^-`)这条路。如果还想让 R=0 信号进入 loss,代数上 sign-correct 的路径有两条,各有 tradeoff。

#### 方案 A1:Switching Variant(单向 marker + 均权退回)

$$
A_{i,t} = \begin{cases}
A_{\text{GRPO}}(R_i) \cdot (c_t^+ - c_{t-1}^+) & R_i = 1 \quad (\text{marker 给 token shape}) \\[4pt]
A_{\text{GRPO}}(R_i) & R_i = 0 \quad (\text{退回 vanilla GRPO,token 均权})
\end{cases}
$$

**物理含义**:marker 是 R=1-only 的 shaper。R=1 trajectory 上 marker 把 sample-level 的正权重在 token 间重分配,挑出"贡献了正确感"的 token 加倍 reinforce;R=0 trajectory 上 marker 没有可信方向(见 §3.5.2 calibrated 行的反号),退回 plain GRPO 的均权 down-weight。

**优点**:零额外算力(沿用现有单向 marker forward);**完全免疫** §3.5.2 R=0 calibrated 行的 sign bug 和 §3.5.4-A2 的 deception 风险,因为 R=0 上没有 token-level 区分度可被钻空子。

**缺点**:R=0 trajectory 失去 token-level credit,跟 plain GRPO 一样把整段错答案均匀压低,无法定位"哪个 token 是真凶"。

#### 方案 A2:Bidirectional Sign-Flip Variant(双向 marker 对冲)

$$
w_t = R \cdot (c_t^+ - c_{t-1}^+) - (1 - R) \cdot (c_t^- - c_{t-1}^-)
$$

其中 $c_t^- := \log p_T(m^- \mid x, y_{\le t})$,$m^- = $ `"This answer is verified incorrect."`,需要额外一次 teacher forward。

**物理含义**:R=1 时 reward 那些拉高"正确信念"的 token;R=0 时 punish 那些拉高"错误信念"的 token(R=0 分支显式取负,见 §3.5.2 的 trajectory-sum 验证)。

**Sign 检查通过**:trajectory-sum 在两种 R 上都对了 —— R=1 时 $\sum w_t = c_T^+ - c_0^+ > 0$ 触发 reinforce;R=0 时 $\sum w_t = -(c_T^- - c_0^-) < 0$ 触发 down-weight。比方案 A1 多保留了 R=0 trajectory 上的 token-level credit 锐度,能"精准抓出 1+1=3 这种罪魁 token"。

**潜在 Failure Mode(目标函数偷换 / verifier deception)**:这是 A2 比 A1 多付的代价,**必须在 sweep 前预先 monitor**。考察 R=0 trajectory 上的 token-level sign:

| token 类型 | $\Delta c_t^-$ | $w_t = -\Delta c_t^-$ | PG 效果 |
|---|---|---|---|
| 写 `1+1=3`(明显错)| $\gg 0$ | $\ll 0$ | 重罚 ✓ |
| 写 "I don't know"(中性)| $\approx 0$ | $\approx 0$ | 无 |
| 写 "by Cauchy-Schwarz, …"(看起来体面的瞎话)| $< 0$ | $> 0$ | **被 reinforce ⚠** |

最后一行才是关键。在 OPSD 共享参数下 teacher = student,policy 被训练"在失败轨迹上写让自己的 $c^-$ 评估头放松警惕的内容"。**目标函数从"最大化真实 R"被悄悄偷换为"最小化轨迹结尾对 wrongness 的主观信念"**;这两个目标只在 marker 估计与 R 完全 grounded 时一致。规则 verifier(sciknoweval)下风险可控,LLM-as-judge / model-based reward 下会被 Goodhart 蚕食。

**其他工程坑**:
- **+1× teacher forward**:bidirectional 必须分别 condition 在 $m^+$ 和 $m^-$ 上各跑一次,KV-cache 只能 share prompt 那段,marker + response 段省不掉。算力开销 ~+30~50%。
- **$c^-$ 的语言 prior 不对称**:`"verified incorrect"` 在预训练数据中出现频率比 `"verified correct"` 低数个数量级,$c^-$ 的 calibration 大概率比 $c^+$ 差。**sweep 前必须做 sanity probe**:取一批 known-wrong response,验证 $c_t^-$ 是否确实随 t 单增。
- **建议配套**:加 marker BCE calibration loss $L_{\text{calib}} = \text{BCE}(p_T(m \mid y), R)$ 当 anchor,把 marker estimator 锚回 R,缓解 deception drift。

#### 决策表

| 维度 | A1 Switching | A2 Bidirectional |
|---|---|---|
| Sign 正确性 | ✓ | ✓ |
| R=0 利用度 | 仅 sample-level | sample + token-level |
| 额外算力 | 0 | +1× teacher forward |
| Deception 风险 | 无 | **有**(需 monitor + calib anchor) |
| 实现复杂度 | 低(switch on R) | 中(双 forward + 两套 marker 流) |
| 推荐情形 | 默认起点 / R 来自规则 verifier 也想稳的场景 | 已有 calib 监控基础 + 愿意 sweep 验证的场景 |

**两者都不是当前 sweep 范围内的 commitment**。建议路径:先回收 §4 现有 marker sweep,marker mode 本身 work 之后,再用 A1 做最小增量(只引 GRPO scaling),最后才考虑 A2。

### 3.5.5 与原想统一的三个朴素方案的实际关系

| 朴素方案 | 真实地位 |
|---|---|
| `L = R · L_marker`(整段 R 加权) | 等价于 §3.5.4 switching 在"R=0 完全丢弃"那一极端。它丢掉了 R=0 信号,但 sign 没错。**比"乘 ΔW"安全**。 |
| VC-OPSD sign-flip `(2R-1) · ΔW` | 在 R=0 上同样有 §3.5.2 的 sign 风险。早期 VC-OPSD 工作之所以 empirically work,可能是 model 当时还没学好自验证,落在表第 4 行;模型变强后该被重审。|
| 学 $V_\phi(y_{\le t})$ 走 GAE | 仍然不优雅(破坏 OPSD 共享参数 + 引 λ/γ),但 sign 是对的,因为 V 学的是真实 expected R 而不是 marker belief。 |

也就是说**之前推过的 "(2R-1) · ΔW" 不是一个 sound 的复合**,只是一个在 model 弱时凑巧能 work 的 hack。

### 3.5.6 对当前 sweep 与未来工作的影响

1. **当前 §4 marker sweep 不受影响**:`marker / marker_damp / refmarker / gt_marker` 都只在 success group 上算 `ΔW_t`,本来就走 R=1 子集,不踩 §3.5.2 的坑,也不踩 §3.5.4-A2 的 deception 坑。
2. **chat 里讨论过的"GRPO + marker shaper"(naive `A_GRPO · Δc^+`)被作废**。要走 R-conditioning,只走 §3.5.4 的 A1 或 A2,不要 implement 那个 unidirectional 乘积版本。
3. **"用 R 解除 sibling 依赖"这条 motivation 有两条可选解,但都不 free**:
   - A1 switching:零额外算力,sign 安全,**完全没有 R=0 token credit** —— 等于承认 marker 是 R=1-only 信号。
   - A2 bidirectional:能拿到 R=0 token credit,但要 +1× teacher forward + deception monitoring + calib anchor,且必须先过 $c^-$ sanity probe。
4. **`L_calib = BCE(p_T(m | y), R)` 在两条路径上都建议加**。在 A1 里它是 nice-to-have(只为了让 marker 估计跟 R 对齐);在 A2 里它是 **必备的 anchor**(没有它,A2 的 deception 风险无法被结构性压住,只能靠监控发现后回退)。它跟 PG 完全解耦,不引 sign bug。

### 3.5.7 这次记录的 takeaway

两条嵌套的 lesson,后一条比前一条更隐蔽:

**Lesson 1 — Sign check on R-misaligned subset**

任何时候想把一个"语言空间的信念变化"(`ΔW_t`, marker delta, log-odds shift)当 PG advantage 乘进 loss,都要先做 §3.5.2 那张表:**在 R 跟"信念方向"不一致的子集上,trajectory-sum 的 sign 是不是仍然对**。如果不是,这个复合就是 model-stage-dependent 的 hack,不是 sound 框架。`A_GRPO · Δc^+` 这个 naive 乘积栽在这一关上。

**Lesson 2 — Sign correctness ⊉ objective alignment**

但 sign 对了**只是必要条件**,不是充分条件。§3.5.4-A2 的 `R·Δc^+ - (1-R)·Δc^-` sign 完全对,trajectory-sum 在两种 R 上都符合 PG 方向,然而它在 R=0 trajectory 上**悄悄把目标函数偷换**成了"最小化轨迹结尾对 wrongness 的主观信念" —— 在 OPSD teacher=student 共参数下,这就是训 policy 学会写让自己 verifier 放松警惕的体面瞎话。**目标函数与原 PG 目标(max E[R·log π])只在"marker 估计与 R 完全 grounded"这一极强假设下重合**。

形式化判别准则:写出 composed advantage $A^{\text{composed}}_{i,t}$ 之后,检查 $\sum_{i,t} A^{\text{composed}}_{i,t}$ 是否等于(或单调相关于)$\sum_i R_i \cdot |y_i|$ 这种纯 R 函数。如果 advantage 的总和依赖于 belief estimator 的内部状态而**不仅**依赖于 R 和轨迹长度,那这个 composition 就在 R 信号之外引入了额外优化方向 —— 而 belief estimator 在 OPSD 下又是 policy 自己,这条额外优化方向天然存在 reward-hacking 的 fixed point。

**底层 framing**

P1 (telescoping)、P2 (marker 是 ΔV^R 的特例)、P3 (sign-flip 是对称特例) 三个结论本身代数上对,但它们刻画的是 **belief-space** 的代数性质,不是 **reward-space** 的 PG 正确性 —— 这是 §3.5 第一稿滑过去的第一层。Sign 对了但 objective 被偷换 —— 这是 §3.5.4-A2 揭示的第二层。两层教训叠起来说的是同一件事:**在 OPSD 共参数架构下,任何用模型自己的内部信念做 credit 的方案,都要把模型当成既能被骗也会去骗的对手来 stress-test,而不是当成被动的概率估计器**。

---

## 3.6 Token-level Information Decomposition:从 ΔW 中减除 GT-shortcut

### 3.6.1 动机:gt_marker 的 +7.4pt 是真的,但 length collapse 也是真的

`SDPO_Marker` sweep 实测(20260524):

| Variant | best val acc | final | final resp len | final entropy |
|---|---|---|---|---|
| **gt_marker** | **0.6525** @155 | 0.575 | **12 tok** | 0.18 |
| SDPO baseline (ref) | 0.579 @205 | 0.570 | 77 | 0.17 |
| marker_damp | 0.350 @160 | 0.336 | **9 tok** | 0.07 |
| marker (pure) | 0.320 @5 | 0.320 | 153 | 0.69 (训练 step 7 即崩) |

两个关键事实:

1. **gt_marker 峰值领先 SDPO baseline +7.4pt** —— GT 出现在 teacher prefix 里(`"correct answer is {ground_truth}"`)给出的信号是真有用。
2. **末态 response length 崩到 12 tok**,acc 从 0.6525 掉到 0.575 —— 学生学到的不是"如何推理出正确答案",而是"直接照抄 prefix 里的 GT"。

第二个事实说明:gt_marker 的 ΔW_t 信号里**至少有两个不同的成分**叠在一起:
- 一部分来自"GT 在 prefix 里直接提到了这个 token" → **shortcut 信号**(我们不想学)
- 一部分来自"在 verdict 'verified correct' 条件下,这个 token 是合理推理的一步" → **真信号**(我们想学)

§3.5 的两层教训是关于 R-conditioning 的 sign;这里的问题正交 —— 在 R=1 子集内部,信号本身就被 shortcut 污染。

### 3.6.2 形式化:用双 teacher 对比定位 shortcut

记 `m_gt = "verified correct, correct answer is {gt}"`,`m_mk = "verified correct"`。在 OPSD 共享参数下,同一个 student 上的 per-token teacher log-ratio 是:

$$\Delta W_t^{\text{gt}} := \log p_\theta(y_t \mid x, m_{\text{gt}}, y_{<t}) - \log p_\theta(y_t \mid x, y_{<t})$$

$$\Delta W_t^{\text{mk}} := \log p_\theta(y_t \mid x, m_{\text{mk}}, y_{<t}) - \log p_\theta(y_t \mid x, y_{<t})$$

定义 **shortcut 分量**:

$$\boxed{\;\Delta W_t^{\text{shortcut}} \;:=\; \Delta W_t^{\text{gt}} - \Delta W_t^{\text{mk}} \;=\; \log \frac{p_\theta(y_t \mid x, m_{\text{gt}}, y_{<t})}{p_\theta(y_t \mid x, m_{\text{mk}}, y_{<t})}\;}$$

**直觉:**`ΔW^shortcut_t` 是"GT 出现在 prefix 里相对于只有 verdict 时,teacher 在这个 token 上多出来的概率提升"。这个量**完全独立于 student**(student 项相消),衡量的纯粹是"GT 注入给 teacher 带来了多少边际信息"。

**物理含义:**
- `ΔW^shortcut_t` 大 → 这个 token 主要因为 prefix 写了 GT 才被 teacher 高概率推荐 → **shortcut token**
- `ΔW^shortcut_t` ≈ 0 → teacher 看不看 GT 这个 token 的概率都差不多 → **task-intrinsic token**(真正的推理步骤)

**减除后的 task signal:**

$$\Delta W_t^{\text{task}} := \Delta W_t^{\text{gt}} - \Delta W_t^{\text{shortcut}} = \Delta W_t^{\text{mk}}$$

也就是说,纯 marker 的 ΔW 就是 task signal。但**单独训 marker 模式信号弱**(实测 step 7 即崩),所以我们的策略不是"用 marker 替代 gt_marker",而是 **"用 gt_marker 的强信号驱动训练,用 marker 标识哪些 token 属于 shortcut 并从梯度里抑制"**。

### 3.6.3 Mask 设计

给定 `ΔW^shortcut_t`,在现有 SDPO loss 上加 token-level 抑制:

$$\mathcal{L}_{\text{decomp}} = \sum_t w_t \cdot \ell_t^{\text{SDPO-gt}}$$

其中 `ℓ^SDPO-gt_t` 是 gt_marker 教师下的 per-token JSD/KL 项,`w_t` 是 mask。三种候选 mask:

| Mask 类型 | 公式 | 备注 |
|---|---|---|
| **M1 Hard top-k 剔除** | `w_t = 𝟙[rank(ΔW^shortcut_t) > k]` | 干净,k 取 top 10%~20% |
| **M2 Soft 指数衰减** | `w_t = exp(-max(0, ΔW^shortcut_t) / τ)` | 平滑,τ=1.0 起步 |
| **M3 Hard 阈值** | `w_t = 𝟙[ΔW^shortcut_t < τ]` | 阈值需要绝对量,标定难 |

推荐 **M1 top-k 剔除** 作为首跑:每个 trajectory 内做相对排序,阈值自适应,不依赖绝对尺度。

### 3.6.4 实现路径

#### 改动定位

1. **`verl/trainer/ppo/ray_trainer.py:_maybe_build_self_distillation_batch`**(L740~L900):
   - 当 `teacher_context_mode == "gt_marker_decomp"` 时,构造**两个** teacher 输入:gt_marker prefix 和 marker prefix
   - 当前 `gt_marker` 分支已经做了 GT marker 拼接(L802-849),复用代码,平行加一份 marker prefix

2. **teacher 前向**(actor worker `update_actor` / `compute_log_prob`):
   - gt_marker teacher 走原路径(需要 grad,作为蒸馏目标)
   - marker teacher 走 `torch.no_grad()`(只用来算 mask,不流梯度) → **额外计算 ≈ +30% teacher cost,无额外显存峰值**
   - 输出两套 per-token log-prob:`logp_gt_T`, `logp_mk_T`

3. **mask 计算**(SDPO loss 入口,`core_algos.py` 或 self-distill loss 函数):
   - `ΔW^shortcut_t = logp_gt_T[t] - logp_mk_T[t]`
   - 在 response_mask 内做 per-trajectory 排序,生成 `w_t`
   - 应用到现有 JSD/KL token loss 上

#### 新增 config(`actor.yaml` self_distillation 段)

```yaml
teacher_context_mode: gt_marker_decomp  # 新增模式
decomp_baseline_marker: "This answer is verified correct."  # 与 self_verified_marker 默认一致
decomp_mask_mode: topk          # topk / soft / threshold
decomp_topk_frac: 0.15          # M1: 剔除 top 15% shortcut token
decomp_soft_tau: 1.0            # M2: 软衰减温度
```

#### 新增 metrics(都在 train 端)

- `actor/self_distill/dw_shortcut_mean` / `dw_shortcut_max` / `dw_shortcut_p95`
- `actor/self_distill/mask_frac`(被 mask 掉的 token 比例,应在 10%~20%)
- `actor/self_distill/mask_position_hist`(被 mask 的 token 在 response 中的位置分布;假设应集中在尾部"答案"位置)
- `actor/self_distill/dw_gt_mean` / `dw_mk_mean`(分别看两个 teacher 的原始 ΔW 分布)

### 3.6.5 分阶段实验计划

**Phase 0 — 假设验证(只 log,不 mask):**
- 跑一次 `gt_marker_decomp` with `decomp_mask_mode=none`,只把 `ΔW^shortcut` 分布和位置 histogram log 出来
- **必须验证两件事**才能继续:
  1. `ΔW^shortcut` 分布有清晰的长尾(p95 / mean > 3),证明 shortcut 信号确实集中在少数 token
  2. 位置分布**集中在 response 尾部 / 答案 token 附近**,而不是均匀分布
- 如果分布是均匀的或集中在错误位置 → 假设伪,decomposition 没用,直接停止

**Phase 1 — Top-k mask(预计 1~2 周):**
- `decomp_mask_mode=topk, topk_frac=0.15`
- **判读标准**:
  - val acc peak ≥ 0.62(不大幅低于 gt_marker 的 0.6525)
  - **response length 在训练末期保持 ≥ 100 tok**(关键 KPI,证明 shortcut 被抑制)
  - val acc 末态 drop ≤ 0.03(峰值之后稳定,不像 gt_marker 掉 0.078)

**Phase 2 — Mask 形式扫描:**
- 对比 M1 (topk=0.10 / 0.15 / 0.20) vs M2 (soft τ=0.5 / 1.0 / 2.0)
- 选稳定性 / acc trade-off 最好的一组

**Phase 3 — 写作:**
- 标题草拟:*Token-level Information Decomposition for Privileged-Context Self-Distillation*
- Story:gt_marker 暴露 shortcut → 形式化 ΔW^shortcut → mask 实验验证 shortcut 可分离 → 在保持 +6pt 增益的前提下消除 length collapse

### 3.6.6 预期 vs 风险

| Phase | 期望结果 | 失败信号 | 失败时的解释 |
|---|---|---|---|
| 0 | ΔW^shortcut 长尾 + 尾部集中 | 分布均匀 | shortcut 在每个 token 上都有(可能是 teacher 整体被 GT 推高一个常数),decomposition 失败 |
| 1 | acc ≥ 0.62 且 length ≥ 100 | length 仍崩 | shortcut 不止在 mask 的 top-k 上;可能需要更激进的 top-k 或换 mask 形式 |
| 1 | acc ≥ 0.62 且 length ≥ 100 | acc 掉到 < 0.58 | 我们错杀了 task 信号 token;需要降低 topk_frac |
| 2 | M1/M2 一致显示某 region 稳定 | 全部组合不稳定 | decomposition 本身分辨力不够,需要更强的 baseline teacher(例如多个 paraphrased marker 的平均) |

### 3.6.7 与 §0 / §3.5 的关系

- **vs §0 (为什么要 marker)**:§0 列出了 ref 模式的三个痛点,marker 模式是解药。但 marker 模式自己有 shortcut 问题(本节)。§3.6 = "如何让 marker 模式既能用上 GT 强信号又不被 shortcut 污染"。
- **vs §3.5 (R-conditioning 的 negative result)**:§3.5 关注**轨迹级符号问题**(R=0 token 怎么给 credit);§3.6 关注**token 级信号分解**(R=1 内部哪些 token 是 shortcut)。两者**正交**,可以独立做、也可以未来合并(R=1 子集上跑 §3.6,R=0 子集上跑 §3.5-A1)。
- **vs L_calib**:§3.5 提到的 L_calib(BCE 校准 anchor)和本节互补 —— L_calib 让 marker likelihood 本身可信,decomposition 让 ΔW 信号干净。两者都开是最强组合。

### 3.6.8 一句话总结

> **把 ΔW 拆成 `ΔW^gt = ΔW^task + ΔW^shortcut`,用第二个 teacher pass(verdict-only marker)定位并 mask 掉 shortcut 分量;训练目标仍然是 gt_marker 的强信号,但学生学到的是与 GT 注入无关的那部分推理梯度。**

> **更新 (2026-05-24)**:§3.6 的"识别 + mask"路线在 §3.7 被一个更对称的构造取代 —— 不再先识别 shortcut 再剔除,而是用**两个都含 GT 但 verdict 相反**的 marker 做减法,让 GT-copy 信号在结构上直接抵消、不需要 mask 阈值调参。§3.7 是当前主推方向,§3.6 保留作为"identify-then-mask 思路"的 anchor。

---

## 3.7 Counterfactual Discriminative Markers (CDM) — 当前主推方案

> **TL;DR**:与其先识别 shortcut 再 mask 掉,不如直接构造两个**verdict 相反但都注入 GT** 的 teacher 上下文 `m^{+gt}` / `m^{-gt}`,以它们对同一 student token 的 log-prob 差作为新的 token credit:
>
> $$\Delta W^{\text{discr}}_t \;=\; \log p_T(y_t \mid x, m^{+gt}, y_{<t}) - \log p_T(y_t \mid x, m^{-gt}, y_{<t})$$
>
> **两个 context 都拼了 GT**,所以"照抄 GT"这种 shortcut token 在两个分母里 likelihood 都被同等推高,**减法里直接抵消**,留下的非零信号恰好是"承载 verdict 区分度"的 token —— 即真正的推理步骤。Trajectory-sum 严格 telescope 到 sentence-level verdict log-Bayes-factor,符合 §0.3 想要的 answer-independent value 解释。整个机制是对 §3.6 "ΔW^shortcut mask"思路的对称替代,不需要 mask 阈值,也避开 §3.5 的 R-composition sign-trap(纯 loss 替换,不和 R 相乘)。

### 3.7.1 动机:为什么 §3.6 mask 路线需要被替代

§3.6 提出 `ΔW^shortcut = ΔW^gt - ΔW^mk` 用 verdict-only marker 当 reference,再 mask 掉 top-k shortcut token。这条路线有三个工程痛点:

1. **Mask 阈值是调参 surface**。top-k 选 0.10 还是 0.20 直接决定保留多少 task 信号、丢掉多少 shortcut,而 ΔW^shortcut 的分布尺度本身随训练步漂移,**没有先验地"对"的阈值**。Phase 2 整个 scan 就是在解决这个调参痛点。
2. **"识别再剔除"不是干净的概率论操作**。被 mask 的 token 完全脱离 loss → gradient 上是 hard zero,**梯度方差被人为放大**;mask 反而成了一个 student-看到 / teacher-看到 的不对称裁剪,引入第二种 bias。
3. **Marker-only teacher 自己信号弱**(§4 实测 step 7 即崩),拿它当 reference 去校 GT teacher 是"用弱模型校强模型",reference 本身有 noise。

CDM 换一条思路:**不再"先识别 shortcut 再 mask"**,而是**让 shortcut 在 loss 的构造层就被对称地抵消掉**。这样:
- 没有 mask 阈值 — 减法是处处生效的恒等操作。
- 没有梯度硬截断 — 所有 token 都有平滑梯度,只是 magnitude 是 verdict 差的结果。
- 没有"弱模型校强模型" — 两个 teacher 都拼了 GT,信号强度对等。

### 3.7.2 形式化:对称 verdict + 共享 GT

记
$$
m^{+gt} := \texttt{"This answer is verified correct, reference answer is }\{gt\}\texttt{."}
$$
$$
m^{-gt} := \texttt{"This answer is verified incorrect, reference answer is }\{gt\}\texttt{."}
$$

两个上下文**结构上几乎完全相同**:
- 都在 assistant header 前注入(沿用 §3.6 / `gt_marker` 已有的 prefix 注入位置)
- 都把同一份 `{gt}` 文本拼到 marker 末尾
- 唯一差异只在 `correct` ↔ `incorrect` 这一个 verdict 形容词

CDM 的 token credit 定义为:
$$
\boxed{\;\Delta W^{\text{discr}}_t \;:=\; \log p_T(y_t \mid x, m^{+gt}, y_{<t}) - \log p_T(y_t \mid x, m^{-gt}, y_{<t})\;}
$$

注意 student 项 `log p_s(y_t)` **在减法里自动消失**,所以 ΔW^discr 是完全的 teacher-internal 量,不依赖 student 当前 logits。

### 3.7.3 Bayes telescoping:trajectory sum 等于 verdict log-Bayes-factor

按 §2 Bayes 翻转(条件 prefix-order invariance 近似下):

$$
\log \frac{p_T(y_t \mid x, m^{+gt}, y_{<t})}{p_T(y_t \mid x, m^{-gt}, y_{<t})}
= \log \frac{p(m^{+gt} \mid x, y_{\le t}) / p(m^{+gt} \mid x, y_{<t})}{p(m^{-gt} \mid x, y_{\le t}) / p(m^{-gt} \mid x, y_{<t})}
$$

定义 `b_t := log[p(m^{+gt} | x, y_<=t) / p(m^{-gt} | x, y_<=t)]`(看到前 t 个 token 后,模型对"verdict=correct 给定 GT 已知"相比"verdict=incorrect 给定 GT 已知"的 log-Bayes-factor),则

$$
\Delta W^{\text{discr}}_t = b_t - b_{t-1}, \qquad \sum_{t=1}^{|y|} \Delta W^{\text{discr}}_t = b_{|y|} - b_0
$$

**Trajectory-sum 是 sentence-level verdict log-Bayes-factor 在看到完整 response 后的净更新。** 这跟 §2 的 marker 单边 `ΔW_t = c_t^+ - c_{t-1}^+` 是完全平行的结构,只是 belief 量从"verdict=correct 的 marginal log-prob"升级成"verdict=correct vs incorrect 的 log-odds"。后者是更干净的判别量:不受单边 marker 的 prior bias 影响(`p(m^+)` 和 `p(m^-)` 共有的 language frequency bias 在减法里抵消)。

### 3.7.4 为什么 GT-copy shortcut 被结构性抵消

把 token 按"被哪些条件推高"做分类(在 OPSD 共享参数下 teacher = student-with-extra-context):

| Token 类型 | $p_T(y_t \mid x, m^{+gt}, y_{<t})$ | $p_T(y_t \mid x, m^{-gt}, y_{<t})$ | $\Delta W^{\text{discr}}_t$ | 说明 |
|---|---|---|---|---|
| **GT-copy**(token 直接来自 prefix 里的 GT 文本)| ↑↑(被 GT 推高)| ↑↑(同样被 GT 推高,verdict 不影响"复制 GT"这个 LM 行为)| **≈ 0** | shortcut 抵消 ✓ |
| **Format / boilerplate**("Therefore", "The answer is", 标点)| 与 verdict 无关 | 与 verdict 无关 | **≈ 0** | language prior 抵消 ✓ |
| **支持"正确"的推理步骤**(正确的中间结论、相关公式)| ↑(verdict=correct 让它更合理)| ↓(verdict=incorrect 让 model 倾向不写这步)| **> 0** | reinforce 真信号 ✓ |
| **支持"错误"的轨迹形状**(已经偏离 GT 的步骤)| ↓ | ↑(verdict=incorrect 让这种 token 反而合理化)| **< 0** | down-weight 真错处 ✓ |

**关键观察**:GT-copy shortcut 的 sign 是 0(不是被 mask 掉),因为 "把 GT 抄进 response" 这个行为对两个 verdict 都同等似然 —— LM 不会因为 verdict 是"correct"还是"incorrect"就改变"复制 prefix 里出现过的字符串"这个 mechanical 行为。**这是结构性的,不依赖任何阈值或调参**。

对比 §3.6 mask 路线:那里 shortcut token 的 ΔW^shortcut > 0,需要主动检测并裁掉;CDM 里 shortcut token 的 ΔW^discr 直接 ~ 0,不需要任何额外操作。

### 3.7.5 退化模式与鲁棒性

CDM 在两种极端下都不会反向坏事:

**(a) Marker-dominant 极端**:如果 model 完全 ignore GT,只看 verdict 形容词决定 token 分布(等价于 §2 的纯 marker 模式),那么 ΔW^discr 退化成 `(c_t^+ - c_{t-1}^+) + (c_t^- - c_{t-1}^-)` 的对称结构。Trajectory-sum 仍然 ≥ 0 当 R=1,与 §2 marker 结论一致,**信号方向正确**。

**(b) GT-dominant 极端**:如果 model 完全 ignore verdict,只用 GT 决定每个 token(即 LM 行为完全是"在已知 GT 的前提下重写 response"),那么两个 teacher 的 logits 几乎相同,ΔW^discr 处处 ≈ 0。Loss → 0,梯度 → 0。**没有错误信号,但也没有反向梯度** —— 训练静默退出,不破坏已有参数。

**对比 §3.6 mask 路线在退化下的行为**:GT-dominant 时 §3.6 的 ΔW^shortcut 处处大,mask 掉绝大部分 token,留下的 task 信号也是 ΔW^mk(marker-only,弱信号),实际效果约等于"用一个弱 marker mode 训练"——**会持续往一个无效方向推**。CDM 的退化是"安全的零",§3.6 的退化是"持续的小推"。

**正常工作区间**(model 同时考虑 marker 和 GT,且 verdict 影响中间推理步骤的概率):ΔW^discr 在推理步骤 token 上非零、在 shortcut token 上 ≈ 0,正是我们想要的信号几何。第一轮 sweep 末态 `gt_marker` length 崩到 12 tok 说明 model 此时还处在 GT-dominant 模式;CDM 在这个 regime 下的预期行为是"信号弱但不破坏",随训练推进 model 学会区分 verdict 后信号会自然出现。

### 3.7.6 与 Reverse-KL SDPO 的整合:变分推断 ELBO 视角

把 CDM 接入现有 SDPO loss 有两个等价 framing。

**Framing A — Token-level CFG distillation**:对每个 token 把 teacher posterior 定义为
$$
q^*(y_t) \propto p_T(y_t \mid x, m^{+gt}, y_{<t}) \cdot \exp(\Delta W^{\text{discr}}_t)
$$
然后 reverse-KL 投影到 student:`L = KL(π_θ || q*)`。展开后等价于
$$
L^{\text{CDM}}_t \;=\; -\sum_{v} \pi_\theta(v \mid x, y_{<t}) \cdot \big[\log p_T(v \mid x, m^{+gt}, y_{<t}) - \log p_T(v \mid x, m^{-gt}, y_{<t})\big]
$$
**(student-expectation × teacher log-ratio)**。这跟 [`reward_bayes_distillation_v2.md`](reward_bayes_distillation_v2.md) opd_bayes 的 log-likelihood-ratio 结构一致,只是 ratio 的两端从"verdict 不一致"换成"verdict 相反但 GT 一致"。**直接复用 opd_bayes 的 topk pipeline**,只需替换 marker injection。

**Framing B — Variational marker posterior ELBO**:设隐变量 `m ∈ {m^+, m^-}` 表示"verdict 信念",先验 `p(m) = 1/2`。后验
$$
p(m=+ \mid x, y) = \sigma\!\big(\sum_t \Delta W^{\text{discr}}_t\big)
$$
取 student 的 variational family `q_θ(y | x)` 拟合这个后验的 evidence。Reverse-KL 下 ELBO 的 token-level 梯度恰好是 Framing A 的形式。这给出 CDM 的概率论合法性:**不是 ad-hoc 的双 teacher 减法,而是 verdict 隐变量后验的变分推断**。

两个 framing 都得出同一个 loss,实现上只需 Framing A 的算式。

### 3.7.7 实现路径

#### 数据装配 — 新增 `_build_cdm_self_distillation_batch`

参考已有 `_build_verdict_self_distillation_batch`([`ray_trainer.py:911-1000`](../verl/trainer/ppo/ray_trainer.py#L911-L1000)):
- 接受同一 batch 的 prompt + response + ground_truth
- **位置**:沿用 §3.6 / `gt_marker` 的**assistant-turn 前置**([`ray_trainer.py:700-865`](../verl/trainer/ppo/ray_trainer.py#L700-L865)),不是 verdict 路径的 user-turn 拼接
- **两次注入**:per-sample 各生成一个 `m^{+gt}` 和 `m^{-gt}` 字符串(`.format(ground_truth=gt)`),左 pad 到 batch 内最大 marker 长度
- **输出 6 个张量**:`teacher_input_ids_pos / mask_pos / position_pos` + `teacher_input_ids_neg / mask_neg / position_neg`,字段名与 verdict 路径一致以便复用下游 forward
- 不输出 `verdict_R`(CDM 不用 R)

#### Loss 分支 — 新增 `loss_method='cdm'`

在 [`core_algos.py:_compute_verdict_distillation_loss`](../verl/trainer/ppo/core_algos.py) 加一个 `elif loss_method == "cdm":` 分支,核心两行:
```python
log_lr = (teacher_topk_log_probs_pos - teacher_topk_log_probs_neg).detach()  # [B, T, K]
per_token_loss = -(student_topk_log_probs.exp() * log_lr).sum(-1)             # [B, T]
```
其中 `student_topk_log_probs` 来自第一次 student forward 的 topk indices,`teacher_topk_log_probs_pos/neg` 来自两次 teacher forward 在**同样 indices** 上的 gather(已有 `topk_indices=student_topk_indices` 复用机制,见 [`dp_actor.py:917-934`](../verl/workers/actor/dp_actor.py#L917-L934))。

可选 full-vocab 形式(`distillation_topk=null`):同样的减法在完整 vocab 上做,显存代价 +1× vocab tensor。

#### 配置 — actor.py 新增

```python
# verl/workers/config/actor.py SelfDistillationConfig
teacher_context_mode: str = "ref"  # +"cdm"
cdm_positive_template: str = "This answer is verified correct, reference answer is {ground_truth}."
cdm_negative_template: str = "This answer is verified incorrect, reference answer is {ground_truth}."

# 验证
valid_loss_methods = {"sdpo", "vc_opsd_sign", "opd_bayes", "vec", "cdm"}
valid_teacher_ctx_modes = {"ref", "marker", "ref_and_marker", "gt_marker", "cdm"}
# cdm 要求 full_logit_distillation=True(topk 路径走 student-aligned indices)
```

#### Dispatch 改动

- [`ray_trainer.py:673-692`](../verl/trainer/ppo/ray_trainer.py#L673-L692) `_maybe_build_self_distillation_batch`:把 `"cdm"` 加进 verdict 派发集
- [`dp_actor.py:717-721`](../verl/workers/actor/dp_actor.py#L717-L721) `verdict_loss_active`:把 `"cdm"` 加进集合

整个改动 **复用 90% 已有 verdict 双 teacher pipeline**,只需新的 build 函数 + 新的 loss 分支 + config 注册。

#### 诊断 metrics

- `actor/self_distill/cdm_dw_discr_mean` / `p95` / `min` — ΔW^discr 分布
- `actor/self_distill/cdm_dw_discr_abs_mean` — magnitude(零信号 vs 真零的区分)
- `actor/self_distill/cdm_pos_neg_logp_gap_mean` — `mean(teacher_logp_pos) - mean(teacher_logp_neg)`,sanity check(应当 > 0 当 R=1 trajectory 占多数)
- `actor/self_distill/cdm_per_token_loss_mean` / `actor/self_distill/cdm_nonzero_token_frac`(|ΔW^discr| > 1e-3 的比例)

### 3.7.8 实验矩阵

第一轮 sweep,3 个 cell 互为消融:

| Tag | `teacher_context_mode` | `loss_method` | 目的 |
|---|---|---|---|
| `gt_marker` (baseline 复用) | `gt_marker` | `sdpo` | 已跑,best 0.6525 但 length 崩 — anchor 第一行 |
| `cdm` | `cdm` | `cdm` | 主推方案 — Bayes 双向减法 |
| `cdm_no_gt` | `marker` + `cdm` 改用 `m^+ = "correct"` / `m^- = "incorrect"` | `cdm` | **消融 GT 注入是否必要** — 没有 GT 时退化为 §3.5.4 A2 形式 |

通过率判读(降序优先):
1. **`cdm` val acc peak ≥ 0.62 且 final length ≥ 100 tok** — 主目标
2. **`cdm` final acc drop ≤ 0.05**(vs gt_marker 的 -0.078)— stability 改进
3. **`cdm_nonzero_token_frac` 在训练初期 > 30% 且随训练增长** — 信号被正确激活的证据
4. **`cdm` vs `cdm_no_gt`**:如果 cdm > cdm_no_gt,确认 GT 注入即便在抵消框架下仍有 net 增益(GT 提升了 verdict 判别精度);如果约等,说明 verdict 信号本身足够、GT 仅冗余

### 3.7.9 与 §3.5 / §3.6 的关系小结

| 方面 | §3.5 (R-conditioning) | §3.6 (mask shortcut) | **§3.7 (CDM, 本节)** |
|---|---|---|---|
| 处理对象 | R=0 trajectory 的 sample-level 加权 | R=1 内部的 token-level shortcut | R=1 内部的 token-level shortcut |
| 机制 | A(R) × ΔBelief 复合 | 阈值 mask | 对称减法构造 |
| Sign 风险 | **有**(§3.5.2 trap) | 无(纯加权 ≥ 0) | 无(纯 loss 替换) |
| Deception 风险 | **有**(§3.5.4 A2) | 无 | 无(verdict 对称,无单边 hack 通道) |
| GT 利用方式 | N/A | mask 出 GT-copy | 双 verdict + 共享 GT,减法抵消 |
| 额外算力 | 0 / +1× teacher | +1× teacher (no-grad) | **+1× teacher (需 grad,因 loss 来自 ratio 两端)** |
| 是否需要调参 | 否(A1)/ 是(A2 monitoring) | 是(top-k 阈值) | **否**(对称减法是恒等操作) |

CDM 在三个维度上都是 strict improvement over §3.6,且不踩 §3.5 的两层 trap。代价是 teacher forward 量 +1× —— 跟 §3.5.4-A2 同量级,但目标干净。

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

**第一轮 sweep 经验事实(20260524)**:`gt_marker` 峰值 0.6525 (+7.4pt vs SDPO baseline),但 final response length 崩到 12 tok —— 强信号 vs GT-copy shortcut 的张力暴露。`marker` 纯模式 step 7 即崩,`marker_damp` 信号被 damping 吃掉。两条后续路径已经形成:

- **§3.6 — Identify-then-mask**:用第二个 verdict-only marker teacher 定位 ΔW^shortcut,top-k mask 掉。需要阈值调参,留作 ablation anchor。
- **§3.7 — Counterfactual Discriminative Markers (CDM, 当前主推)**:构造 `m^{+gt}` / `m^{-gt}` 两个 verdict 相反但都注入 GT 的 teacher,用 `ΔW^discr_t = log p_T(y_t | x, m^{+gt}) - log p_T(y_t | x, m^{-gt})` 当 token credit。Trajectory-sum 是 sentence-level verdict log-Bayes-factor;GT-copy shortcut 在两个 logit 里同等出现,**减法层直接抵消、不需要 mask 阈值**。可证明在 reverse-KL 下等价于"verdict 隐变量后验的变分推断 ELBO"。整个机制是 §3.5 R-conditioning 失败教训之后的 sign-clean 路径:不和 R 复合,纯 loss 替换。

§3.7 是接下来要 implement 与 sweep 的方向;§3.5 / §3.6 留作 negative result 与早期路线 anchor。
