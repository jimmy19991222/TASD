# SDPO Loss 的 Bayes 解读

> **TL;DR** — 用一条 Bayes 恒等式把 SDPO 现有的各类 loss 拆开,统一看清楚它们各自在优化什么。
> - 关键 setup:**OPSD**(on-policy self-distillation,本 codebase `self_distillation_config` 的设定)框架下**只有一个网络 `π_θ`**,所谓 teacher / student 只是同一组参数在两种 conditioning 下的 forward(忽略 teacher 的 EMA 延迟)。这条约束让 `log[p_T(v|r,...)/p_s(v|...)] = ΔW(v)` 成为一个**精确的 Bayes 因子**,而不是两个独立模型的 density log-ratio。
> - ⚠️ **范围限定**:本文所有结论只适用于 **OPSD**。传统 **OPD**(on-policy distillation,teacher 是独立训练的另一个网络)下 `p_T(v|s_<t) ≠ p_s(v|s_<t)`,Bayes 闭式需要额外 calibration 假设。
> - **Token-level PG**: `Q_t = ΔW_t + log p_s(s_t|s_<t)`。`baseline='student'` 直接消掉 `log p_s` 项 → pure value PG。其他 baseline 仍带 likelihood pollution。
> - **Vocab-level rKL**: `KL(p_s‖p_T) = -E_{v∼p_s}[ΔW(v)]` — **闭式就是负的期望 ΔW**,纯 value 目标,mode-seeking。
> - **Vocab-level fKL**: `KL(p_T‖p_s) = E_{v∼p_T}[ΔW(v)]`,梯度是 vocab-soft CE,**让 student 匹配"student prior × p(r|v) 的 Bayes 后验"**,mode-covering。
> - **JSD / α-mix**: rKL 和 fKL 的对称组合,在两种 driver 之间插值。

- **作者**: 月丘
- **日期**: 2026-05-22
- **scope**: 本文只解读现有 loss 的 Bayes 语义,**不**提出新方法。新 baseline 提议见 [verdict_credit_assignment.md](verdict_credit_assignment.md)。
- **代码锚点**: token-level PG 在 [`compute_teacher_qv_advantage`](../verl/trainer/ppo/core_algos.py#L480);vocab-level KL 在 [`core_algos.py:1677-1705`](../verl/trainer/ppo/core_algos.py#L1677-L1705)。

---

## 0. Setup:一个网络,两种 conditioning

### 0.0 OPD vs OPSD:本文只覆盖后者

先把术语理清楚:

| 范式 | teacher 是什么 | teacher vs student 关系 |
|------|---------------|------------------------|
| **OPD**(on-policy distillation) | 一个**独立训练**的、更强的 LM | 两套**独立**参数 `θ_T`, `θ_s` |
| **OPSD**(on-policy **self**-distillation,本 codebase) | 同一份 `π_θ`,但 context 里加入 privileged `r` | **同一组参数**,仅 conditioning 不同(EMA 延迟忽略) |

本 codebase(`self_distillation_config`)走的是 OPSD。本文**所有**推导前提的 `p_T(v|s_<t) = p_s(v|s_<t)` 只在 OPSD 下成立。OPD 下 `p_T` 是另一个网络的密度,需要先假设它是 well-calibrated 才能借鉴本文结论。下文不再重复区分,默认讨论 OPSD。

### 0.1 没有 teacher 网络和 student 网络

OPSD/SDPO 在这个 codebase 里**不存在两个独立的模型**,只存在一个 policy `π_θ`,被用在两种 context 下:

| 记号 | 实际含义 |
|------|---------|
| `p_s(a_t \| s_<t)` | `π_θ(a_t \| s_<t, question, system_prompt)` — 不带 privileged info |
| `p_T(a_t \| r, s_<t)` | `π_θ(a_t \| s_<t, question, system_prompt, r)` — context 里多了 reference `r` |

后文继续用 `p_T` / `p_s` 这两个 KD 风格的符号只为了**简化叙述**,真实情况是 **single policy with vs without `r` in context**。两次 forward 调用同一份 weight `θ`。

### 0.2 关键恒等式:`p_T` 把 `r` marginalize 掉 = `p_s`

把 `r` 从 teacher 的 condition 里去掉,得到的就是 student:

$$
p_T(v \mid s_{<t}) \;=\; \pi_\theta(v \mid s_{<t}, \text{question}, \text{system\_prompt}) \;=\; p_s(v \mid s_{<t})
$$

这不是任何近似——是定义层面的恒等式。

### 0.3 ΔW 因子:`log p_T / log p_s` 的 Bayes 解读

对 vocab 里任意 token `v`,由 Bayes' rule:

$$
p_T(v \mid r, s_{<t}) \;=\; \frac{p(v \mid s_{<t}) \cdot p(r \mid v, s_{<t})}{p(r \mid s_{<t})}
$$

取 log,并把 `p(v|s_<t) = p_s(v|s_<t)` 代进去:

$$
\log p_T(v \mid r, s_{<t}) - \log p_s(v \mid s_{<t}) \;=\; \underbrace{\log p(r \mid v, s_{<t}) - \log p(r \mid s_{<t})}_{=:\;\Delta W(v)}
$$

**`ΔW(v)` 的语义**:在前缀 `s_<t` 处,**把 token `v` 加进去之后,模型对"`r` 是 valid continuation"这个事件的 log-belief 上升了多少**。

- `ΔW(v) > 0`:token `v` 是支持 `r` 的"好"选择
- `ΔW(v) < 0`:token `v` 让 `r` 看起来更不像 valid continuation
- 对所有 v 求期望:`E_{v∼p_s}[ΔW(v)] ≤ 0`(由 Jensen,见 §2.1 末尾)

**所以 `log[p_T(v|r)/p_s(v)]` 在 OPSD 框架下不是"两个网络的 density log-ratio",而是一个 well-defined Bayes 因子** —— 干预变量 = `r` 是否在 context 里。这一点对下面所有 loss 的 Bayes 解读都是 prerequisite。(传统 OPD 下需要额外假设 teacher 是被良好 calibrate 的 posterior,本文不展开。)

---

## 1. Token-level PG Loss(`compute_teacher_qv_advantage`)

### 1.1 原始信号 `Q_t`

SDPO 在采样 token `s_t` 处用的 raw signal:

$$
Q_t \;:=\; \log p_T(s_t \mid r, s_{<t})
$$

把 §0.3 的恒等式代入(`v = s_t`):

$$
\boxed{\;Q_t \;=\; \Delta W_t + \log p_s(s_t \mid s_{<t}), \qquad \Delta W_t := \Delta W(s_t)\;}
$$

**`Q_t` = "纯 value 增量 ΔW_t" + "student 自己在这个 token 上的 log-likelihood"**。这是后面 baseline 分析的核心起点。

### 1.2 各 baseline 在 Bayes 视角下做了什么

PG advantage 的标准形式 `A_t = Q_t − V_t`。把 `Q_t` 替换成 §1.1 的分解:

```
A_t = ΔW_t + log p_s(s_t | s_<t) - V_t
```

| baseline | `V_t` | `A_t` 化简后 | 是否 pure value |
|----------|------|--------------|----------------|
| **`student`** | `log p_s(s_t \| s_<t)` | **`ΔW_t`** | ✅ **纯 value PG,无 likelihood pollution** |
| `ce` | `-log p_s(s_t \| s_<t)` | `ΔW_t + 2·log p_s(s_t \| s_<t)` | ❌ 反而把 student 的 log-likelihood 放大 2 倍 |
| `group_mean` | scalar(batch / token 维度均值) | `ΔW_t + log p_s(s_t \| s_<t) − const` | ❌ 中心化但 token-level 结构未变 |
| `group_hier` | scalar(hierarchical 版) | 同上 | ❌ 同 |

**关键 takeaway**:`baseline='student'` 是 Bayes-aligned 的唯一选择 —— 它正好 cancel 掉 §1.1 分解里的 `log p_s` 项,得到纯 ΔW_t。其他 baseline 都把 `log p_s` 项以不同形式留在 advantage 里,导致 PG 梯度方向同时被 value 信号和 student 自身的 likelihood 信号驱动。

### 1.3 Token-level PG 的 loss 形式

最终 loss(忽略 PPO clip、IS 等):

$$
\mathcal{L}_{\mathrm{PG}}^{\mathrm{token}} \;=\; -\mathbb{E}_t\!\left[A_t \cdot \log p_s(s_t \mid s_{<t})\right]
$$

按 baseline 不同,语义为:

- **`student`** 下:`L = -E[ΔW_t · log p_s(s_t)]` — **value-weighted MLE on sampled action**
- **`ce`** 下:`L = -E[(ΔW_t + 2 log p_s) · log p_s]` — value-weighted MLE + 2 倍 self-reinforcement(`log² p_s` 项是 mode-collapse driver)
- **`group_mean / group_hier`** 下:value 信号 + 残留的 single-power likelihood,中间档

---

## 2. Vocab-level KL Losses(`teacher_qv_full_logit_loss`)

token-level PG 只在采样到的 `s_t` 一格上更新。vocab-level loss 把目标扩展到整张 vocab(B × T × V 张量),用 teacher 的全分布做 soft target。代码里通过 `self_distillation_config.alpha` 控制具体形式([`core_algos.py:1677-1699`](../verl/trainer/ppo/core_algos.py#L1677-L1699)):

| `alpha` | KL 方向 | 名称 |
|---------|---------|------|
| `0.0` | `KL(p_T ‖ p_s)` | **forward KL**(mode-covering) |
| `1.0` | `KL(p_s ‖ p_T)` | **reverse KL**(mode-seeking) |
| `(0,1)` | 广义 JS divergence | α-mixture |

下面把 §0.3 的 Bayes 恒等式代入每个 KL,看它们的闭式语义。

### 2.1 Reverse KL: `KL(p_s ‖ p_T)`(`alpha = 1.0`)

定义:

$$
\mathrm{KL}(p_s \| p_T) \;=\; \sum_v p_s(v \mid s_{<t}) \log \frac{p_s(v \mid s_{<t})}{p_T(v \mid r, s_{<t})}
$$

用 `log p_s - log p_T = -ΔW(v)`(§0.3 反向):

$$
\boxed{\;\mathrm{KL}(p_s \| p_T) \;=\; -\,\mathbb{E}_{v \sim p_s(\cdot \mid s_{<t})}[\Delta W(v)]\;}
$$

**这个等式是本文最干净的结果**。

**语义**:
- **最小化 rKL** ⇔ **最大化 `E_{v∼p_s}[ΔW(v)]`** ⇔ student 把 mass 移到 `ΔW(v)` 高的 token 上
- 这是**纯 value 目标 at vocab level** — 整个梯度方向完全由 ΔW 决定,没有任何 likelihood 项混进来
- 数学上 ≥ 0:由 Jensen,`E_{v∼p_s}[ΔW(v)] = E[log p(r|v)] - log p(r) ≤ log E[p(r|v)] - log p(r) = 0`

**Mode-seeking 几何**:rKL 在 `p_T(v|r)=0` 的地方对 student `p_s(v)=0` 没有惩罚(因为 `0 · log(0/0)=0`),但 `p_T=ε` 时学生放 mass 上去会爆炸。结果 student 只敢蹲在 teacher 大概率的位置 → **student 容易塌成 teacher 后验的单一 mode**。

**等价于 token-level PG 吗**:
- rKL 闭式 = `-Σ_v p_s(v) · ΔW(v)`
- 用 student 采样 `s_t ∼ p_s` 做 Monte Carlo 估计 → `-ΔW(s_t)`
- 配上 score-function trick 求梯度 → `-ΔW(s_t) · ∇_θ log p_s(s_t)`(忽略 baseline)
- **这就是 §1.2 里 `baseline='student'` 的 token-level PG**

所以:**`baseline='student'` 的 token-level PG 是 vocab-level reverse KL 的 stochastic 一阶估计**,两者优化同一个目标,只是后者用解析的 vocab 求和、前者用 sampled token + REINFORCE。这条对应解释了为什么 codebase 里 non-full-logit 模式只支持 reverse KL([`core_algos.py:1703`](../verl/trainer/ppo/core_algos.py#L1703) 的 assertion):token-level 近似自然落到 rKL,fKL/JSD 必须有 full vocab 才能算。

### 2.2 Forward KL: `KL(p_T ‖ p_s)`(`alpha = 0.0`,即 codebase 默认)

定义:

$$
\mathrm{KL}(p_T \| p_s) \;=\; \sum_v p_T(v \mid r, s_{<t}) \log \frac{p_T(v \mid r, s_{<t})}{p_s(v \mid s_{<t})}
$$

代入 `log p_T - log p_s = ΔW(v)`:

$$
\boxed{\;\mathrm{KL}(p_T \| p_s) \;=\; \mathbb{E}_{v \sim p_T(\cdot \mid r, s_{<t})}[\Delta W(v)]\;}
$$

**与 rKL 的对称美**:
- rKL = `−E_{v∼p_s}[ΔW(v)]`
- fKL = `+E_{v∼p_T}[ΔW(v)]`

两者都是 ΔW 的期望,差别仅在采样分布(student prior vs teacher posterior)。fKL 在采样分布 `p_T(v|r) ∝ p_s(v) · p(r|v)` 下取期望——这是 student prior 被 `p(r|v)` 做了 Bayes 更新后的后验。

**梯度形式(teacher detached,匹配 codebase 实现)**:

$$
\nabla_\theta \mathrm{KL}(p_T \| p_s) \;=\; -\sum_v p_T(v \mid r, s_{<t}) \cdot \nabla_\theta \log p_s(v \mid s_{<t})
$$

这是 **vocab-soft cross-entropy** — 把 teacher 后验当 soft target,做加权 MLE。

**等价 reformulation(让 Bayes 解读更显式)**:由 `p_T(v|r) ∝ p_s(v) · p(r|v)`,可以 normalize 出权重 `w(v) := p(r|v) / Σ_v' p_s(v') p(r|v') = p(r|v) / p(r|s_<t)`,则:

$$
\nabla_\theta \mathrm{KL}(p_T \| p_s) \;=\; -\,\mathbb{E}_{v \sim p_s} \left[ w(v) \cdot \nabla_\theta \log p_s(v) \right]
$$

**fKL 在做 importance-weighted MLE under student prior,权重 = "把 v 选作 token 后 r 出现的相对似然"**。

**Mode-covering 几何**:fKL 在 `p_T(v|r) > 0, p_s(v) → 0` 处惩罚趋于无穷,所以 student 必须**覆盖** teacher 后验的每一个有质量的位置。结果 student 学到一个 over-smoothed 分布,覆盖所有合理 mode 但每个都不尖锐。

### 2.3 JSD / α-mixture(`0 < alpha < 1`)

代码里的实现是 generalized Jensen-Shannon divergence,定义 mixture `m_α(v) := α · p_T(v|r) + (1-α) · p_s(v)`,然后:

$$
\mathrm{JSD}_\alpha(p_T, p_s) \;=\; (1-\alpha) \cdot \mathrm{KL}(p_s \| m_\alpha) + \alpha \cdot \mathrm{KL}(p_T \| m_\alpha)
$$

(注:`alpha=0` 退化为 fKL,`alpha=1` 退化为 rKL,这与 `core_algos.py:1693-1699` 的 `torch.lerp` 实现一致。)

**为什么不像 §2.1/§2.2 那样有干净的 ΔW 闭式**:
分母变成 `m_α` 之后,`log p_s − log m_α` 不再纯粹 = `±ΔW`,因为 `m_α` 是两个分布的加性混合,不是 Bayes 更新。所以**JSD 没有一个纯 Bayes 因子的解读**——它在 rKL 和 fKL 中间做平滑,但失去了"被某种 posterior 加权的 ΔW"这种 clean 语义。

**几何**:
- **bounded**:JSD ∈ [0, log 2],不会像 rKL/fKL 在 support mismatch 时爆炸
- **symmetric**(when α=0.5):没有 mode-seeking / mode-covering 偏好
- **gradient direction**:在 rKL 和 fKL 的梯度方向之间插值

**实操定位**:JSD 是一种 **regularization**——主要好处是数值稳定性和对 support mismatch 的鲁棒性,不是因为它有更精确的 Bayes 语义。如果优化目标想"value-aligned",rKL(α=1)更纯;想"distribution-matching",fKL(α=0)更纯;JSD 是工程妥协。

---

## 3. 横向对比

| Loss | 闭式 | 采样分布 | 几何 | 信号纯度 | 在 codebase 里 |
|------|------|---------|------|---------|---------------|
| Token-PG, `baseline='student'` | `-ΔW_t · log p_s(s_t)` (sampled MC) | `s_t ∼ p_s` | mode-seeking(implicit) | **pure value** ΔW | `qv_student-fl`、token-level 模式 |
| Token-PG, `baseline='ce'` | `-(ΔW_t + 2 log p_s) · log p_s` | `s_t ∼ p_s` | self-reinforce → 塌缩 | value + **2× likelihood** | `qv_ce-fl`(Phase 4 最优) |
| Token-PG, `baseline='group_mean/hier'` | `-(ΔW_t + log p_s − c) · log p_s` | `s_t ∼ p_s` | mode-seeking | value + **1× likelihood** | `qv_group_*` |
| **Vocab rKL**(α=1) | `-E_{v∼p_s}[ΔW(v)]` | `v ∼ p_s` | mode-seeking | **pure value** | full_logit, α=1.0 |
| **Vocab fKL**(α=0,**默认**) | `E_{v∼p_T}[ΔW(v)]` (= IW MLE under p_s) | `v ∼ p_T(·\|r)` | **mode-covering** | **pure value, posterior-weighted** | full_logit, α=0.0 |
| **JSD / α-mix** | 无干净闭式 | mixture | 插值,bounded | mixed | full_logit, α∈(0,1) |

### 3.1 三个观察

**(O1) 大部分 SDPO loss 在 OPSD 框架下都是 pure value 目标,只是采样/加权分布不同**:

`baseline='student'` PG、vocab rKL、vocab fKL 都没有 likelihood pollution——它们的差别全在"用什么分布对 ΔW 加权":
- **rKL** → 在 student prior `p_s(v)` 下加权 ΔW(v)
- **fKL** → 在 teacher posterior `p_T(v|r) ∝ p_s(v) p(r|v)` 下加权 ΔW(v)  
- **token-PG with student baseline** → 在 student prior 下采样一格做 MC 估计 rKL

真正"被污染"的只有 `baseline='ce'`(显式放大 `log p_s` 2 倍)和 `group_mean/hier`(残留 `log p_s` 1 倍)。**Phase 4 现有最优是 `qv_ce-fl`,意味着我们当前生产配置里同时存在两路 likelihood 污染源**(token-PG 的 ce baseline + 没用 vocab rKL/fKL 的纯 value 选项)。

**(O2) fKL 与 rKL 的本质对偶**:

| 维度 | rKL `KL(p_s‖p_T)` | fKL `KL(p_T‖p_s)` |
|------|-------------------|-------------------|
| 采样分布 | `p_s`(student prior) | `p_T(·\|r)`(teacher posterior) |
| ΔW 期望符号 | `-` | `+` |
| 学生学到的几何 | mode-seeking(蹲 teacher mode 上) | mode-covering(覆盖 teacher posterior 全部 support) |
| 对 student support 缺失的惩罚 | 弱(在 `p_s=0` 的位置自动消) | 强(`p_T>0, p_s→0` 时无穷大) |
| **entropy 影响** | **倾向于降低 student entropy** | **倾向于保持/提高 student entropy** |

**(O3) Phase 4 的 α-mix 在做什么**:

当前 codebase 把 PG loss(token-level)和 fKL(vocab-level, α=0)混合(`PG_alpha · L_PG + (1-PG_alpha) · L_fKL`)。从 Bayes 视角看:
- PG 这一路(`qv_ce-fl`)= pure ΔW + 2× student likelihood self-reinforcement(mode-seeking driver)
- fKL 这一路 = teacher posterior 下的 value-weighted MLE(mode-covering driver)

两者方向不完全冲突(都最大化 ΔW),但在 entropy 行为上对冲:PG 推塌缩,fKL 推保持 entropy。这或许解释了为什么 Phase 4 加入 fKL 后 reward 比纯 PG 高(fKL 防止 entropy 过快塌缩,保留了 exploration capacity),即使 fKL 名义上是"distillation",在 Bayes 解读下它也在优化 value。

---

## 4. Implementation notes

### 4.1 Teacher detach 的影响

`F.kl_div(input, target, log_target=True)` 计算 `target · (log target − input)`。代码里 teacher 走 `no_grad` forward(标准 KD 习惯),所以 `target=teacher_log_probs` 是 detached 的。这让 fKL 的梯度形式 §2.2 成立:

$$
\nabla_\theta \mathrm{KL}(p_T \| p_s) = -\sum_v p_T(v|r) \cdot \nabla_\theta \log p_s(v)
$$

**如果不 detach** teacher(理论上 OPSD 可以这么做,因为同一组参数 θ),fKL 还会通过 `p_T(v|r)` 的梯度增加一项,对应"让 teacher posterior 自身更尖锐"的压力。**目前 codebase 选择 detach**,所以分析按 detach 走。(传统 OPD 由于 teacher 是独立网络,detach 是 default,根本没这个选项。)

### 4.2 Non-full-logit 模式只支持 rKL

[`core_algos.py:1703`](../verl/trainer/ppo/core_algos.py#L1703) 的 assertion:

```python
assert self_distillation_config.alpha == 1.0, "Only reverse KL is supported for non-full-logit distillation"
log_ratio = student_log_probs - teacher_log_probs
per_token_loss = log_ratio.detach() * student_log_probs
```

由 §2.1 末尾的解释:**这正是 vocab rKL 的 single-sample MC 估计**,信号 = `-ΔW(s_t).detach() · log p_s(s_t)`。所以 token-level KL 模式天然只能做 rKL,fKL/JSD 必须 full vocab。

### 4.3 ΔW 的 sign convention

为避免符号混淆,这里 collected 一遍:

- `ΔW(v) := log p(r|v, s_<t) − log p(r|s_<t)` (§0.3 定义)
- `Q_t = ΔW_t + log p_s(s_t|s_<t)` (§1.1)
- `KL(p_s‖p_T) = -E_{v∼p_s}[ΔW(v)]` (§2.1)
- `KL(p_T‖p_s) = +E_{v∼p_T}[ΔW(v)]` (§2.2)
- 最小化 KL ⇔ 最大化 expected ΔW(at the respective sampling distribution)
- `E_{v∼p_s}[ΔW(v)] ≤ 0`,`E_{v∼p_T}[ΔW(v)] ≥ 0`(均由 Jensen + Bayes posterior 性质)

---

## 5. 与其他文档的关系

- 对 token-level PG 的细致 baseline 推导(`qv_student-fl` / `qv_ce-fl` / `group_*`)见 [`verdict_credit_assignment.md §3.4`](verdict_credit_assignment.md)。本文 §1 是它的 condensed 版本。
- 对 verdict-based credit assignment 的新方案(`verdict_mc` 等)见 [`verdict_credit_assignment.md §4-§7`](verdict_credit_assignment.md)。本文**不**讨论新方法,只解读现有 loss。
- 经验现象(entropy 塌缩、reward 涨幅)的归因分析见 [`teacher_qv_analysis.md`](teacher_qv_analysis.md)。本文的 §3.1 (O1)/(O2) 给这些经验现象提供了 Bayes 视角的理论解释。

---

## 6. 总结一句话

**SDPO 的所有现有 loss(token-PG + vocab KL 系列)在 OPSD shared-param 框架下都可以归约为"用某种分布对 ΔW(v) 加权"——`baseline='student'` PG、vocab rKL、vocab fKL 都是 pure value 目标,差别仅在采样/加权分布;`baseline='ce'/'group_*'` PG 是 pure value + likelihood pollution 的叠加。**(同样的 loss 形式搬到 OPD 上不再自动等价,因为 `p_T(v|s_<t) ≠ p_s(v|s_<t)`,需要额外的 calibration 假设。)

---

*本文写于 2026-05-22,分支 `geodesic-sdpo-v2 @ d4f002a`。Bayes 解读基于 [verdict_credit_assignment.md](verdict_credit_assignment.md) §3.1 / §3.4 的推导,在此处做 vocab-level 的拓展。*
