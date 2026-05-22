# 用 Bayes 看 SDPO 类自蒸馏 loss:不同 baseline、不同 KL,其实在优化同一个量

> **TL;DR** — SDPO / OPSD(on-policy self-distillation)类 teacher-distill RL 的核心信号 `Q_t = log p_T(s_t | r, s_<t)`,**在 OPSD 这种 teacher 与 student 共享 policy 参数的设定下**,可以做一个一行的 Bayes 分解,得到 `Q_t = ΔW_t + log p_s(s_t | s_<t)`,其中 `ΔW_t` 是 "把这个 token 加进去后,reference `r` 作为 valid continuation 的 log-belief 上升了多少"。这个分解把整套 loss 拆得很干净:**`baseline='student'` 的 token-level PG ≡ vocab-level reverse KL 的 MC 估计 ≡ pure value 目标**;**`baseline='ce'` 反而把 `log p_s` 放大 2 倍**,引入 self-reinforcement;**forward KL ≡ teacher posterior 加权下的 pure value 目标**;**JSD 没有干净的 Bayes 闭式**,只是 rKL/fKL 的工程妥协。"likelihood ≠ value" 这个对 SDPO 的常见批评只对一半,真正的 likelihood pollution 来自 baseline 选择,不是 SDPO 范式本身。
>
> ⚠️ **范围限定**:本文所有 Bayes 推导前提的 `p_T(v|s_<t) = p_s(v|s_<t)` 等式只在 **OPSD(self-distillation,teacher/student 共享一份 weight,仅 conditioning 不同)** 下成立。**传统 OPD(on-policy distillation,teacher 是独立训练的另一个网络)下这个等式不成立**,本文结论需要额外的 calibration 假设才能借鉴。本 codebase 的 `self_distillation_config` 是 OPSD,所以分析直接适用。

- **作者**:月丘
- **日期**:2026-05-22
- **完整技术细节**:[research/sdpo_loss_bayes_reading.md](sdpo_loss_bayes_reading.md)

---

## 0. 背景:SDPO 类自蒸馏 RL 在做什么

LLM 的 RL 后训练有两条主流路线:

1. **Outcome-based RL**(PPO / GRPO 等):verifier 给整条 rollout 一个标量 reward,PG / RLHF 类目标更新 policy
2. **Teacher distillation RL**(SDPO / OPD / OPSD / KD-RL):一个更"强"的 teacher 给 student rollout 的每个 token 打 log-probability,student 学着"在 teacher 觉得好的 token 上 mass 更高"。其中 **OPD**(on-policy distillation)teacher 是独立网络;**OPSD**(on-policy self-distillation)teacher 与 student 共享同一份 `π_θ`,仅 conditioning 不同——本文聚焦后者。

第二类的核心信号长这样:

```
Q_t = log p_T(s_t | r, s_<t)
```

其中 `r` 是 teacher context 里的 reference(prompt + 可选的 gold answer / hint / 参考解法等 privileged info)。

这个范式上有个长期质疑:**`log p_T(s_t)` 不是 value function,它是 density**。teacher 觉得这个 token "看起来对" 不等于它"对 final outcome 真有贡献"。从 RL 理论上看,SDPO 更像 KD,不像 credit assignment。

但实证上 SDPO 类方法确实能涨 reward——所以它一定在做点对的事。**到底在做什么对?在做什么不对?为什么各种 baseline / KL 方向选择会有那么大差异?**

这就是这篇要回答的问题。

---

## 1. 关键架构约束:OPSD 里没有两个网络(对照 OPD)

讨论 SDPO 的所有数学之前,先把一个常被默认掉的事实说清楚:

> **OPSD 框架里不存在两个独立的 teacher / student 模型,只有一个 policy `π_θ` 在两种 conditioning 下的 forward**。

### 1.1 OPD vs OPSD

| 范式 | teacher 是什么 | teacher vs student 关系 |
|------|---------------|------------------------|
| **OPD**(on-policy distillation) | 一个**独立训练**的、更强的 LM | 两套**独立**参数 `θ_T`, `θ_s` |
| **OPSD**(on-policy **self**-distillation,本 codebase) | 同一份 `π_θ`,但 context 里加入 privileged `r` | **同一组参数**,仅 conditioning 不同(忽略 teacher 的 EMA 更新) |

本文所有推导只在 OPSD 框架下严格成立。传统 OPD 下 `p_T` 是另一个网络的密度,Bayes 解读需要额外 calibration 假设——这里**不**展开。本 codebase 的 `self_distillation_config` 是 OPSD,所以下面 §2 起的推导直接适用。

### 1.2 OPSD 的精确恒等式

具体:

| 记号 | 真实含义 |
|------|---------|
| `p_s(a_t \| s_<t)` | `π_θ(a_t \| s_<t, question, system_prompt)` — 不带 privileged info |
| `p_T(a_t \| r, s_<t)` | `π_θ(a_t \| s_<t, question, system_prompt, r)` — context 里多了 `r` |

后文继续用 `p_T` / `p_s` 这两个 KD 风格的符号只是为了简化叙述。**真实情况是 single policy with vs without `r` in context**,两次 forward 调用同一份 weight。

**这条约束给出一个精确恒等式**:`p_T` 把 `r` 从 context 里 marginalize 掉,**就是** `p_s`:

$$
p_T(v \mid s_{<t}) \;=\; p_s(v \mid s_{<t})
$$

这不是 calibration 假设,是定义层面的等式(同一份 weight、同一个前缀,只是后者多了一个被 marginalize 的 `r`)。下面所有推导都用得到。在传统 OPD 下这个等式不成立,所以 §2 之后的 Bayes 闭式也不成立。

---

## 2. 一行 Bayes 推导:`Q_t = ΔW_t + log p_s`

对 vocab 里任意 token `v`,由 Bayes' rule:

$$
p_T(v \mid r, s_{<t}) \;=\; \frac{p(v \mid s_{<t}) \cdot p(r \mid v, s_{<t})}{p(r \mid s_{<t})}
$$

取 log,用 §1 的恒等式 `p(v|s_<t) = p_s(v|s_<t)` 代进去:

$$
\log p_T(v \mid r, s_{<t}) - \log p_s(v \mid s_{<t}) \;=\; \underbrace{\log p(r \mid v, s_{<t}) - \log p(r \mid s_{<t})}_{=:\;\Delta W(v)}
$$

**`ΔW(v)` 的语义**:在前缀 `s_<t` 处,**把 token `v` 加进去后,模型对"`r` 是 valid continuation"这个事件的 log-belief 上升了多少**。

- `ΔW(v) > 0`:`v` 是支持 `r` 的"好"选择
- `ΔW(v) < 0`:`v` 让 `r` 看起来更不像 valid continuation
- 在 student prior 下取期望:`E_{v∼p_s}[ΔW(v)] ≤ 0`(由 Jensen)
- 在 teacher posterior 下取期望:`E_{v∼p_T}[ΔW(v)] ≥ 0`

**所以 `log[p_T(v|r)/p_s(v)]` 在 OPSD 框架下不是"两个网络的 density log-ratio",而是一个 well-defined Bayes 因子** —— 干预变量是"`r` 是否在 context 里"。这是后面所有 Bayes 解读的核心 lemma,也是 OPSD 比 OPD 数学上"干净"的关键(OPD 里同样写法的 log-ratio 是两个独立网络的密度比,不是 Bayes 因子)。

对采样到的 token `s_t`,即:

$$
\boxed{\;Q_t \;=\; \Delta W_t + \log p_s(s_t \mid s_{<t}), \qquad \Delta W_t := \Delta W(s_t)\;}
$$

**SDPO 的 raw signal `Q_t` = "纯 value 增量 ΔW_t" + "student 自己在这个 token 上的 log-likelihood"**。

---

## 3. Token-level PG:不同 baseline 在 Bayes 视角下做什么

PG advantage 的标准形式 `A_t = Q_t − V_t`。把 §2 的分解代入:

```
A_t = ΔW_t + log p_s(s_t | s_<t) − V_t
```

逐个 baseline:

| baseline | `V_t` | `A_t` 化简后 | 是否 pure value |
|----------|------|--------------|----------------|
| **`student`** | `log p_s(s_t \| s_<t)` | **`ΔW_t`** | ✅ **纯 value PG** |
| `ce` | `-log p_s(s_t \| s_<t)` | `ΔW_t + 2·log p_s(s_t \| s_<t)` | ❌ 反而把 `log p_s` 放大 **2 倍** |
| `group_mean / hier` | scalar 均值 | `ΔW_t + log p_s − const` | ❌ 残留 1× `log p_s` |

### 一个反直觉的结论

**`baseline='student'` 是 Bayes-aligned 的唯一选择**——它正好 cancel 掉 §2 分解里的 `log p_s` 项,得到纯 ΔW_t。其他 baseline 都把 `log p_s` 项以不同形式留在 advantage 里。

**最反直觉的是 `baseline='ce'`(Phase 4 现有最优)**:它的 `V_t = -log p_s`(rationale 是"做 cross-entropy 形式的 baseline"),但代入 §2 分解后,负负相消,**`log p_s` 项实际被放大了 2 倍**。这意味着 `qv_ce-fl` 的 loss 形式约等于:

$$
\mathcal{L}_{\mathrm{qv\_ce}} = -\mathbb{E}\left[\underbrace{\Delta W_t}_{\text{value}} \cdot \log p_s(s_t) + \underbrace{2 \log p_s(s_t)}_{\text{2× self-reinforce}} \cdot \log p_s(s_t)\right]
$$

第二项是 `log²p_s` —— **学生越自信的 token,梯度推它更自信**。这是 mode-seeking 塌缩的直接 driver,解释了为什么 `qv_ce-fl` 的 entropy 比 `qv_student-fl` 塌得快。

---

## 4. Vocab-level KL:三种方向的 Bayes 闭式

[`core_algos.py`](../verl/trainer/ppo/core_algos.py) 里 vocab-level loss 通过 `alpha` 参数选择 KL 方向:

| `alpha` | KL 方向 | 名称 |
|---------|---------|------|
| `0.0` | `KL(p_T ‖ p_s)` | **forward KL**(mode-covering)— codebase 默认 |
| `1.0` | `KL(p_s ‖ p_T)` | **reverse KL**(mode-seeking) |
| `(0,1)` | 广义 JS divergence | α-mixture |

下面把 §2 的 ΔW 恒等式分别代进每个 KL。

### 4.1 Reverse KL: `KL(p_s ‖ p_T)`

$$
\mathrm{KL}(p_s \| p_T) \;=\; \sum_v p_s(v) \log \frac{p_s(v)}{p_T(v|r)} \;=\; -\,\mathbb{E}_{v \sim p_s}[\Delta W(v)]
$$

(用了 `log p_s − log p_T = −ΔW`。)

**这是本文最干净的等式**:**vocab-level reverse KL 的闭式就是负的期望 ΔW**。

- **最小化 rKL** ⇔ **最大化 `E_{v∼p_s}[ΔW(v)]`** ⇔ student 把 mass 移到 `ΔW(v)` 高的 token 上
- 完全是 **value 目标 at vocab level**,没有任何 likelihood 项混进来
- 数学上 ≥ 0 由 Jensen 保证

**与 token-level PG 的关系**:用 student 采样 `s_t ∼ p_s` 做单样本 MC,配 score-function trick:

```
-ΔW(s_t).detach() · ∇_θ log p_s(s_t)
```

——**这正好是 §3 里 `baseline='student'` 的 token-level PG 梯度**。

**所以:`baseline='student'` 的 token-PG 是 vocab rKL 的 single-sample MC 估计**,两者优化同一个目标。这条对应也解释了 [`core_algos.py:1703`](../verl/trainer/ppo/core_algos.py#L1703) 为什么 assert non-full-logit 模式只支持 reverse KL——token-level 近似自然只能落到 rKL,fKL/JSD 必须有 full vocab。

### 4.2 Forward KL: `KL(p_T ‖ p_s)`(codebase 默认)

$$
\mathrm{KL}(p_T \| p_s) \;=\; \mathbb{E}_{v \sim p_T(\cdot|r)}[\Delta W(v)]
$$

**与 rKL 的对称美**:

|  | rKL | fKL |
|---|---|---|
| 闭式 | `−E_{v∼p_s}[ΔW(v)]` | `+E_{v∼p_T}[ΔW(v)]` |
| 采样分布 | student prior | teacher posterior `p_T(·\|r) ∝ p_s(v)·p(r\|v)` |

**两者都是 ΔW 的期望,差别仅在采样分布**——一个在 student prior 上加权,另一个在 student prior 被 `p(r|v)` 做 Bayes 更新后的后验上加权。**没有 likelihood pollution**。

**梯度形式(teacher detached)**:

$$
\nabla_\theta \mathrm{KL}(p_T \| p_s) \;=\; -\sum_v p_T(v|r) \cdot \nabla_\theta \log p_s(v) \;=\; -\,\mathbb{E}_{v \sim p_s}\!\left[\frac{p(r|v)}{p(r|s_{<t})} \cdot \nabla_\theta \log p_s(v)\right]
$$

第二个等号用了 `p_T(v|r) = p_s(v) · p(r|v) / p(r|s_<t)`。**fKL 在做 importance-weighted MLE under student prior,权重 = 把 v 选作 token 后 r 出现的相对似然**。

**Mode-covering 几何**:fKL 在 `p_T(v|r) > 0, p_s(v) → 0` 处惩罚趋于无穷,student 必须**覆盖** teacher 后验的每个有质量位置,学到 over-smoothed 但 full-support 的分布。

### 4.3 JSD / α-mixture

代码实现是 generalized JS divergence,mixture `m_α(v) = α · p_T(v|r) + (1-α) · p_s(v)`:

$$
\mathrm{JSD}_\alpha = (1-\alpha)\,\mathrm{KL}(p_s\|m_\alpha) + \alpha\,\mathrm{KL}(p_T\|m_\alpha)
$$

**为什么 JSD 没有干净的 Bayes 闭式**:分母变成 `m_α` 之后,`log p_s − log m_α` 不再 = `±ΔW`,因为 `m_α` 是加性混合,**不是 Bayes 更新**。所以 JSD 在 rKL 和 fKL 之间做插值,但失去了"被某种 posterior 加权的 ΔW"这种 clean 语义。

**实操定位**:JSD 是 **regularization**——主要好处是数值稳定(bounded in [0, log 2])和对 support mismatch 的鲁棒性,**不是因为它有更精确的 Bayes 语义**。想 value-aligned → rKL;想 distribution-matching → fKL;JSD 是工程妥协。

---

## 5. 三个观察

### O1. 大部分 SDPO loss 其实是 pure value 目标,差别只在加权分布

| Loss | 闭式 | 采样/加权分布 | likelihood pollution |
|------|------|--------------|---------------------|
| Token-PG, `baseline='student'` | `-ΔW_t · log p_s(s_t)` (MC) | `p_s` | 无 |
| Token-PG, `baseline='ce'` | `-(ΔW_t + 2 log p_s) · log p_s` | `p_s` | **2×** `log p_s` |
| Token-PG, `baseline='group_*'` | `-(ΔW_t + log p_s − c) · log p_s` | `p_s` | **1×** `log p_s` |
| Vocab rKL(α=1) | `-E_{v∼p_s}[ΔW(v)]` | `p_s` | 无 |
| Vocab fKL(α=0,**默认**) | `E_{v∼p_T(·\|r)}[ΔW(v)]` | `p_T(·\|r)` | 无 |
| Vocab JSD(α∈(0,1)) | 无闭式 | `m_α` mixture | 无,但 Bayes 语义模糊 |

**6 个 loss 配置里有 4 个是 pure value**(`student` PG、rKL、fKL、JSD),只有 2 个带 likelihood pollution(`ce` PG、`group_*` PG)。"likelihood ≠ value" 这条对 SDPO 的常见批评**只对一半**——它精确命中 `ce`/`group_*` baseline,**不命中 `student` baseline 和所有 vocab-level KL**。

### O2. rKL 与 fKL 是关于 entropy 的对偶

|  | rKL | fKL |
|---|---|---|
| 学生几何 | mode-seeking(蹲 teacher mode) | mode-covering(覆盖 teacher 全 support) |
| `p_T>0, p_s→0` 惩罚 | 弱 | 强(无穷) |
| **entropy 影响** | **降 student entropy** | **维持/提高 student entropy** |

**这给"为什么 Phase 4 加入 fKL 后 reward 更高"提供了 Bayes 解释**:Phase 4 的 α-mix 把 PG loss(token-level)和 fKL(vocab-level α=0)混合:
- PG 那一路(`qv_ce-fl`)= pure ΔW + 2× student likelihood self-reinforce → 强 mode-seeking
- fKL 那一路 = teacher posterior 下的 value-weighted MLE → mode-covering

两者**方向上不冲突**(都最大化 ΔW),但**entropy 行为上对冲**——PG 推塌缩,fKL 推保持 entropy。**最终 reward 提升不一定来自"fKL 提供了 distillation 信号",而可能是"fKL 防止 PG 把 entropy 塌得太快"**,保留了 exploration capacity。

如果这个解释成立,有可证伪的预测:
- 用 `baseline='student'` 替代 `ce`(去掉 PG 的 self-reinforce),应该不需要 fKL 也能保持 entropy
- 用 vocab rKL 替代 token-level PG with `ce`,reward 应该接近现有 `qv_ce-fl + α=0.5` 而 entropy 更稳

### O3. JSD 在 Bayes 视角下不优于 fKL/rKL,只在工程上更稳

JSD 没有 ΔW 闭式,**Bayes 解读上比 fKL/rKL 弱**——它只是两者的加性插值,失去了"被特定 posterior 加权的 ΔW"这种 clean 语义。它的存在价值是 bounded(数值稳定)和对零概率位置鲁棒,而不是优化目标更"正确"。

**所以在 Phase 4 之外做 alpha 扫时,优先扫 α∈{0, 1}(纯 fKL 或 纯 rKL),而不是 JSD 内插值**——除非 observed 数值不稳定。

---

## 6. 取走的三件事

### 6.1 "Likelihood ≠ value" 这个批评只对一半

SDPO 的 raw signal `Q_t` 确实可以分解出 value 部分(`ΔW_t`)和 likelihood 部分(`log p_s(s_t|s_<t)`),但**likelihood 部分能不能消掉,完全取决于 baseline / KL 方向选择**:

- `baseline='student'` → cancel 掉 `log p_s` → pure value PG
- vocab rKL / fKL → 没有 likelihood 项 → pure value
- `baseline='ce'` → 反而把 `log p_s` 放大 2 倍 → strong self-reinforce
- `baseline='group_*'` → 残留 1× `log p_s` → mild self-reinforce

**问题不在 SDPO 范式本身,而在具体的 baseline 实现**。在 OPSD 框架下,纯 value 信号本来就一直在 `baseline='student'`、vocab rKL、vocab fKL 这几个 path 里。

### 6.2 OPSD 的 shared-param 设定让 contrast 自动发生

在传统 KD / OPD 设定下(teacher 是独立训练的固定模型,或独立的另一个网络),`log[p_T/p_s]` 是两个网络的 density log-ratio,没有清晰的语义。但**在 OPSD shared-param 设定下,它就是 Bayes 因子 ΔW**——干预变量是"`r` 是否在 context 里"。

这个 contrast 不需要任何额外 forward / 任何新代码——**它在 `Q_t - log p_s` 这个减法里自然发生**,只要 baseline 选 `student` 就拿到。这是 OPSD 相对 OPD 的一个被低估的优势:Bayes 解读是免费送的,不需要 calibration 假设。

### 6.3 fKL 比"distillation regularizer"做得更多

fKL 在 SDPO 的 α-mix 里通常被解读为"防止 student 漂离 teacher 的 distillation 项"。但 Bayes 分解告诉我们 **fKL 本身就是 pure value 目标**,只是在 teacher posterior 而非 student prior 上加权 ΔW。

这意味着:
- α-mix 里 fKL 项**也在优化 reward**,不只是"regularization"
- 但它的 mode-covering 几何让它**同时维持 student entropy**,这是 token-PG with `ce` 没有的性质
- 所以 fKL 在 α-mix 里同时承担"value 优化"和"entropy 维持"两个 role——这才是 Phase 4 α-mix 比 α=1.0(纯 PG)有效的真正机制

---

## 7. Caveats

1. **Teacher detach**:本文 fKL 梯度推导假设 teacher 走 no_grad forward(`F.kl_div` 的 `target` 参数),这是 codebase 现状。理论上 OPSD 可以不 detach(同一组参数),此时 fKL 还会增加一项推 teacher posterior 更尖锐的压力,但这不是当前实现。
2. **shared-param 恒等式 `p_T(·|s_<t) = p_s(·|s_<t)` 是精确的(仅在 OPSD 下)**(同一份 weight),不需要任何 calibration 假设。所以 §2 的 ΔW 解读不受 teacher 是否"calibrated"影响——这条性质 OPSD 比传统 OPD / KD 更干净。若把本文搬到 OPD(独立 teacher 网络)上,所有 Bayes 闭式都需要先假设 teacher 是被良好 calibrate 的 posterior,数学就脏多了。
3. **EMA teacher 的影响**:严格地说 OPSD 实现里 teacher 通常带一个 EMA 拷贝(用于稳定 KL 信号)。本文忽略了这层 EMA 延迟,假设 teacher weight = student weight。当 EMA 速度足够快(或 EMA decay → 0)时这个近似无害;EMA 较慢时,`p_T(v|s_<t) = p_s(v|s_<t)` 等式只在 EMA 同步那一刻成立,期间有小幅 drift。本文的所有结论在这个近似精度内成立。
4. **ΔW 本身的 calibration 仍然取决于模型对 `r` 的 Bayes coherence**——这取决于训练数据和 system prompt 设计。如果模型对 `r` 不敏感(privileged info 没起作用),所有 ΔW 都接近 0,SDPO 信号退化。

---

*完整 vocab-level KL 推导、代码锚点、token-level PG 各 baseline 化简的详细步骤,见 [research/sdpo_loss_bayes_reading.md](sdpo_loss_bayes_reading.md)。代码实现:[`verl/trainer/ppo/core_algos.py`](../verl/trainer/ppo/core_algos.py)(`compute_teacher_qv_advantage` 在 480 行,vocab-level KL 在 1677 行)。*
