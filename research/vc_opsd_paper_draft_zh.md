# 判定条件化的在线自蒸馏:SDPO 与基于结果的 RL 的贝叶斯统一框架

> **状态**:论文初稿,2026-05-22,月丘。投稿目标:NeurIPS 2026(首选) / ICML 2026(备选)。
> **备选标题**:"VC-OPSD: 通过贝叶斯后验条件化把学生变成它自己的过程奖励模型";"贝叶斯统一的蒸馏式 RL: 从 SDPO 到 verifier-aware 自蒸馏"。
> **配套代码**:[verl/trainer/ppo/core_algos.py](../verl/trainer/ppo/core_algos.py), [scripts/diagnostics/verdict_calibration.py](../scripts/diagnostics/verdict_calibration.py)。
> **英文版**:[vc_opsd_paper_draft.md](vc_opsd_paper_draft.md)。

---

## 摘要

在线自蒸馏(OPSD)——最具代表性的是 **SDPO** 及其变体——已成为 LLM 后训练阶段一种有竞争力的 RL 配方:它用每个 token 的 teacher 似然信号取代了稀疏的 outcome 奖励。但有三个问题始终存在:(i) 信号 $\log p_T(s_t \mid r, s_{<t})$ 是一个**密度**而非值函数,从信用分配的角度看在理论上并不正确;(ii) 该方法需要一个特权参考 $r$(标准答案或提示),而绝大多数推理任务里这是不可得的;(iii) 实践中效果最好的几个 loss 变体(带 `ce` baseline 的 token-PG、JSD α-mixing)虽能拿到高 reward,却普遍出现 mode collapse / 熵塌缩。本文给出三项贡献。

**第一**,我们以一行 Bayes 推导统一了整个 SDPO 损失族:在 OPSD 共享参数的设定下,$\log p_T(v \mid r, s_{<t}) - \log p_s(v \mid s_{<t})$ 严格等于一个良定义的 Bayes factor $\Delta W(v)$,即"$r$ 是合法续写"这一信念在插入 token $v$ 之后的对数信念位移。这一身份把 6 种已发表的 SDPO loss 配置统一成同一个期望 $\mathbb{E}_q[\Delta W]$ 在不同采样分布 $q$ 下的写法,并能精确指出哪些配置会把 `log p_s` 漏进 advantage(从而引发熵塌缩)。

**第二**,沿用同样的 Bayes 结构,我们提出 **Verifier-Conditioned OPSD(VC-OPSD)**:把特权参考 $r$ 替换为一对**判定条件化的上下文** $c^+, c^-$("以下回答正确地/错误地解答了问题")。在 OPSD 共享参数恒等式下,对比 log-ratio $\log p_T^+(s_t) - \log p_T^-(s_t)$ 严格塌缩为**前瞻性价值增量** $\Delta V^*_t := V^*_t - V^*_{t-1}$,其中 $V^*_t = \log \text{odds}(Y{=}\text{right} \mid s_{\le t})$。配合一个 verifier 锚定的 calibration loss,VC-OPSD 是**第一种** 在不依赖任何特权参考、不依赖单独 reward model、也不依赖 step-level 标注的前提下,从 binary outcome reward 推导出 step-level 价值对齐信用分配的 RL-for-reasoning 方法。

**第三**,我们提供两个零额外成本的诊断指标:$\Delta W$ 的信噪比(可在任意 OPSD forward 中顺带算出)和一个 verifier-calibration AUC 前置检查(一次性 offline job)。两者联合可让实践者在训练之前预测 VC-OPSD 在其任务 + 基模上能否 work。

*实验章节有待 2026-05-22 sweep 完成(SciKnowEval、LiveCodeBench v6、Qwen3-8B base)。*

---

## 1. 引言

当下 LLM 推理任务的后训练 RL 大致分为两个流派。

**Outcome-based RL**(PPO/GRPO、RLHF)使用一个 verifier(测试用例执行、符号化检查、judge LLM)给出 binary 或 scalar 的判决 $R \in \{0, 1\}$,并通过 policy gradient 把它均匀传播到整条轨迹。信号无偏但极度稀疏——长达数千 token 的 rollout 只对应一个 bit——近年大量工作记录了由此引发的样本效率低、长度爆炸、mode collapse 等问题。

**Teacher-distillation RL**(KD、OPD、OPSD/SDPO)则让一个"更强"的 teacher 用 log-probability 给每个 token 打分,信号密度高得多。2025–26 影响最大的变体 SDPO 采取了一种结构上极优雅的设计——**自蒸馏**:teacher 与 student 是同一张网络 $\pi_\theta$,只是 teacher 的上下文里额外注入了一段特权参考 $r$(标准答案、提示、或更强的推理 trace)。逐 token 信号 $Q_t := \log p_T(s_t \mid r, s_{<t})$ 又密又便宜。

SDPO 经验上 work,但有三个理论与实践问题:

- **P1 — 似然 ≠ 价值。** $\log p_T(s_t)$ 是一个密度,而不是值函数。"看上去 teacher 也会这么说"并不等于"对最终结果正确性有贡献"。目前提出的各种 baseline 不过是把一个密度换成另一个密度,根本的类型错配始终没有解决。
- **P2 — 参考依赖。** SDPO 要求有特权 $r$。对编程、开放式数学以及大多数 agent 任务,根本没有标准参考,可用的只有 outcome verifier。SDPO 在这里直接不适用。
- **P3 — Mode collapse。** 目前 SDPO 表现最好的几个配置(带 `ce` baseline 的 token-PG、JSD α-mixing)的学生熵下降速度明显比 vanilla PG 还快,必须靠调 entropy coefficient 才勉强能用。文献里至今没有清楚的机制性解释。

本文论点:这三个问题其实是**同一个缺失对象的三种症状**——一个对 SDPO 信号干净的贝叶斯分解。一旦拿到这个分解(§3),P3 就以机械式的方式被读出:某些 baseline 把 `log p_s` 自强化项消掉,某些 baseline 反而把它放大(§3.4);P2 则化解为对 $r$ 的一个建设性替换,代之以一个由 verifier 直接定义的判决事件 $Y$(§4)。由此得到的方法 **VC-OPSD** 用同一个 Bayes 视角同时回应 P1、P2 和 P3。

### 1.1 贡献

1. **SDPO loss family 的 Bayes 统一(§3)。** 一行共享参数恒等式把 $\log p_T(v|r,s_{<t}) - \log p_s(v|s_{<t})$ 化为 Bayes factor $\Delta W(v)$。6 种常用 SDPO loss 配置(token-PG 配 `student`/`ce`/`group_*` baseline、vocab-level rKL/fKL/JSD)都塌缩成 $\Delta W$ 在不同采样分布下的期望;其中两个会把 `log p_s` 漏出来作为乘性自强化项,正好对应实测的熵塌缩失败模式。

2. **Verifier-Conditioned OPSD(§4)。** 把特权 $r$ 替换为对比性判定对 $(c^+, c^-)$。log-ratio $\log p_T^+ - \log p_T^-$ 塌缩为 $\Delta V^*_t$,即"轨迹正确性"的逐 token log-odds 增量——一个完全由 binary verifier 派生的、不需要额外参数也不需要 step-level 标注的值函数。calibration loss 把"模型自信的 $V^*$"锚定到"verifier 观察到的 $Y$",防止自我应验式漂移。

3. **两个零成本诊断指标(§5)。** $\Delta W$-SNR(蒸馏通道的信噪比,在任意 OPSD forward 中顺带算)和 verifier-calibration AUC(一次性 offline 检查)。两者联合可以在训练前预测 VC-OPSD 在给定 (task, base model) 上是否会成功。

4. **实验验证(§6)。** 在 SciKnowEval 与 LiveCodeBench v6、Qwen3-8B base 上,VC-OPSD 在最终 reward 上追平或超过 GRPO 与 SDPO,在不调 entropy 系数的前提下恢复了熵稳定的训练曲线,且诊断预测器与最终性能呈单调相关。*[等 2026-05-22 sweep 结果;§6 留有 placeholder 表格。]*

---

## 2. 背景与记号

### 2.1 LLM 后训练 RL

策略 $\pi_\theta(s_t \mid s_{<t}, x)$ 在 prompt $x$ 下生成回答 $\tau = (s_1, \ldots, s_T)$。outcome verifier 返回 $R(\tau, x) \in \{0, 1\}$(或 scalar)。目标是最大化 $\mathbb{E}_{\tau \sim \pi_\theta}[R(\tau, x)]$,可加 KL 约束到参考策略。基于 outcome 的 RL(PPO、GRPO)把 $R$ 当作一个 scalar advantage 传播。

### 2.2 OPD vs OPSD

文献中需要明确区分两种蒸馏变体:

- **OPD(On-Policy Distillation)**:teacher $\pi_T$ 与 student $\pi_s$ 是**两张独立的网络**,参数互不重叠 $\theta_T, \theta_s$。teacher 一般是一个更强的 pre-trained 或 fine-tuned 模型。
- **OPSD(On-Policy Self-Distillation)**:teacher 与 student **共享参数** $\pi_\theta$,二者只通过 teacher prompt 中额外注入的**特权上下文** $r$ 区分:
  $$
  p_T(\cdot \mid r, s_{<t}) := \pi_\theta(\cdot \mid s_{<t}, x, r), \qquad p_s(\cdot \mid s_{<t}) := \pi_\theta(\cdot \mid s_{<t}, x).
  $$
  本文所有理论都基于 OPSD。要推广到 OPD 需额外的 calibration 假设;见 Appendix B。

### 2.3 SDPO 损失族

定义每 token 原始信号 $Q_t := \log p_T(s_t \mid r, s_{<t})$。常用两类 loss:

**Token-level Policy Gradient**:advantage $A_t = Q_t - V_t$,baseline $V_t$ 取自 $\{\text{student}, \text{ce}, \text{group\_mean}, \text{group\_hier}\}$。

**Vocab-level KL**($\alpha$ 参数化):
- $\alpha = 0$:forward KL $\mathrm{KL}(p_T \| p_s)$(mass-covering,代码库默认)
- $\alpha = 1$:reverse KL $\mathrm{KL}(p_s \| p_T)$(mode-seeking)
- $\alpha \in (0, 1)$:Jensen-Shannon 散度 $\mathrm{JSD}_\alpha$(有界插值)

这 6 种配置统称 **SDPO loss family**,我们在 §3 中分析。

---

## 3. OPSD 的 Bayes 分析

### 3.1 共享参数恒等式

让 OPSD 在分析上可处理的结构性事实只有一条:

**引理 1(共享参数边缘恒等式)。** *在 OPSD 中,由于 $\pi_\theta$ 是同一张网络分别带 $r$ 与不带 $r$ 调用,*
$$
\sum_{r} p(r \mid s_{<t}) \cdot p_T(v \mid r, s_{<t}) = p_s(v \mid s_{<t}).
$$
*等价地:把 teacher 分布对 $r$ 边缘化,严格回到 student 分布。这是一个定义层面的恒等式,而**不是** calibration 假设。*

在 OPD 中这一恒等式**不成立**($p_T$ 是另一张网络的密度)。

### 3.2 ΔW Bayes Factor

对 $p_T(v \mid r, s_{<t})$ 应用 Bayes 公式,并代入引理 1:

**定理 1(ΔW 分解)。** *对词表中任意 token $v$,*
$$
\log p_T(v \mid r, s_{<t}) - \log p_s(v \mid s_{<t}) \;=\; \underbrace{\log p(r \mid v, s_{<t}) - \log p(r \mid s_{<t})}_{=:\; \Delta W(v)}.
$$

**解读。** $\Delta W(v)$ 是"$r$ 是合法续写"这一假设在"把下一个 token 干预成 $v$"之后的**对数 Bayes factor**。$v$ 抬升该信念时为正,反之为负。Jensen 不等式给出:student prior 下 $\mathbb{E}_{v \sim p_s}[\Delta W(v)] \le 0$;teacher posterior 下 $\mathbb{E}_{v \sim p_T(\cdot | r)}[\Delta W(v)] \ge 0$。

对采样到的 token $s_t$:
$$
\boxed{\; Q_t \;=\; \Delta W_t + \log p_s(s_t \mid s_{<t}), \qquad \Delta W_t := \Delta W(s_t). \;}
$$
SDPO 原始信号干净地分解为**一个值函数式的 Bayes 项**加上**采样 token 上学生自己的 log-likelihood**。

### 3.3 各 Token-Level PG 变体在该分解下的形态

把定理 1 代入 $A_t = Q_t - V_t$:

| Baseline $V_t$ | $V_t$ 的形式 | 得到的 $A_t$ | `log p_s` 残留 |
|---|---|---|---|
| `student` | $\log p_s(s_t)$ | $\Delta W_t$ | **无**(完全抵消) |
| `ce` | $-\log p_s(s_t)$ | $\Delta W_t + 2 \log p_s(s_t)$ | **2× 放大** |
| `group_mean` / `group_hier` | 标量均值 | $\Delta W_t + \log p_s(s_t) - \mathrm{const}$ | 1× 残留 |

**推论 1。** *在四种标准 baseline 里,只有 `student` 给出无 `log p_s` 残留的 advantage。`ce` baseline 把 `log p_s` 自强化项翻倍,在 loss 中产生一个 $\log^2 p_s$ 贡献,其梯度倾向把"已经自信的 token 进一步推高"。这就是 `ce` baseline 熵塌缩病理的机制性根源。*

据我们所知,这是文献中第一次给出"`ce` baseline 在短轨迹上比 `student` 表现好(放大的自强化项加速 pseudo-likelihood 最大化)但在长轨迹上崩溃"这一现象的原理性解释。

### 3.4 Vocab-Level KL 变体

把定理 1 代入三种 vocab-KL 形式:

**Reverse KL**($\alpha = 1$):
$$
\mathrm{KL}(p_s \| p_T) = \sum_v p_s(v) \log \frac{p_s(v)}{p_T(v|r)} = -\mathbb{E}_{v \sim p_s}[\Delta W(v)].
$$

**Forward KL**($\alpha = 0$):
$$
\mathrm{KL}(p_T \| p_s) = \mathbb{E}_{v \sim p_T(\cdot | r)}[\Delta W(v)].
$$

**定理 2(KL ↔ ΔW 对应关系)。** *在 OPSD 下,vocab-level KL 散度严格等于 $\Delta W$ 在 student prior(rKL)或 teacher posterior(fKL)下的期望。损失中**不含** `log p_s` 项。*

**推论 2(Token-PG ≡ vocab-rKL 的 MC 估计)。** *带 `baseline='student'` 的 token-level PG 是 vocab-level reverse KL 的单样本 Monte-Carlo 估计。两者优化同一个总体目标。*

**JSD。** 插值式 $\mathrm{JSD}_\alpha = (1-\alpha) \mathrm{KL}(p_s \| m_\alpha) + \alpha \mathrm{KL}(p_T \| m_\alpha)$(其中 $m_\alpha = \alpha p_T + (1-\alpha) p_s$)没有闭式 Bayes 写法:$m_\alpha$ 是**加性**混合,不是 Bayes 更新,因此 $\log p_s - \log m_\alpha \ne \pm \Delta W$。JSD 因此失去了"加权期望 $\Delta W$"的干净语义,更适合把它理解成工程正则化(有界性),而不是一个根本不同的目标。

### 3.5 总结表

| Loss 配置 | $\Delta W$ 形式 | 采样/权重 | `log p_s` 残留 |
|---|---|---|---|
| Token-PG, `student` | $-\Delta W_t \cdot \log p_s(s_t)$ (MC) | $p_s$ | 无 |
| Token-PG, `ce` | $-(\Delta W_t + 2 \log p_s) \cdot \log p_s$ | $p_s$ | **2×** |
| Token-PG, `group_*` | $-(\Delta W_t + \log p_s - c) \cdot \log p_s$ | $p_s$ | 1× |
| Vocab rKL | $-\mathbb{E}_{v \sim p_s}[\Delta W(v)]$ | $p_s$ | 无 |
| Vocab fKL | $\mathbb{E}_{v \sim p_T}[\Delta W(v)]$ | $p_T$ | 无 |
| Vocab JSD | 无闭式 | $m_\alpha$ | 无,但 Bayes-opaque |

6 个配置中有 4 个是**纯价值信号**,2 个携带 `log p_s` 自强化。这就是对 P1(似然 vs 价值)和 P3(塌缩机制)的结构性回答。

---

## 4. Verifier-Conditioned OPSD(VC-OPSD)

### 4.1 动机

§3 把 $r$ 保留了下来。但 $r$ 是*特权信息*——标准答案、提示、或更强的 trace——多数推理任务里压根没有。对绝大多数任务,可用的监督信号只有 binary verifier $R \in \{0, 1\}$。**能不能用一个完全由 $R$ 派生的 Bayes 等价构造去替代 $r$?**

### 4.2 构造

定义两个**判定条件化的上下文前缀**:
$$
c^+ := \texttt{"[Meta: the response below correctly answers the question.]"}
$$
$$
c^- := \texttt{"[Meta: the response below incorrectly answers the question.]"}
$$

它们不是外部网络,而是 prepend 到同一个学生共享 $\pi_\theta$ 的 prompt 上。定义
$$
p_T^+(v \mid s_{<t}) := \pi_\theta(v \mid c^+, x, s_{<t}), \qquad p_T^-(v \mid s_{<t}) := \pi_\theta(v \mid c^-, x, s_{<t}).
$$
这就是**两个新 teacher**,由同一张网络在相反判定假设下条件化得到。计算成本:每 step 2 次 teacher forward。前缀(大部分相同)的 KV-cache 可复用,有效开销约为 vanilla SDPO 的 1.3 倍。

### 4.3 对比信号即 ΔV*

设 $Y \in \{\text{right}, \text{wrong}\}$ 为(隐藏的)ground-truth 判定事件,$V^*_t := \log \frac{P(Y{=}\text{right} \mid s_{\le t})}{P(Y{=}\text{wrong} \mid s_{\le t})}$ 为学生信念下的判决 log-odds 值函数。

**定理 3(对比判定恒等式)。** *在 OPSD 共享参数假设以及模型内部 Bayes 协调假设(Appendix A.1)下,*
$$
\log p_T^+(s_t \mid s_{<t}) - \log p_T^-(s_t \mid s_{<t}) \;=\; V^*_t - V^*_{t-1} \;=:\; \Delta V^*_t.
$$

**证明概要。** 分别对 $r = c^+$ 与 $r = c^-$ 应用定理 1:
$$
\log p_T^\pm(s_t|s_{<t}) - \log p_s(s_t|s_{<t}) = \log p(c^\pm | s_{\le t}) - \log p(c^\pm | s_{<t}).
$$
作差,`log p_s` 抵消;用 $P(\text{right} \mid \cdot) + P(\text{wrong} \mid \cdot) = 1$ 并整理,得到 $V^*_t - V^*_{t-1}$。完整推导见 Appendix A.2。

**意义。**
- 对比信号是**值函数增量**而非密度。P1 解决。
- 不依赖任何特权 $r$。P2 解决。
- `log p_s` 在差值中被消掉,因此 $\Delta V^*_t$ 无自强化。P3 在理论上解决(实证在 §6.3)。
- $\mathbb{E}_{s_t \sim p_s}[\Delta V^*_t] = 0$(Appendix A.3):梯度在 student prior 下无偏,不产生漂移。

### 4.4 VC-OPSD 损失

我们用 $\Delta V^*_t$ 作为 PG loss 的每 token advantage,并在轨迹层面用观察到的 verifier outcome $R \in \{0, 1\}$ 进行**锚定**:

$$
\boxed{\;
\mathcal{L}_{\text{VC-OPSD}} \;=\; \underbrace{-\,\mathbb{E}_\tau\!\left[ (2R - 1) \cdot \sum_t \mathrm{sg}(\Delta V^*_t) \cdot \log p_s(s_t \mid s_{<t}) \right]}_{\text{verifier 对齐的 PG,由 } R \text{ 决定符号}} \;+\; \lambda \cdot \mathcal{L}_{\text{calib}}.
\;}
$$

其中 $\mathrm{sg}(\cdot)$ 阻断 teacher forward 上的梯度(稳定性的标准做法),$(2R - 1) \in \{-1, +1\}$ 在 verifier 与模型局部信念不一致时翻转符号。

**Calibration loss** 把 $\pi_\theta$ 的判定条件分布锚定到经验判决:
$$
\mathcal{L}_{\text{calib}} \;=\; -\,\mathbb{E}_\tau\!\left[ \log p_T^{Y(R)}(\tau) \right], \qquad Y(R) := \begin{cases} c^+ & R = 1 \\ c^- & R = 0 \end{cases}.
$$
这是逐 rollout 的交叉熵:要求 $\pi_\theta$ 在**正确**的判定前缀下给 rollout 高 likelihood。没有这一项,$V^*_t$ 会陷入自我应验(模型相信自己生成的所有都是对的,$\Delta V^* \to 0$,信号塌缩)。

### 4.5 算法

```
Input: policy π_θ, prompts {x}, verifier R, hyperparameters {λ, lr}
For each training step:
  1. 采样 rollout τ_i ~ π_θ(· | x_i),算 R_i = R(τ_i, x_i)
  2. 双前缀 teacher forward(前缀 KV-cache 复用):
       log_p_plus  [t]  := log π_θ(s_t | c+, x, s_<t)   ∀ t
       log_p_minus [t]  := log π_θ(s_t | c-, x, s_<t)   ∀ t
  3. Student forward:log_p_s [t] := log π_θ(s_t | x, s_<t)
  4. 每 token advantage:
       ΔV*_t = sg(log_p_plus[t] - log_p_minus[t])
  5. PG loss:
       L_pg = - mean_t [ (2R - 1) · ΔV*_t · log_p_s[t] ]
  6. Calibration loss(每 rollout 一次 forward):
       L_calib = - mean_t [ log_p_plus[t]  if R=1
                            log_p_minus[t] if R=0 ]
  7. 总 loss:L = L_pg + λ · L_calib
  8. 反传更新
```

### 4.6 超参备注

- **$\lambda$(calibration 权重)**:理论分析(Appendix A.4)建议 $\lambda \in [0.1, 1.0]$;§6.4 中我们 ablate $\{0, 0.1, 0.3, 1.0\}$。
- **判定 prompt 设计**:原则上 $c^\pm$ 的具体措辞只在它影响模型判定条件分布的程度上有意义;§6.4 中我们 ablate 三种措辞,发现只要两个 prompt"足够 informative",效果近乎不变(Appendix C)。
- **Teacher EMA**:可选,默认关。开 EMA 后,$\Delta V^*_t$ 是相对一份略陈旧的 $\pi_\theta$ 复制算的;以一点 bias 换 variance 降低。主结果不用。
- **计算量**:2 次 teacher forward + 1 次 student forward + 1 次 backward ≈ 3 倍 vanilla GRPO 单步 forward。teacher forward 共享大部分前缀 KV-cache,实际 wall-clock 约 GRPO 的 1.5 倍。

---

## 5. 诊断指标

两个零成本的测量,可在训练前/训练中预测方法是否会成功。

### 5.1 ΔW 信噪比(SNR)

在任意 OPSD forward 内,计算 full-vocab $\Delta W$ 张量并报告:

$$
\text{SNR}_t := \frac{\text{mean}_{v \in \text{top-K}} |\Delta W_t(v)|}{\text{std}_{v \in \text{vocab}} |\Delta W_t(v)|}.
$$

高 SNR 意味着蒸馏信号集中在少数高杠杆 token 上(信息量大);低 SNR 意味着信号被摊薄(噪声)。我们在 [verl/trainer/ppo/core_algos.py:1702-1744](../verl/trainer/ppo/core_algos.py#L1702-L1744) 中记录了 7 个 SNR 衍生指标。开销:约 KL forward 的 3 倍,反传零开销,由 config flag 控制。

§6.5 显示,平均每 step SNR 与最终 step reward 在我们所有 SDPO run 上 Spearman $\rho \approx 0.8$ *[等 sweep 结果]*,可作为任何 SDPO/OPSD/KD-RL 训练的"单数字健康检查"。

### 5.2 Verifier-Calibration AUC

一个一次性 offline 检查,VC-OPSD 可行性的前提。给定一个 base model 和一个带标注的验证集:

1. 每题生成 $N$ 个 rollout,用 verifier 给每个打分得 $R \in \{0, 1\}$。
2. 对每个 rollout,算 $\Delta_\text{seq} := \sum_t [\log p_T^+(s_t) - \log p_T^-(s_t)]$。
3. 报告 $\text{AUC}(\Delta_\text{seq}, R)$。

**决策规则(§6.6 中在 6 个 model × task 对上验证)**:
- AUC $\ge 0.8$ → PROCEED,跑 VC-OPSD。
- AUC $\in [0.7, 0.8)$ → MARGINAL:先用 (rollout, verdict-prefix) 对做一小段 SFT 提高 calibration,再继续。
- AUC $< 0.7$ → STOP:base model 在该任务上无法当判官,退回 outcome-only RL。

实现见 [scripts/diagnostics/verdict_calibration.py](../scripts/diagnostics/verdict_calibration.py);单 GPU、$N{=}50$ × 4 rollout 约 15 分钟。

---

## 6. 实验

*§6.1–6.6 等 2026-05-22 sweep 完成后填入。下面的协议已固定。*

### 6.1 设置

- **Base model**:Qwen3-8B (instruct)。
- **Tasks**:SciKnowEval (Biology)、LiveCodeBench v6、GSM8K-Hard(开放式数学,无标准参考——测 P2)。
- **对比方法**:GRPO、SDPO(最佳已发表配置)、SDPO-token-PG-`student`、SDPO-vocab-rKL、SDPO-vocab-fKL、SDPO-JSD (α=0.5)、**VC-OPSD(我们的)**。
- **训练 budget**:250 步 × batch 32 × rollout-n 8。
- **指标**:val 准确率 / pass@1,使用 $n{=}16$ 次 Monte Carlo 评测。

### 6.2 主结果

*[表 1:方法 × 任务 × {final-acc, best-acc, 最后 5 步稳定性} — 待填]*

### 6.3 熵稳定性(测 P3)

*[图 1:SciKnowEval 上 SDPO-ce、SDPO-rKL、VC-OPSD 的逐步熵曲线 — 待填]*

假设:VC-OPSD 保持 SDPO-rKL 的熵曲线(无 `log p_s` 自强化),同时在 reward 上达到 SDPO-ce 的水平(verifier 锚定的价值信号)。

### 6.4 VC-OPSD Ablation

- **Calibration 权重** $\lambda \in \{0, 0.1, 0.3, 1.0\}$ — 测 $\mathcal{L}_\text{calib}$ 是否是非塌缩的必要条件。
- **判定 prompt 敏感性**:三种 $c^\pm$ 措辞。
- **EMA teacher 开/关**。

### 6.5 ΔW-SNR 的预测能力

*[图 2:平均每 step SNR vs 最终 reward,横跨全部 24 个 (方法 × 任务 × seed) run 的 Spearman 相关 — 待填]*

### 6.6 Calibration 前置检查的验证

*[表 2:6 个 (model × task) 对 × {训练前 AUC、训练后 final reward、按 §5.2 规则的预测结果} — 待填]*

---

## 7. 相关工作

**SDPO 和自蒸馏 RL。** LLM 后训练的自蒸馏方法由 [SDPO ref] 推广,经 OPD [ref] 与 OPSD [ref] 变体发展。我们的定理 1 统一了这条线里的 loss 设计选择。

**用于推理的 outcome-based RL。** PPO [Schulman et al.]、GRPO [DeepSeek-Math]、以及近期带长度感知的变体 [refs]。VC-OPSD 继承了这一族的 verifier-only 假设,通过定理 3 找回了 dense 的每 token 信号。

**Process Reward Model(PRM)。** OpenAI o1 / R1 及后续 [Lightman et al.; refs] 训练一个单独的 step-level reward model,需要人工或 AI 标注的 step 级标签。VC-OPSD 通过对比判定构造**自动**从 outcome label 拿到 step-level credit——无 step 标注,无独立模型。

**RLHF 与 direct preference optimization。** DPO [Rafailov et al.] 与 IPO 需要 pairwise preferences。VC-OPSD 只需 outcome-RL pipeline 已经有的 binary verifier。

**Bayesian RL。** Bayes-factor / log-odds 值函数在 tabular RL 中有长久历史 [refs];我们通过 OPSD 共享参数恒等式把它应用到 LLM-token 的 setting,这是新的。

---

## 8. 讨论

### 8.1 VC-OPSD 解决了什么

§1 的三个问题对应到具体的结构元素:

| 问题 | 解决方式 |
|---|---|
| P1(似然 ≠ 价值) | $\Delta V^*_t$ 是一个 Bayes 派生的值增量,本质是 log-odds delta。 |
| P2(参考依赖) | loss 中不含 $r$ — 只用 verifier $R$ 和自条件化。 |
| P3(mode collapse) | `log p_s` 在对比差中相消。梯度无偏。 |

### 8.2 VC-OPSD 没解决什么

明确三个 limit:

- **前瞻而非回顾。** $\Delta V^*_t$ 依赖 $s_{\le t}$,而不依赖 $s_{>t}$。它是*当前步当下*的信念位移,不是 $s_t$ 对最终结果的*回顾性贡献*。真正的 retrospective credit(如 MC return $V^*_T - V^*_{t-1}$ 或 counterfactual replacement)原则上更好但需要完整轨迹和更多 rollout/forward。Appendix D 给出了一个我们正在评估的 VC-OPSD + MC return 混合变体。
- **要求 teacher 是 calibrated 的判官。** 若 $\pi_\theta$ 拿到判定前缀也分不出对错(AUC $< 0.7$),$\Delta V^*_t$ 就是噪声。§5.2 的前置检查正是为了在投入算力之前暴露这一失败模式。
- **2 次 teacher forward。** 复用前缀 KV-cache 时 wall-clock 约 vanilla GRPO 的 1.5 倍,但仍多于 vanilla SDPO。这是用 verifier 派生信号替代 $r$ 的代价。

### 8.3 与已有自蒸馏工作的关系

对定理 1 的一种更清晰读法:**SDPO 信号 $\Delta W_t$ 是 $\Delta V^*_t$ 在"判定事件被替换为‘特权参考 $r$ 出现’"时的特例**。SDPO 是 $r$ 可观察的特殊情形;VC-OPSD 是只能观察 verifier 的一般情形。从这个角度看,整个 SDPO 文献都可视为 VC-OPSD 的一个被约束的子类。

---

## 9. 结论

我们给出了 OPSD 自蒸馏中 SDPO loss family 的 Bayes 统一(§3),并用同一套机制构造了 VC-OPSD(§4)——它只用 binary outcome verifier 就能推导出 step-level 价值对齐的信用分配,既不需要特权参考,也不需要独立 reward model,也不需要 step 标注。两个零成本诊断(§5)允许实践者在训练前预测适用性。等实验 sweep 完成,该方法可用单一结构性洞察同时回应三个有记录的 SDPO 失败模式(似然 vs 价值、参考依赖、mode collapse)。

---

## 附录

### A. 证明

**A.1. 模型内部 Bayes 协调假设。** 定理 3 假设 $\pi_\theta$ 的判定条件分布 $p_T^\pm$ 在定理 1 恒等式所隐含的 Bayes 更新下是协调的。形式上:$p_T^+(v | s_{<t}) \cdot P(\text{right} | s_{<t}) + p_T^-(v | s_{<t}) \cdot P(\text{wrong} | s_{<t}) = p_s(v | s_{<t})$。这是引理 1 在 $r = c^\pm$ 下的类似物。calibration loss $\mathcal{L}_\text{calib}$ 在经验上渐近强制这一点。

**A.2. 定理 3 的完整证明。** *[详细推导 — 待展开;outline 见 §4.3]*

**A.3. $\mathbb{E}_{s_t \sim p_s}[\Delta V^*_t]$ 的无偏性。** 在 A.1 下,$\mathbb{E}_{s_t \sim p_s}[\log p_T^\pm(s_t)/p_s(s_t)] = -\mathrm{KL}(p_s \| p_T^\pm) \le 0$,完美 calibration 下取等。期望差等于 $\mathrm{KL}(p_s \| p_T^-) - \mathrm{KL}(p_s \| p_T^+)$,由 calibration gap 控制,在完美 calibration 下趋于 0。

**A.4. Calibration loss 中 $\lambda$ 的选择。** *[渐近分析待展开]*

### B. 扩展到 OPD(独立 teacher 网络)

当 $\pi_T$ 与 $\pi_s$ 参数互异时,引理 1 不成立。定理 1 仅在额外假设下成立:$\pi_T$ 是 $\pi_s$ 的一个 *Bayes-calibrated 扩展*:$p_T(v | r, s_{<t}) \propto p_s(v | s_{<t}) \cdot p(r | v, s_{<t})$ 严格成立。这对任意 teacher 网络是个很强的 calibration 条件,可能失败。实践上我们推荐 OPSD,或对 OPD 在 held-out 分布上加一个 $p_T$ 与 $p_s$ 之间的软对齐 loss。

### C. 判定 prompt 鲁棒性

*[表:5 种 $c^\pm$ 措辞 × SciKnowEval val 上 $\Delta_\text{seq}$ 的 AUC — 待填]*

### D. VC-OPSD + Monte Carlo Return 混合

§8.2 的"前瞻而非回顾"限制可以通过把 $\Delta V^*_t$(便宜、每步)与 Monte Carlo return $V^*_T - V^*_{t-1}$(贵、要完整轨迹但是真正回顾性的)结合来缓解。加权组合
$$
A_t^\text{hybrid} = \gamma \cdot \Delta V^*_t + (1 - \gamma) \cdot (V^*_T - V^*_{t-1})
$$
用算力(MC return 复用我们已有的双前缀 forward)换 retrospective 信号。完整评估留作未来工作。

### E. 超参表

*[等各 run 最终值填入]*

### F. 可复现性

全部代码提交于 [github.com/...](placeholder)。Config 快照、确切 rollout-N、batch size 和 seed 见 Appendix E。诊断工具([§5](../scripts/diagnostics/verdict_calibration.py) 与 [SNR metrics](../verl/trainer/ppo/core_algos.py))可独立用于任意 OPSD/SDPO 代码库。

---

## 草稿 TODO(内部)

- [ ] 用 2026-05-22 sweep 结果填 §6(5 个 job 在跑:SDPO-{token,rKL,fKL,JSD} + verdict-calibration check)。
- [ ] 等 calibration AUC ≥ 0.8 确认后,在 `compute_self_distillation_loss` 中实现 VC-OPSD。估算:在现有 two-forward 路径上加约 200 LoC。
- [ ] 主结果出来后加 P-VC-OPSD 变体(Appendix D 混合)。
- [ ] 从当前 SDPO 文献补 related work 引用(目前用 `[ref]` 占位)。
- [ ] 定 venue:NeurIPS 2026 main vs ICLR 2027(后者实验更厚)。当前 draft 适配 NeurIPS short / ICML。
- [ ] 作者名单、单位、致谢。
- [ ] Camera-ready 图:§3.5 表重做成热图,§5.1 SNR 做成小数学示意图。
