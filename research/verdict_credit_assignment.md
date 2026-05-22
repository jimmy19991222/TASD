# Verdict-Based Retrospective Credit Assignment

> **TL;DR** — 用户提出一个 novel idea:在 student rollout 之后 append `[wrong/right]` verdict token,让 teacher forward 整条序列,从 teacher 视角下"verdict 与每个 prior token 的关系"中提取 per-token credit weight。本文细致拆解这条思路,提出一个干净的操作化(contrastive prepended verdict),**经 Bayes 推导后发现该操作化等价于 prospective ΔV\*_t**——这是一个非平凡的 double-check 结果,并由此推出真正符合"retrospective + 看完整轨迹"的两个候选方案(Monte Carlo return + counterfactual replacement)。

- **作者**: 月丘
- **日期**: 2026-05-22
- **状态**: 思路阶段,未实验验证
- **相关代码**: [`verl/trainer/ppo/core_algos.py:480-820`](../verl/trainer/ppo/core_algos.py#L480-L820)([`compute_teacher_qv_advantage`](../verl/trainer/ppo/core_algos.py#L480) + [`compute_teacher_qv_full_logit_loss`](../verl/trainer/ppo/core_algos.py#L688))
- **前置阅读**: [`experiments/teacher_qv_ablation/REPORT.md`](../experiments/teacher_qv_ablation/REPORT.md), [`experiments/teacher_qv_ablation/LOSS.md`](../experiments/teacher_qv_ablation/LOSS.md)

---

## 0. 问题缘起

Phase 1~4 的 teacher_qv 实验已经把 `A_t = log p_t(a_t) − V_t` 这条路线探索得比较透彻。最大的 systemic finding 是:

> **`log p_t(a_t)` 是 likelihood,不是 value function**。teacher 觉得这个 token "像它会说的"不代表这个 token "对最终 verifier reward 的贡献大"。所有 `baseline_type` 的设计实际上都在不同的 *density* 概念之间换花样,但没有一个真正在做 **value-based credit assignment**。

这个 framing 是用户在与我的对话中提出的核心批评(详见 [`research/teacher_qv_analysis.md`](teacher_qv_analysis.md))。本文的方案试图正面回应这条批评:**让 teacher 直接对最终结果做 judge,然后用这个 judge 反推 token-level credit,不再依赖 likelihood 当代理变量**。

> **架构前提(贯穿全文)**:本文讨论的是 **OPSD**(on-policy **self**-distillation,本 codebase 的 `self_distillation_config` 设定),**不是**传统 OPD(on-policy distillation,后者 teacher 是独立训练的另一个网络)。OPSD 框架下 **teacher 和 student 不是两个独立网络,而是同一个 policy `π_θ` 在两种 conditioning 下的 forward**——`student = π_θ(· | s_<t, question, system_prompt)`,`teacher = π_θ(· | s_<t, question, system_prompt, privileged_info r)`(忽略可能的 teacher EMA 延迟)。下文继续沿用 `p_T` / `p_s` 这两个符号是为了对齐 KD 文献叙述,**但实际上它们调用的是同一份模型 weight**。这个约束在 §3.4.1.5 起会**重写整个分析的结论**,尤其是 §3.4.4 的 baseline 比较。
>
> ⚠️ 后文有些段落历史上写作"OPD 框架",**应统一理解为 OPSD**。传统 OPD(独立 teacher 网络)下本文的 shared-param 结论不成立,需要额外 calibration 假设。

---

## 1. 用户原始 idea 的精确陈述

> "把 student rollout 的一个 seq,后面增加一个 `[which is wrong/right]`,然后判断在 teacher 视角下 forward,判断这个 token 和之前 token 的相关性,作为 credit assign 的 weight。"
>
> "只有 rollout 出了一整个 seq,teacher 才能知道每一个 token 对最终答案有多少贡献。"

拆解出三个核心 commitment:

1. **Sequence-completed**: 必须先把整条 rollout 跑完,credit 才能被 well-defined。这排除了 prefix-running / on-policy V*_t 路线。
2. **Teacher-as-judge**: teacher 不再是 density model,而是给 trajectory-level verdict 的 critic。
3. **Per-token decomposition**: 最终要把 sequence-level 的 verdict 信号分解到 token 级别。

这三个 commitment 合起来对应 RL 文献里的 **retrospective credit assignment / counterfactual attribution**,与我之前推过的 prospective `V*_t` 是不同的概念家族。

---

## 2. 第一次操作化尝试:Contrastive Prepended Verdict

### 2.1 设计

不在末尾 append,而是在开头 prepend 两种条件性 prompt,做两次 teacher forward,取 log-prob 之差:

```
ctx_right = "下面这段推理最终得出了正确答案: "
ctx_wrong = "下面这段推理最终得出了错误答案: "

# 两次 teacher forward
log_p_right[t] = log p_T(s_t | ctx_right, s_<t)        # (B, T)
log_p_wrong[t] = log p_T(s_t | ctx_wrong, s_<t)        # (B, T)

# Per-token contrastive credit
Δ_t = log_p_right[t] − log_p_wrong[t]                  # (B, T)

# 按 verifier 给的 R 决定方向
A_t = (2R − 1) · Δ_t                                    # (B, T)
```

**含义**:`Δ_t > 0` ⇔ token t 在 teacher 眼里更"像对的轨迹会出现的 token";verifier 说对了(R=1)就强化这种 token,说错了(R=0)就抑制。

### 2.2 选 prepend 而不是 append 的工程理由

如果按用户原话 append `[wrong/right]` 在末尾,要从末尾位置往前提取 per-token attribution,可选方案都有大坑:

| 操作化 | 计算方式 | 致命问题 |
|--------|----------|----------|
| Attention rollout | verdict 位置对 t 位置的 attention | [Jain & Wallace 2019](https://arxiv.org/abs/1902.10186) 已系统反驳 attention as attribution |
| Gradient saliency | `‖∇_{x_t} log p_T(verdict)‖` | 噪声大,且需要额外 backward |
| Leave-one-out | `log p_T(verdict\|seq) − log p_T(verdict\|seq\{t})` | O(T) 次额外 forward,不可接受 |

prepend 之后,**每个 token 的 log-prob 是直接读出的标量,不需要任何 attribution 算法**——这是工程层面的胜利。

---

## 3. ⚠️ Double-check:Contrastive Prepended Verdict 等价于 Prospective ΔV\*_t

**这是本文最重要的发现,推翻了我对用户的回答。**

### 3.1 Bayes 推导

#### 3.1.1 记号约定

| 记号 | 含义 |
|------|------|
| `s_<t` | `{s_1, s_2, ..., s_{t-1}}`,严格小于 t 的所有 token(**不含** s_t) |
| `s_≤t` | `{s_1, s_2, ..., s_t}`,小于等于 t 的所有 token(**含** s_t) |
| `c ∈ {right, wrong}` | verdict 事件 |

关键关系:**`s_≤t = s_<t ∪ {s_t}`** —— 这条等式是后面 Step 2 里 `s_≤t` 出现的全部原因,不是新引入的符号,只是 "`s_<t` 拼上 `s_t`" 的简写。

#### 3.1.2 从链式法则到 Bayes 恒等式

**Step 1:写出条件概率链式法则的两个方向**

对任意联合分布,链式法则可以从两个方向展开同一个联合概率:

$$
p(X, Y \mid Z) = \underbrace{p(X \mid Z) \cdot p(Y \mid X, Z)}_{\text{方向 A:先采 } X \text{,再 condition } Y} = \underbrace{p(Y \mid Z) \cdot p(X \mid Y, Z)}_{\text{方向 B:先 condition } Y \text{,再采 } X}
$$

代入 `X = s_t`、`Y = c`、`Z = s_<t`:

- **方向 A**: $p(s_t, c \mid s_{<t}) = p(s_t \mid s_{<t}) \cdot p(c \mid s_t, s_{<t})$
- **方向 B**: $p(s_t, c \mid s_{<t}) = p(c \mid s_{<t}) \cdot p(s_t \mid c, s_{<t})$

**Step 2:替换 `(s_t, s_<t) → s_≤t`**

方向 A 最后一项的条件里同时出现了 `s_t` 和 `s_<t`。由 §3.1.1 的记号约定,`s_<t ∪ {s_t} = s_≤t`,所以:

$$
p(c \mid s_t, s_{<t}) = p(c \mid s_{\le t})
$$

方向 A 简化为:

$$
p(s_t, c \mid s_{<t}) = p(s_t \mid s_{<t}) \cdot p(c \mid s_{\le t})
$$

**Step 3:令方向 A = 方向 B,整理出 Bayes 恒等式**

两条展开都等于 $p(s_t, c \mid s_{<t})$,所以:

$$
p(s_t \mid s_{<t}) \cdot p(c \mid s_{\le t}) = p(c \mid s_{<t}) \cdot p(s_t \mid c, s_{<t})
$$

把 `p(s_t | c, s_<t)` 解出来,得到 **Bayes 恒等式**:

$$
p_T(s_t \mid c, s_{<t}) = \frac{p_T(c \mid s_{\le t})}{p_T(c \mid s_{<t})} \cdot p_T(s_t \mid s_{<t})
$$

取对数:

$$
\log p_T(s_t \mid c, s_{<t}) = \log p_T(c \mid s_{\le t}) - \log p_T(c \mid s_{<t}) + \log p_T(s_t \mid s_{<t})
$$

这条等式对**任意联合分布**成立,与 token order、依赖结构(Markov / non-Markov / long-range)都无关——它只是条件概率定义本身的恒等变形。**前提是 teacher 内部 conditional probabilities 互相 Bayes-coherent**(real LM 上未必,详见 §3.3 末尾与 §8.1)。

#### 3.1.3 Contrastive verdict 做差消掉 likelihood prior

把 Bayes 恒等式分别代入 `c = right` 和 `c = wrong`:

$$
\log p_T(s_t \mid \text{right}, s_{<t}) = \log p_T(\text{right} \mid s_{\le t}) - \log p_T(\text{right} \mid s_{<t}) + \log p_T(s_t \mid s_{<t})
$$

$$
\log p_T(s_t \mid \text{wrong}, s_{<t}) = \log p_T(\text{wrong} \mid s_{\le t}) - \log p_T(\text{wrong} \mid s_{<t}) + \log p_T(s_t \mid s_{<t})
$$

注意最后一项 `log p_T(s_t | s_<t)` 不依赖 `c`,做差时**自然消掉**——这是 contrastive 操作的核心收益:

$$
\begin{aligned}
\Delta_t &= \log p_T(s_t \mid \text{right}, s_{<t}) - \log p_T(s_t \mid \text{wrong}, s_{<t}) \\
&= \big[\log p_T(\text{right} \mid s_{\le t}) - \log p_T(\text{right} \mid s_{<t})\big] - \big[\log p_T(\text{wrong} \mid s_{\le t}) - \log p_T(\text{wrong} \mid s_{<t})\big] \\
&= \big[\log p_T(\text{right} \mid s_{\le t}) - \log p_T(\text{wrong} \mid s_{\le t})\big] - \big[\log p_T(\text{right} \mid s_{<t}) - \log p_T(\text{wrong} \mid s_{<t})\big] \\
&= V^{*}_t - V^{*}_{t-1}
\end{aligned}
$$

其中定义:

$$
V^{*}_t := \log p_T(\text{right} \mid s_{:t}) - \log p_T(\text{wrong} \mid s_{:t})
$$

是 teacher 在前缀 `s_:t` 处对"最终判决为 right"和"最终判决为 wrong"两个事件的 logit 差,即 **prospective verdict logit difference**。

#### 3.1.4 具体例子检验等价性

序列 `s = [s_1=2, s_2=+, s_3=3, s_4==, s_5=5]`,在位置 `t=5` 看两条路径如何"读"出同一个数:

- `s_<5 = [2, +, 3, =]`(4 个 token)
- `s_≤5 = [2, +, 3, =, 5]`(5 个 token,补上了 `s_5=5`)

**LHS(CPV 路线):2 次 forward**
- Forward 1 — 输入 `[ctx_right, 2, +, 3, =, 5]`,读位置 5 处 `s_5="5"` 的 log-prob → `log p_T(5 | ctx_right, s_<5)`
- Forward 2 — 输入 `[ctx_wrong, 2, +, 3, =, 5]`,读位置 5 处 `s_5="5"` 的 log-prob → `log p_T(5 | ctx_wrong, s_<5)`
- 做差 → `Δ_5`

**RHS(Prospective 路线):1 次 forward**
- 输入 `[2, +, 3, =, 5]`(student 原序列,无任何 prefix)
- 读位置 4 处 vocab 里 `[right]`/`[wrong]` 两格的 logit,做差 → `V*_4`
- 读位置 5 处 vocab 里 `[right]`/`[wrong]` 两格的 logit,做差 → `V*_5`
- 做差 → `V*_5 − V*_4`

**Bayes 恒等式断言**:在 teacher 内部 Bayes-coherent 的假设下,**`Δ_5 = V*_5 − V*_4`**。

两条路径读的是不同 forward / 不同 position / 不同 vocab 维度的数,但 chain rule 把它们 algebraically 联系起来。

**为什么仍然推荐 Prospective**:
- **Compute 减半**:1 forward × 1 input 序列 vs 2 forward × 2 input 序列
- **无 prefix 扰动**:Prospective 不引入 `ctx_right`/`ctx_wrong` 这种自然语言 prefix——这种 prefix 会带来 style/topic 上的 distribution shift,real LM 上"等价性"会被这层扰动稀释(详见 §3.3 末尾的 Bayes coherence 讨论)
- **直接复用现有 KD pipeline**:teacher forward 本来就有,免费拿到 `[right]/[wrong]` 的 vocab logit

这是 §6 推荐 `baseline_type='verdict_mc'` 的核心工程理由。

### 3.2 这意味着什么

**Contrastive prepended verdict 在数学上严格等于 prospective verdict logit 的单步增量**。

我在上一轮回答里"承认"用户的批评、推翻了 V*_t 路线、改提"contrastive prepended"——这个改名其实没有获得任何新的信息。两条路线计算上等价,语义上也等价,只是叙事方向不同(一个写成"知道结果再回头看",一个写成"前缀走到 t 时的实时判断")。

**用户的直觉"只有看完整条 seq teacher 才能知道贡献"是对的,但 Bayes 告诉我们:teacher 在每个 prefix 处对最终结果的 belief shift,通过 telescoping sum (`Σ ΔV*_t = V*_T − V*_0`) 收敛到看完整条序列后的 belief**。也就是说,`ΔV*_t` **本身就是 retrospective**,只不过它是 *information-theoretic* 意义上的 retrospective(token t 给最终判断带来的信息量),而不是 *interventional / counterfactual* 意义上的(把 token t 改了看会怎样)。

### 3.3 为什么我之前 framing 错了

我把 prospective 和 retrospective 错误地等同于"信息可见范围"。正确的区分应该是:

| 维度 | Prospective | Retrospective(信息论) | Retrospective(因果) |
|------|-------------|------------------------|----------------------|
| **定义** | E[outcome \| s_:t] 在 t 时刻的估计 | token t 给最终 belief 带来的信息增量 | token t 在因果意义上对 outcome 的贡献 |
| **数学** | V*_t | ΔV*_t = V*_t − V*_{t-1} | E[outcome \| do(s_t = a)] − E[outcome \| do(s_t ≠ a)] |
| **是否需要看完整条 seq** | 否 | 是(隐式,通过 V*_T = ΣΔV* 体现) | 是(显式) |
| **是否需要 intervention** | 否 | 否 | 是 |

**用户的提议想要的可能是第三列(因果),但我们之前讨论的所有 contrastive likelihood 方案落在第二列(信息论)**。两者只在因果图满足某些条件(没有混淆变量、无模型偏差)时才相等。

---

### 3.4 用同样的 Bayes 技巧分解 SDPO loss

§3.1 的 Bayes 恒等式不只能用来分析 CPV/verdict 路线——**完全相同的推导也适用于 SDPO 当前的 teacher_qv 信号**。这个分解会把 SDPO 的 Q_t 拆成 "value 部分 + likelihood 部分" 两个 explicit 组分,从 framework 层面回答"SDPO 信号到底是什么"。

#### 3.4.1 SDPO 信号的 setup

SDPO/OPSD 框架下(本 codebase 的 self-distillation 设定;非传统 OPD),teacher 接收 `prompt + reference r`(reference 可以是 gold answer、参考解法、或其它 auxiliary info)作为 context,然后对 student rollout `s` 的每个 token 给概率:

$$
Q_t := \log p_T(s_t \mid r, s_{<t})
$$

这个 `Q_t` 就是 [`compute_teacher_qv_advantage`](../verl/trainer/ppo/core_algos.py#L480) 里的 `Q_t`,也是所有 `baseline_type` 的 raw signal(详见 [`experiments/teacher_qv_ablation/LOSS.md §1`](../experiments/teacher_qv_ablation/LOSS.md#L11))。

**关键观察**:`Q_t = log p_T(s_t | r, s_<t)` 在结构上**与 §3.1 的 `log p_T(s_t | c, s_<t)` 完全同构**——只是把 verdict `c ∈ {right, wrong}` 换成了 reference 序列 `r`。所以同一个 Bayes 推导可以照搬。

#### 3.4.1.5 关键架构约束:**没有两个网络,只有一个 π_θ**(OPSD,不是 OPD)

**OPSD 框架里不存在 `p_T` 和 `p_s` 这两个独立模型**——只存在一个 policy `π_θ`,被用在两种 conditioning 下:

- 记号 `p_s(a_t | s_<t)` ≜ `π_θ(a_t | s_<t, question, system_prompt)`
- 记号 `p_T(a_t | r, s_<t)` ≜ `π_θ(a_t | s_<t, question, system_prompt, r)`

`p_T` 和 `p_s` **是同一组参数 `θ` 在两种不同 context 下的 forward**,不是两个独立训练的模型。前文一直在写 `p_T` / `p_s` 只是为了和 KD 文献对齐叙述,真实情况是 **single policy with vs without privileged info `r` in context**。

**这条架构约束给出一条精确恒等式**:`p_T` 把 `r` 从 context 里去掉,结果**就是** `p_s`:

$$
p_T(s_t \mid s_{<t}) \;=\; \pi_\theta(s_t \mid s_{<t}, \text{question}, \text{system\_prompt}) \;=\; p_s(s_t \mid s_{<t})
$$

这不是任何 calibration / KL 近似假设——它是定义层面的恒等,**两条 LHS/RHS 调用的是同一份模型 weight 的同一个 forward**。

**这条恒等式重写了 SDPO 的几何**:`Q_t − log p_s(s_t|s_<t)` 测量的不是"teacher density 减去 student density",而是**"同一个 model,在 context 里加上 r 之后,对 s_t 的 log-likelihood 变化了多少"** —— 这是一个 well-defined 的 **causal contrast**(干预变量 = `r` 是否在 context 里),不再是"两个模型的 density log-ratio"。下面 §3.4.2 / §3.4.4 全部基于这个 reframing。

#### 3.4.2 推导:Q_t 的 Bayes 分解

直接套用 §3.1.2 Step 3 的 Bayes 恒等式(`c → r`):

$$
\log p_T(s_t \mid r, s_{<t}) = \log p_T(r \mid s_{\le t}) - \log p_T(r \mid s_{<t}) + \log p_T(s_t \mid s_{<t})
$$

定义 teacher 在前缀 `s_:t` 处对 reference 的 log-likelihood:

$$
W_t := \log p_T(r \mid s_{:t})
$$

以及它的单步增量:

$$
\Delta W_t := W_t - W_{t-1} = \log p_T(r \mid s_{\le t}) - \log p_T(r \mid s_{<t})
$$

那么 SDPO 信号就分解为:

$$
\boxed{\;Q_t = \underbrace{\Delta W_t}_{\text{prospective belief shift (about reference)}} + \underbrace{\log p_T(s_t \mid s_{<t})}_{\text{raw likelihood prior}}\;}
$$

(`r` 是序列时仍然成立:`log p_T(r | s_:t) = Σ_i log p_T(r_i | s_:t, r_<i)`,可以单次 forward 读出。)

**用 §3.4.1.5 的 shared-param 恒等式替换最后一项**:

$$
\boxed{\;Q_t = \Delta W_t + \log p_s(s_t \mid s_{<t})\;}
$$

—— SDPO 信号在结构上 = "纯 value 增量 ΔW_t" + "student 自己在这个 token 上的 log-likelihood"。这条等式是后面 §3.4.4 推出 "`baseline_type='student'` 唯一能拿到 pure value" 的关键。

#### 3.4.3 两个组分的语义对照

| 组分 | 形式 | 性质 | 对应 §3.1 的概念 |
|------|------|------|------------------|
| `ΔW_t` | `log p_T(r \| s_≤t) − log p_T(r \| s_<t)` | **token s_t 给"reference 是正确续写"这个 belief 带来的 log-prob 增量** | 与 `ΔV*_t` 同构(只是把单 token verdict 换成多 token reference) |
| `log p_T(s_t \| s_<t)` | unconditioned teacher likelihood | **s_t 在 teacher marginal 下有多 likely**,与 reference 无关 | 与 §3.1.3 contrast 中**消掉**的那一项是同一个 |

**关键观察**:**`ΔW_t` 才是 SDPO 信号里真正起 credit assignment 作用的部分;`log p_T(s_t | s_<t)` 是搭便车的 likelihood prior**——它不携带 "s_t 是不是好 token" 的信息,只携带 "s_t 是不是常见 token" 的信息。

#### 3.4.4 哪些 baseline 能消掉 likelihood prior(shared-param 关键)

CPV 通过 right/wrong 做差让 `log p_T(s_t | s_<t)` 在两侧恒等而消掉(§3.1.3)。SDPO 里能不能消,**取决于 baseline 的选择**——而 §3.4.1.5 的 shared-param 恒等式让答案变得 explicit:

```
A_t = Q_t - V_t
    = ΔW_t + log p_s(s_t | s_<t) - V_t        ← 注意已经把 log p_T(s_t|s_<t) 替换成 log p_s(s_t|s_<t)
```

逐个 baseline 看:

| baseline | V_t | A_t 化简后 | 是否 pure value |
|----------|-----|------------|----------------|
| **`student`** | `log p_s(s_t \| s_<t)` | **`ΔW_t`** | ✅ **是,完全消掉 likelihood prior** |
| `ce` | `-log p_s(s_t \| s_<t)` | `ΔW_t + 2 log p_s(s_t \| s_<t)` | ❌ 不仅没消,反而把 student likelihood 项放大 2 倍 |
| `group_mean` | scalar(batch / token 维度的均值) | `ΔW_t + log p_s(s_t \| s_<t) - const` | ❌ 中心化但 token-level 结构不变 |
| `group_hier` | scalar(hierarchical 版) | 同 group_mean | ❌ 同 |

**这条重新校准了核心叙事**:

- 之前(误以为 teacher / student 是独立网络时):"所有 baseline 都消不掉 likelihood,SDPO 困在 likelihood pollution 里"
- 现在(shared-param 框架下):**`baseline_type='student'` 是 Bayes-aligned 的唯一选择,它直接给纯 ΔW_t**;`ce` / `group_mean` / `group_hier` 才是真的有 likelihood pollution

换句话说,**likelihood pollution 是 baseline 选错了**,不是 SDPO 范式本身的限制。在 OPSD 框架下,纯 value 信号本来就一直在 `baseline_type='student'` 路径里 sitting there。(若搬到传统 OPD 上,这个 cancellation 不再自动成立,需要假设两个网络在 marginal 上一致。)

**为什么 contrast 在这里能工作**:CPV 用 right/wrong 在 verdict 维度做差消掉公共项;student baseline 用 `Q_t − log p_s` 在 reference 维度做差消掉公共项——**两者本质都是 contrast,只是消的轴不同**。OPSD 的 shared-param 设定让 student baseline 自动具备了这个 contrast 结构(因为 `p_T(·|s_<t) = p_s(·|s_<t)`),不需要额外 forward。OPD 下没有这个免费午餐。

#### 3.4.5 这解释了 SDPO 经验 vs 理论的张力(并重新校准"likelihood ≠ value"的批评)

[`teacher_qv_analysis.md`](teacher_qv_analysis.md) 里对 SDPO 的核心批评:

> "log p_T(a_t) 是 likelihood,不是 value function。所有 baseline_type 都在 density 概念之间换花样。"

**Bayes 分解 + shared-param 约束告诉我们这个批评需要 refine,而不是全盘接受**:

- ✅ 对**部分** baseline 成立:`ce` / `group_mean` / `group_hier` 给出的 `A_t` 确实是 "ΔW_t + 残留 student likelihood",value 被稀释——这条批评精确命中
- ❌ 对 `baseline='student'` **不成立**:在 OPSD shared-param 框架下,`A_t = ΔW_t` 是**纯 value 增量**,没有任何 likelihood 项残留(§3.4.4)

所以更精确的批评应该是:**"qv_ce-fl(Phase 1-4 最优配置)的 likelihood pollution 问题是 `baseline=ce` 引入的,不是 SDPO 范式本身的问题"**。

这条 refine 解释了 Phase 1-4 的经验现象:
- **reward 在涨** ← `ΔW_t` 这个 hidden value signal 即使被稀释也在起作用
- **entropy 塌得很狠**(qv_ce-fl entropy 0.001 vs GRPO 0.21) ← `ce` baseline 引入的 **`+2 log p_s(s_t|s_<t)`** 项把 student 推向自己的 mode(self-reinforcement)——这才是 entropy 塌缩的直接 driver

**两个组分目标方向不一致的真正版本**:在 `qv_ce-fl` 下,`ΔW_t` 推向"匹配 reference",`2·log p_s(s_t|s_<t)` 推向"放大 student 已有的高 density"——后者是 self-reinforcing,任何 mode-seeking 倾向都会被放大成塌缩。`qv_student-fl` 应该没有这个塌缩驱动力(因为 `+log p_s − log p_s = 0`),如果它仍然塌,那 driver 在别处。

**可立刻验证的实验**:
- 比较 `qv_student-fl`(理论上 pure ΔW_t)和 `qv_ce-fl`(理论上 ΔW_t + 2·log p_s)在 entropy 曲线上的差异
- 如果 `qv_student-fl` entropy 塌得显著比 `qv_ce-fl` 慢 → Bayes 分解解释拿到经验支持
- 如果两者塌得一样狠 → 说明 entropy 塌缩的真实 driver 不在 likelihood pollution,需要找别的解释(比如 ΔW_t 本身 scale 极大、过于 sparse 等)

#### 3.4.6 Phase 4 α-mixing 的一个新解释

`qv_ce-fl + α=0.5` 是 Phase 4 现有最优。从 Bayes 分解角度看:

- `α · L_PG`:依赖 `Q_t = ΔW_t + log p_T(s_t | s_<t)`,**value 与 likelihood 两个目标耦合**
- `(1-α) · L_fKL = (1-α) · KL(p_T ‖ p_s)`:**纯 likelihood matching**(student 向 teacher density 对齐)

α=0.5 之所以比 α=1.0(纯 SDPO PG)reward 更高,可能的机制:**显式加入 likelihood-matching 项后,两类压力得到解耦**——KL 项专门承担 likelihood 对齐,PG 项里"被污染的 likelihood 成分"和"value 成分 `ΔW_t`"的内在张力得到缓解,value 成分相对突出。

这只是 hypothesis,但它给 Phase 4 的 α-mixing 提供了**理论解释**(之前只是经验调参)。可以验证的实验:
- **相关性测量**:在固定 student/teacher 上算 token level 的 `corr(ΔW_t, log p_T(s_t | s_<t))`,若长期负相关,验证"两类目标方向冲突"
- **value-only ablation**:把 SDPO 改造成 `Q'_t = Q_t − log p_T(s_t | s_<t) ≡ ΔW_t`(即手工 subtract 掉 likelihood 项),对比 reward / entropy 曲线
- **α 扫描的理论预测**:若假设成立,在加入 fKL 后 α 越小 entropy 越稳——已有的 α=0.5 vs α=0.7 数据点可以做后验验证

#### 3.4.7 在 shared-param 下,verdict_mc 的真正卖点是什么

§3.4.4 之前我以为 verdict_mc 的核心贡献是"在 framework 层面对 likelihood pollution 做隔离手术"——**但 shared-param 约束让 `baseline='student'` 已经做到了这件事**。所以需要重新定位 verdict_mc 的 value-add。

**ΔW_t vs (V\*_T − V\*_{t-1}):测量的是不同的 Bayes 因子**

| 信号 | 测量 | 来源 | 适用场景 |
|------|------|------|---------|
| `ΔW_t = log p(r \| s_≤t) − log p(r \| s_<t)` | s_t 让"**specific gold reference r** 是 valid continuation"的 belief 上升多少 | teacher with privileged context | gold reference 明确且唯一 |
| `V*_t − V*_{t-1}` | s_t 让 model 对"**final verdict 是 right**"的 belief 上升多少 | 同一个 model 在 vocab 里 `[right]/[wrong]` logit | model 是 calibrated judge |

两者**重合**的情况:gold reference 唯一、model 是良 calibrated judge → 两个 Bayes 因子等价。

两者**分歧**的情况:
- **multi-solution 任务**(数学题有多条 valid 解法):`ΔW_t` 只奖励 match 到 specific gold path 的 token,惩罚 alternative valid path;`V*_t − V*_{t-1}` 奖励任何让 model confident-on-correct 的 token,**对 alternative valid path 更宽容**
- **gold reference 不在 deployment 时可用**:训练时有 gold(可以用 `ΔW_t`),inference 时没有 — `V*_t` 路线和 inference 时的 self-evaluation 一致,gap 更小

**重定位后的 verdict_mc 价值**:

| 方案 | 信号 | 主要卖点 | 主要风险 |
|------|------|---------|---------|
| `baseline='student'`(应优先跑) | `ΔW_t` | shared-param 下自然消 likelihood;零额外 compute;直接复用现有 pipeline | gold reference 太严苛时会惩罚 valid alternatives |
| `verdict_mc` | `V*_T − V*_{t-1}` | multi-solution robustness;不依赖 gold 在 context 中 | model 作为 judge 需要 calibration 验证(§8.1) |

verdict_mc **不再是 SDPO 的必要替代**,而是**当 `baseline='student'` 在 multi-solution 任务上偏差大时的升级路径**。先验证 `baseline='student'` 的 Bayes-clean 版本是否已经够用;不够时再上 verdict_mc。

---

## 4. 真正符合"看完整条序列 + retrospective"的方案

既然 `Δ_t = V*_t − V*_{t-1}` 仍然是"逐步前向"的(每个 token 只看自己的 prefix),那要满足用户"必须看完整条 seq 才能算"这个 commitment,有以下几个真正不同的方案:

### 4.1 方案 A:Monte Carlo Return(全局 V\*_T)

```
A_t = V*_T − V*_{t-1}
```

含义:每个 token t 拿到的 credit 是"从 t 之前(belief = V*_{t-1})走到最终(belief = V*_T)"这一整段 belief shift。这等于 `Σ_{k≥t} ΔV*_k` —— **token t 拿到的 credit 包括了 t 之后所有 token 累积的影响**。

**与 ΔV*_t 的本质差异**:
- `ΔV*_t` 只 credit 给 t **本身**贡献的那部分信息
- `V*_T − V*_{t-1}` 把 "t 之后所有 token 共同导致的最终 belief" 全部 credit 给 t

**对应 RL 文献**:standard MC return,在 sparse reward setting 下偏差大但方差小。

**满足用户 commitment**:✅ V*_T 显式依赖 s_full,必须等到 rollout 结束才能算。

### 4.2 方案 B:GAE-style λ-return(Δ 与 MC 之间的 interpolation)

```
A_t = (1 − λ) Σ_{k≥t} λ^(k−t) (V*_k − V*_{t-1})
     ↘ 极端 λ→0: 退化为 ΔV*_t
     ↘ 极端 λ→1: 退化为 V*_T − V*_{t-1}
```

**对应 RL 文献**:Generalized Advantage Estimation ([Schulman et al. 2015](https://arxiv.org/abs/1506.02438))。

**满足用户 commitment**:✅(λ > 0 时,信号显式包含 V*_{t+k} for k > 0,需要看完整条 seq)。

### 4.3 方案 C:Counterfactual Replacement(真因果)

```
A_t = V*_T(s_full) − V*_T(s_<t · s'_t · s_>t)
```

其中 s'_t 是某个 baseline token(比如 teacher 在该位置的 argmax,或 `<pad>`)。**这才是用户朴素直觉里的"如果 token t 不是这个,结局会如何"的因果反事实**。

**问题**:严格做需要 O(T) 次额外 forward(每个位置一次替换)。可以用 amortized 近似:
- 替换为整段 token 的 batch(一次 forward 处理 T 个反事实)
- 或用 gradient-based first-order Taylor 近似:`A_t ≈ ‖∇_{x_t} V*_T‖ · ‖x_t − x'_t‖`

**满足用户 commitment**:✅ 完美对应"看完整条 seq + token-level intervention"。

### 4.4 方案 D:Gradient Saliency(其实就是用户原话的字面实现)

用户原话"append `[wrong/right]` 在末尾,看 verdict 与每个 token 的相关性"——最 literal 的实现就是:

```
A_t = ‖∇_{e_t} log p_T(verdict | s_full)‖   # e_t 是 token t 的 embedding
```

或更稳定的版本:

```
A_t = (∇_{e_t} log p_T(verdict)) · e_t      # gradient × input
```

**对应文献**:[Pruthi et al. 2020](https://arxiv.org/abs/2005.00928)、Integrated Gradients ([Sundararajan et al. 2017](https://arxiv.org/abs/1703.01365))。

**满足用户 commitment**:✅ 确实需要完整 seq,且数学上 gradient 对应一阶 Taylor 展开下的 counterfactual,与方案 C 弱等价。

**风险**:saliency 在 transformer 上有已知的 noise 问题(参见 [Bastings & Filippova 2020](https://arxiv.org/abs/2010.05607)),且需要额外 backward,会让训练 throughput 降一半左右。

---

## 5. 四个方案的横向对比

| 方案 | 是否需要看完整 seq | 计算成本 | 因果 vs 关联 | 与 ΔV\*_t 的关系 | 实现复杂度 |
|------|-------------------|----------|--------------|------------------|------------|
| ΔV*_t (≡ contrastive prepend) | ❌ 等价于 prefix-running | 1× teacher forward | 关联(信息论) | 自身 | ⭐ 低 |
| **A: MC return V\*_T − V\*_{t-1}** | ✅ | 1× teacher forward | 关联(全局) | Σ_{k≥t} ΔV*_k | ⭐ 低 |
| **B: GAE λ-return** | ✅ | 1× teacher forward | 关联(衰减) | (1−λ)Σ λ^k ΔV*_{t+k} | ⭐⭐ 中 |
| C: Counterfactual replacement | ✅ | O(T)× teacher forward 或近似 | 因果(真) | 不可约 | ⭐⭐⭐⭐ 高 |
| D: Gradient saliency | ✅ | 1 forward + 1 backward | 因果(一阶近似) | 不可约 | ⭐⭐⭐ 中 |

---

## 6. 推荐落地路径

### 6.0 第〇步(新增,最高优先级):跑 `baseline_type='student'` 验证 Bayes-clean 版本

根据 §3.4.4–§3.4.7 的 shared-param 分析,**`baseline='student'` 在 OPSD 框架下数学上等价于 pure ΔW_t**(零 likelihood pollution),且**这个 baseline 已经在 codebase 里实现了**——直接跑,不需要任何新代码。

**实验目的**:验证"Phase 1-4 entropy 塌缩主要由 `ce` baseline 引入的 `2·log p_s` 项驱动"这个假设。

**对比**:
- `qv_ce-fl`(Phase 4 现有最优)→ 含 likelihood pollution
- `qv_student-fl` → Bayes-clean,pure ΔW_t

**预期结果**:
- 如果 `qv_student-fl` 的 entropy 塌缩显著比 `qv_ce-fl` 慢、reward 持平或更高 → 验证 Bayes 分析,**可以省掉 verdict_mc 实现**(或把 verdict_mc 推到 §6.3)
- 如果 `qv_student-fl` entropy 仍然狠塌,reward 不涨 → 说明问题不在 likelihood pollution,在 ΔW_t 本身(可能 scale 太大、太 sparse 等),进入 §6.1

**为什么这一步必须先跑**:zero implementation cost、直接证伪 / 证实 §3.4 整个 Bayes 解释框架。在写新的 verdict_mc 之前先用 codebase 已有 capability 做这个 sanity check 是 cheapest experiment with highest information value。

### 6.1 第一步:实现方案 A (MC return / verdict_mc)

**何时进入这一步**:§6.0 验证显示 `baseline='student'` 仍然 entropy 塌 / reward 不涨,或者目标任务是 multi-solution(verdict 路线在 §3.4.7 的对比里有优势)。

**理由**:
1. 与现有 teacher_qv pipeline 耦合最低——只需要在 teacher forward 时多读 `[right]/[wrong]` 两个 token 的 logit,V*_t 就有了。
2. 不引入额外 forward/backward,训练成本与现有 qv_ce-fl 持平。
3. 是真正满足用户"看完整 seq"commitment 的最简方案,且和现有 PG/full_logit 框架兼容。
4. 在文献坐标下对应 [Snell et al. 2024 V-STaR](https://arxiv.org/abs/2402.06457)、[Lightman et al. 2023 PRM](https://arxiv.org/abs/2305.20050)的 token-level 落地,理论上有支持。

### 6.2 第二步(若 A 不够):接 GAE λ-return(方案 B)

`λ` 作为新超参,扫 `{0.0, 0.5, 0.9, 0.95, 1.0}`:
- λ=0.0 ≡ 现有 qv_ce-fl 同类(单步 belief shift)
- λ=1.0 ≡ 方案 A (MC return)
- 中间值是 bias-variance tradeoff

预期 λ ≈ 0.95 是最佳点(典型 PPO 取值),如果 λ=1.0 反而最好,说明 teacher 是 calibrated judge,可以 commit 到方案 A。

### 6.3 第三步(研究性):方案 C/D 做 ablation 比对

在小 batch 上跑,看 counterfactual 信号(C/D)和 conditional 信号(A/B)的相关性。如果相关性 > 0.8,说明方案 A 已经足够;< 0.5 说明 conditional ≠ causal,值得做 C。

---

## 7. 实现 Sketch (verl)

新增一个 `baseline_type='verdict_mc'`(MC return 版本),核心改动:

```python
# core_algos.py: in compute_teacher_qv_advantage
elif baseline_type == "verdict_mc":
    # 假设 verdict_token_ids 是 (right_id, wrong_id),从 tokenizer 拿
    # teacher_logits: (B, T, V)  — 现有 teacher forward 已经算出
    
    right_id, wrong_id = verdict_token_ids
    
    # V*_t = log_softmax(teacher_logits)[:, t, right] - log_softmax(teacher_logits)[:, t, wrong]
    # 注意:teacher 在每个位置自然已经对 vocab 里所有 token 给概率,
    # 不需要 append/prepend 任何东西,直接读 logits 即可。
    log_probs = teacher_logits.log_softmax(dim=-1)
    V_star = log_probs[..., right_id] - log_probs[..., wrong_id]   # (B, T)
    
    # MC return: A_t = V*_T - V*_{t-1}
    # V*_T 是 response 最后一个 valid token 处的值
    last_valid_idx = response_mask.long().sum(dim=-1) - 1            # (B,)
    V_star_T = V_star.gather(1, last_valid_idx.unsqueeze(-1))        # (B, 1)
    
    # V*_{t-1} 用 shift 实现,V*_{-1} 设为 0(prior)
    V_star_prev = torch.cat([
        torch.zeros_like(V_star[:, :1]),
        V_star[:, :-1]
    ], dim=1)                                                        # (B, T)
    
    A = V_star_T - V_star_prev                                       # (B, T)
    return A.detach()  # advantage 永远 detached
```

**关键工程要点**:
1. **不需要真的 append `[right]/[wrong]`**——teacher 在 vocab 里本来就有这两个 token 的 logit,直接读就行。这是上文 §3 Bayes 等价性的工程兑现。
2. **复用现有 teacher forward**——零额外 forward。
3. **PG mode 选 sampled 而非 full_logit**——A_t 现在是 trajectory-grounded 的真实 advantage,不再是 vocab-level distributional signal,full_logit 的 `Σ_v p_s(v) · A(v) · log p_s(v)` 公式不适用。
4. **配合现有 reward**——可以加 sign-aware modulation:`A_t ← (2R−1) · A_t` 让 verifier 给方向、teacher 给权重(详见 §8 风险 1)。

---

## 8. 风险与开放问题

### 8.1 Teacher 必须是 calibrated judge

V*_t 假设 teacher 真的能区分对错。需要 sanity check:

```
1. 抽 1000 条已标注的 (rollout, R) 对
2. 计算每条的 V*_T = log p_T(right | rollout) − log p_T(wrong | rollout)
3. 验证 AUC(V*_T, R) > 0.8
```

如果 teacher 自己分不出对错(例如 Qwen3-8B 在某些专业领域),这个方案的所有 V*_t 都是噪声。**这是先验最大风险**。

### 8.2 `right` / `wrong` token 的 vocab 选择

Qwen3 tokenizer 里:
- `right`/`wrong` 可能 tokenize 成多个 sub-token
- 候选对:`(yes, no)`, `(correct, incorrect)`, `(right, wrong)`, `(✓, ✗)`,需要 ablation 选最 calibrated 的

理想情况是用一对 single-token 且 baseline marginal probability 接近 0.5 的 token。

### 8.3 V*_T 与 verifier R 的不一致

teacher 的判断和 verifier 的判断**很可能不一致**——这正好是这个方案的优势(teacher 是 soft critic 比 binary R 信息量大),也是风险(可能给错答案)。

两条缓解:
- **校准项**:加一个 auxiliary loss `λ · ‖σ(V*_T) − R‖²`,把 teacher-judge online 校准到 verifier
- **Hybrid**:用 R 决定整条轨迹的方向,用 V*_t increment 决定 token 内部分布:`A_t = (2R−1) · ΔV*_t`(这个公式 §6 sketch 里也提了)

### 8.4 Distribution shift

teacher 是在 supervised text 上训练的,student rollout 可能 OOD(尤其是当 student 已经塌缩到低熵时)。在 OOD 序列上 teacher 的 V*_t 估计可能不可靠。**早期训练**可能没事(student 还接近 teacher),**后期**风险上升。

可考虑:用 teacher 的 perplexity 做 reweight,或当 perplexity 超过阈值时退化到现有 qv_ce-fl baseline。

### 8.5 这是 novel contribution 吗?

Token-level credit assignment via teacher-as-judge 这个具体框架,目前我能找到的最近邻是:
- **Lightman et al. 2023 (PRM)**:训练独立的 process reward model,而不是复用 teacher
- **V-STaR (Snell et al. 2024)**:类似方向但 step-level 不是 token-level
- **Zhang et al. 2024 Generative Verifiers**:把 verifier 写成 LM,read-out 是 token sequence,但还是 sequence-level

**用 teacher LM 自己做 token-level retrospective verdict、且 zero extra forward 复用现有 KD pipeline**——这个具体 setup 我没有找到 prior art。值得查一下 EMNLP/ICLR 2025 是否有人做了。如果没有,这是一个有 publication potential 的角度。

---

## 9. 与现有 teacher_qv 路线的关系(shared-param 修正后)

§3.4.4 的 shared-param 分析推翻了之前对 `qv_student-fl` 的判断——它**不是** density log-ratio,**是 pure value 增量 ΔW_t**(因为分母 `p_s` = 分子 marginalize 掉 r 的 teacher)。修正后的对比:

| 路线 | A_t 公式 | shared-param 化简后 | 语义 | 缺陷 |
|------|----------|---------------------|------|------|
| **qv_student-fl** | `log p_T(a_t \| r) − log p_s(a_t)` | **`ΔW_t`** | **pure value(reference 的 Bayes 因子)** | 依赖 gold reference 唯一性;ΔW_t 可能 sparse |
| qv_ce-fl (Phase 4 最优) | `log p_T(a_t \| r) + H(p_s, p_T)` | `ΔW_t + 2·log p_s(s_t\|s_<t)` | value + 放大 student likelihood | self-reinforcement,entropy 塌缩(§3.4.5) |
| qv_group_mean / group_hier | `log p_T(a_t \| r) − scalar` | `ΔW_t + log p_s(s_t\|s_<t) − const` | value + 残留 likelihood | mode-seeking 倾向(§3.4.4) |
| Sign-aware (上一轮提议) | `(R-R̄)/σ · [log p_T(a_t \| r) + H(p_s, p_T)]` | `(R-R̄)/σ · [ΔW_t + 2·log p_s]` | scaled qv_ce-fl | 同 qv_ce-fl 的塌缩 driver |
| **verdict_mc(本方案 §7)** | `V*_T − V*_{t-1}` | 同(不依赖 shared-param 化简) | **calibrated judge 的 verdict 增量** | teacher 校准依赖(§8.1) |

**修正后的判断**:

- **qv_student-fl** 应该是 Bayes 意义上**最干净**的现有 baseline——但 Phase 1-4 它表现一般,可能因为(a) gold reference 太严苛 / (b) ΔW_t scale 大且 sparse / (c) 没和 fKL 等 α-mix 配合。先按 §6.0 重跑它。
- **qv_ce-fl** 表现"最好"是因为 `2·log p_s` 项提供了 self-distillation 风格的 strong learning signal,但它的 entropy 塌缩 cost 也来自同一项。
- **verdict_mc** 的差异化定位在 multi-solution robustness(§3.4.7),不再是"唯一能消 likelihood pollution 的方案"。

---

## 10. 总结

1. 用户提出"append verdict + 看 correlation"的 idea,直觉精神是 retrospective credit assignment via teacher-as-judge——这条 framing 是对的,且比现有 likelihood-based 方案更接近 RL 意义上的 value-based credit。
2. 我第一次操作化提出 contrastive prepended verdict,**Bayes 推导显示其严格等价于 prospective ΔV*_t**——这是关键 double-check,推翻了我自己之前对用户的回答。
3. **同一个 Bayes 推导分解 SDPO 信号**:`Q_t = ΔW_t + log p_T(s_t | s_<t)` = "reference 的 Bayes 因子" + "unconditioned likelihood"——给 SDPO 信号一个 framework-level 的拆解(§3.4)。
4. **关键架构修正**:本 codebase 是 **OPSD**(on-policy self-distillation),不是传统 OPD。OPSD 框架下不存在两个独立的 teacher / student 网络,只有一个 `π_θ` 在两种 conditioning 下运行(§3.4.1.5)。这条恒等式让 `p_T(·|s_<t) = p_s(·|s_<t)` 精确成立(忽略 EMA 延迟),从而 `baseline_type='student'` 在数学上**直接等于 ΔW_t**(pure value,零 likelihood pollution),`Q_t - log p_s` 测量的是"context 里加 r 的 causal contrast",不是"两个网络的 density ratio"。这条 cancellation **不适用于传统 OPD**(独立 teacher 网络),那种情形下需要额外 calibration 假设。
5. **重定位 verdict_mc 价值**:不再是"唯一能消 likelihood pollution"的方案(那是 `baseline='student'` 的事),而是"multi-solution robustness 升级路径"(§3.4.7)。
6. **修正后的实施顺序**:
   - 第〇步(highest ROI):跑 `qv_student-fl` 验证 Bayes-clean 版本的 entropy/reward(§6.0)——zero implementation cost
   - 第一步:若 §6.0 不够,实现 `verdict_mc`(MC return,§7),复用现有 forward,零额外 compute
   - 后续:GAE λ-return(§6.2)、counterfactual ablation(§6.3)
7. 关键 prerequisite(对 verdict_mc 才需要):验证 model 是 calibrated judge(§8.1)。`baseline='student'` 不需要这个 prereq,但需要 gold reference 唯一性较强。

---

*本文写于 2026-05-22,分支 `geodesic-sdpo-v2 @ d4f002a`,用户与 assistant 讨论后由 assistant 整理。Bayes 推导部分的 double-check 是讨论中发现的非平凡结论,推翻了 assistant 上一轮的回答。*
