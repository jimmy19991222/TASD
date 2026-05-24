# CDM 损失的熵崩溃与 JSD 双教师重写

**日期**：2026-05-24
**关联代码**：[verl/trainer/ppo/core_algos.py](../verl/trainer/ppo/core_algos.py) `_compute_verdict_distillation_loss` 中的 `loss_method == "cdm"` 分支
**关联实验**：`awesome_jimmy/SDPO_CDM`（runs `cdm`、`cdm_topk_off`，均为 Qwen3-8B + sciknoweval/biology）
**关联记忆**：[[feedback-reverse-kl-entropy-term]]、[[research-verdict-credit-assignment]]、[[project-opsd-shared-params]]、[[feedback-contrastive-push-fluency-risk]]

## 0. 直觉先行:作弊玩家、教练与火星文

数学推导在 §1–§5。先用「师生」比喻把整件事的物理图像理一遍 —— 信息论(JSD/MI)、RL 工程坑(reward hacking / 模式坍缩)、LLM 语言流形(fluency)三层逻辑交叠在一起,直觉容易打结,所以读完 §0 再回头看 §1 起的公式,会发现公式里每一项都在比喻里有对应物。

把 student 想成一个**想拿高分的作弊玩家**,两个教师是**带着标准答案打分的教练**。

### 0.1 V1 为什么塌:卡 Bug 而不是答题

V1 的规则:「对每个 token,看好教练打分 `log p_T^+` 和坏教练打分 `log p_T^-`,差值 `log_lr` 越大就越说那个词」(`per_token_loss = −Σ p_s · log_lr`)。

Student 不去学推理,**它去暴力搜词表找 Bug**:某个生僻 token 上,好教练给 0.1 分(怪但勉强),坏教练给 1e-4 分(根本不会说),`log_lr` 直接顶到 +7。Student 把所有概率倾倒到这个 token,**loss 一阶刷掉,熵清零**。这是 §1.2 表里 step 13–17 熵从 0.30 → 0.01、`log_lr_abs_mean` 0.4 → 1.5 的物理原因。

数学根因(§1.3):V1 缺了反向 KL 的 `-H(p_s)` 项 —— 那一项相当于「不许只押一个 token」的硬约束,被偷工减料省掉了,所以 student 可以塌成 delta。

### 0.2 JSD 是怎么补救的:换成一根橡皮筋

V2 把规则换成「**整个分布**要贴近好教练、远离坏教练」 —— $L = \mathrm{JSD}_\alpha(\pi_T^+,\pi_s) - \lambda\,\mathrm{JSD}_\alpha(\pi_T^-,\pi_s)$。

JSD 是分布间的距离 ∈ [0, ln 2],它像一根**橡皮筋**:student 想再去钻 V1 那个 Bug —— 把 99% 概率堆一个乱码词 —— 它跟好教练的 JSD 立刻顶到 ln 2 上限,**第一项 `JSD_pos` 直接吃满**,损失反而变大。橡皮筋强迫 student 保持分布的形状(=语言多样性 + 合理的 token 覆盖)。

数学上对应的就是 JSD 展开里**自带的 `H(p_s)` / `H(m_α)` 项**(§1.3 V1 缺的那一半,§3.2 用 MI 重写为 $I_\alpha(V;C)$),不需要额外加熵正则。

### 0.3 为什么要用两个教练相减:堵抄 GT 的捷径

只用好教练有个致命问题:好教练手里有 GT,它指导时不可避免**「念答案」**(把 GT 字符串本身的 token 打高分)。Student 学到的最优捷径不是推理,**是直接抄 GT**。

CDM 的精妙在于坏教练**也拿着同一份 GT**(只是 verdict 反过来)—— 它念答案的程度跟好教练一模一样。

| Student 行为 | 靠近 +gt 教练 | 靠近 −gt 教练 |
|---|---|---|
| 抄 GT | ✅ 得分 | ✅ 也得分 → 被 −λ 抵消 |
| 真推理 | ✅ 得分 | ❌ −gt 推理是错的,不靠近 |

差分把抄 GT 这块 marker-confound 项 $\Delta_\text{copy}$ 干掉了(§3.4 给出严谨证明:λ=1 时第二项归零),**只剩「正确性」这条 student 能学的轴**。

### 0.4 λ 为什么不能太大:咆哮过度逼出火星文

把 λ 设成 2.0 等于对 student 喊:「**无论如何不要像坏教练那样说话!**」

Student 听了一下:坏教练虽然推理错,但说的是**人话**(语法、词序都正常)。最快的「不像坏教练」方案是 —— **改说火星文**。乱码 token 在 $\pi_T^-$ 上几乎为零,student 把质量挪过去,$\mathrm{JSD}(\pi_T^-,\pi_s)$ 直接到 ln 2,推力项被刷满。

代价是 $\pi_T^+$ 也在人话流形上,所以 student 跟好教练的 JSD **同时**爆掉。但 λ 大、推力主导,净 loss 还在降 —— 这就是 §4.1 的 fluency destruction:loss 数值在跌,行为在崩。

§4.3 雷达表的「**两个 JSD 同时↗ + 熵反弹**」就是抓这个走火入魔。λ=0.5 的工程理由:让拉力比推力强 2:1,student 不敢在「让两个 JSD 同时变大」的方向上下注 —— 牺牲掉一点 λ=1 的理论 GT-copy 完美抵消,换取「先说人话、再在人话流形里找正确推理」的稳定性。

### 0.5 比喻 ↔ 公式对照表

| 比喻 | 数学对应物 | 文档位置 |
|---|---|---|
| 教练打分差 = `log_lr` | log Bayes factor of verdict | §3.1 |
| 「卡 Bug 找局部峰」 | REINFORCE 梯度只指向 argmax(log_lr) | §1.1 |
| 「橡皮筋」 | JSD ∈ [0, ln 2] + 自带 `H(p_s)` | §1.3 / §3.2 |
| 「抄 GT 在两个教练上对称」 | $\Delta_\text{copy}$ 在 (1−λ) 系数下消去 | §3.4 |
| 「火星文」 | $\pi_s$ 偏离自然语言流形,push 项一阶刷损失 | §4.1 |
| 「先说人话再推理」 | λ ≤ 0.5,拉力主导 | §4.2 |

读完 §0 后续的公式就只是把这些直觉**钉死成可计算的量** + **可监控的 metric**。

## 1. 问题：CDM 第一版触发熵崩溃

CDM（Counterfactual Discriminative Markers）的目标是用同一条采样轨迹喂给两套带 GT 的教师上下文：

- m^+gt = "This answer is verified correct, reference answer is &lt;gt&gt;."
- m^-gt = "This answer is verified incorrect, reference answer is &lt;gt&gt;."

让 GT-copy / format-mirror 这类「教师只是在抄 GT」的捷径在两边对称出现、相减时结构性抵消，剩下的差异 `log p_T(v|x,m^+gt) − log p_T(v|x,m^-gt)` 才是「正确性」的信号。

### 1.1 第一版损失

```python
log_lr = (teacher_pos − teacher_neg).detach()                  # (B, T, V_or_K)
per_token_loss = −(student_full.exp() * log_lr).sum(−1)        # = −Σ p_s · log_lr
```

我把它叫做「reverse-KL 方向的损失」，但**它不是反向 KL，而是把 `log_lr` 当作每词 reward 的 REINFORCE**：

$$
L_{\text{REINF}}(t) = -\sum_v p_s(v) \cdot \text{log\_lr}(v),\qquad
\nabla_z L = -p_s(v)\bigl(\text{log\_lr}(v) - \mathbb{E}_{p_s}[\text{log\_lr}]\bigr)
$$

REINFORCE 梯度只指挥学生「冲向 log\_lr 最大的那个 token」，并没有任何东西把概率往回拉。

### 1.2 训练曲线证实崩溃

`cdm_topk_off`（full vocab）实验：

| step | actor/entropy | log_lr_signed_mean | log_lr_abs_mean | response_length/mean |
| ---- | ------------- | ------------------ | ----------------| -------------------- |
| 1    | 0.50          | +0.08              | 0.49            | 308                  |
| 10   | 0.55          | −0.16              | 0.41            | 265                  |
| 18   | 0.01          | −0.50              | 0.78            | 663                  |
| 28   | 0.04          | −0.72              | 1.53            | 65                   |

熵在 step 13–17 的 4 个 step 内从 0.30 跌到 0.01；同时 `log_lr_abs_mean` 从 0.4 一路涨到 1.5（log_lr 分布越来越尖），signed mean 越来越负——student 把质量倾倒到 log_lr 的局部峰，而那些峰在数值上恰好偏向 `log p_T^-` 更大的 token（后面解释）。

### 1.3 真正的反向 KL 缺了什么

$$
\mathrm{KL}(p_s \| \pi^*) = \sum_v p_s \log p_s - \sum_v p_s \log \pi^*
                         = -H(p_s) - \sum_v p_s \log \pi^* + \log Z
$$

正确的反向 KL 含两半，前一半 `−H(p_s)` 在被最小化时等价于**最大化 H(p_s)**——自带熵正则。我只写了第二半，等于把熵正则砍掉了。

> **教训**（已落 [[feedback-reverse-kl-entropy-term]]）：写「学生跟随某个每词分数 s(v)」时，绝对不要图省事写 `-(p_s * s).sum()`。先 `log_pi_star = s − logsumexp(s)`，再 `F.kl_div(log_pi_star, log_p_s, log_target=True).sum(-1)`，跟 `opd_bayes` 分支保持同一种写法。

## 2. 重写：双教师 JSD 组合

修复方向有两条：

1. 直接把第一版补成正确的反向 KL `KL(p_s ‖ softmax(log_lr))`。
2. **改用两套独立 JSD 的组合**——本次采用的方案。

第二条之所以更稳：

- **可重用 SDPO 现成的 α-混合 JSD 机器**（`distillation_topk` / `add_tail` / `alpha` 都直接复用），保证跟同一组 α=0.5 baseline 的对比公平。
- **JSD 上界 ln 2**，差分 `JSD_pos − λ·JSD_neg ∈ [−ln 2, ln 2]`，损失绝不会跑到 −∞。第一版 reverse-KL 投影把 log_lr 当作目标，本质上是无界的。
- **物理含义直观**：第一项把 student 拉向「+gt 教师」，第二项把 student 推离「−gt 教师」。两项都通过一个固定的 distance metric 衡量，可以分别监控。
- **GT-copy 捷径仍然取消**：抄 GT 的概率质量在 `π_T^+` 和 `π_T^-` 上几乎相等，对 JSD\_pos 和 JSD\_neg 的贡献量级一致，差分时几乎抵消。

### 2.1 损失定义

$$
\boxed{\;L_t = \mathrm{JSD}_\alpha\!\bigl(\pi_T^{+gt}, \pi_s\bigr)
              \;-\;\lambda\cdot\mathrm{JSD}_\alpha\!\bigl(\pi_T^{-gt}, \pi_s\bigr)\;}
$$

其中 α 复用 `self_distillation.alpha`（默认 0.5，对应对称 JSD），λ 由新增的
`self_distillation.cdm_neg_weight`（默认 1.0）控制。

### 2.2 实现要点

落在 `_compute_verdict_distillation_loss` 的 `loss_method == "cdm"` 分支（[core_algos.py:1815](../verl/trainer/ppo/core_algos.py)）：

```python
def _prep_log_probs(lp):
    # topk + add_tail / topk renorm，跟标准 SDPO 同款
    ...

s_lp    = _prep_log_probs(student_full)
tpos_lp = _prep_log_probs(teacher_pos.detach())   # 防御性 detach: 上游已 no_grad,这里把"梯度只从 s_lp 流"的契约写在本地
tneg_lp = _prep_log_probs(teacher_neg.detach())

def _alpha_jsd(student_lp, teacher_lp):
    # α=0 / α=1 / 0<α<1 三分支跟主 SDPO 分支字符级一致
    ...

jsd_pos = _alpha_jsd(s_lp, tpos_lp)
jsd_neg = _alpha_jsd(s_lp, tneg_lp)

per_token_loss = jsd_pos − cdm_neg_weight * jsd_neg
```

Topk 路径的索引已经在 builder 阶段对齐到 student 的 top-k——`teacher_pos`、`teacher_neg` 都是在同一组 vocab id 上重新评估，所以 JSD 是「同维度同语义」的。

### 2.3 监控指标

- `self_distillation/cdm_jsd_pos_mean`：JSD\_pos 平均，应在 (0, ln 2) 内；正常下降表示 student 在向 +gt 教师靠拢。
- `self_distillation/cdm_jsd_neg_mean`：JSD\_neg 平均，应在训练中保持/增大；过快下降意味着 student 也在跟 −gt 教师靠拢（=被 GT-copy 捷径拖进去）。
- `self_distillation/cdm_jsd_diff_mean`：直接对应 loss 的核心项 `JSD_pos − JSD_neg`，理想是单调下降。
- `self_distillation/cdm_log_lr_(signed/abs)_mean`：保留作为「教师正负差异有多大」的诊断（不再进 loss，只是观察）。
- `actor/entropy`：现在有了两份 JSD 中天然的 `H(p_s)` 项做正则，应该不再 freefall。

## 3. Bayes 视角下的推导

§2 的损失看起来只是「把 SDPO 的 JSD 从单教师扩成双教师差分」。但在 OPSD 共享参数的前提下，它有一个干净的 Bayes 含义 —— 学生在做的事是**判别式后验对齐**，不是简单的「跟两个老师按权重求和」。

### 3.1 OPSD 倒推：教师 = 学生 + verdict 观测的后验

OPSD 的本质约束：teacher 不是另一个网络，而是同一份 $\pi_\theta$ 读到了一个 verdict marker $V$。所以

$$\pi_T^V(v) := p_\theta(v\mid x, V)$$

对 student 自己做 Bayes 倒推：

$$
\boxed{\;\log\pi_T^V(v) \;-\; \log\pi_s(v) \;=\; \log p_\theta(V\mid x, v) \;-\; \underbrace{\log p_\theta(V\mid x)}_{\text{const in }v}\;}
$$

$\Delta W^V(v) := \log[\pi_T^V/\pi_s](v)$ **= 把 token $v$ 当作答案首字时，模型自己赋予 verdict 观测 $V$ 的对数似然**（差一个 $v$ 无关的常数）。这是 [[project-opsd-shared-params]] 中「ΔW 即纯 value，不掺 likelihood」的根源。

代入 $V \in \{V^{+gt}, V^{-gt}\}$：

- $\Delta W^+(v) = \log p_\theta(V^+ \mid x, v) + c_+$
- $\Delta W^-(v) = \log p_\theta(V^- \mid x, v) + c_-$
- $\log_\text{LR}(v) := \Delta W^+(v) - \Delta W^-(v) = \log\dfrac{p(V^+\mid x,v)}{p(V^-\mid x,v)}$ —— **每词的对数 Bayes factor**（两个 $v$ 无关常数相互抵消）。

### 3.2 JSD 的 Bayes 身份：互信息

广义 α-JSD：

$$\mathrm{JSD}_\alpha(p, q) \;=\; (1-\alpha)\,\mathrm{KL}(p\|m_\alpha) \;+\; \alpha\,\mathrm{KL}(q\|m_\alpha),\qquad m_\alpha = (1-\alpha)p + \alpha q$$

引入隐变量 $C\sim\text{Bernoulli}(\alpha)$ 选择 $p$ 还是 $q$，再采 $V \mid C$。直接展开 $I(V;C) = H(V) - H(V\mid C)$ 得到

$$
\boxed{\;\mathrm{JSD}_\alpha(p, q) \;=\; I(V;\,C)\;}
$$

α-JSD 就是「看到一个 token 后能多准确反推它来自 $p$ 还是 $q$」的互信息（先验权重 = α）。代入 CDM 的两个 JSD：

- $\mathrm{JSD}_\alpha(\pi_T^{+gt},\pi_s) = I_\alpha(V;\,C^+)$ —— $V$ 对「是否插入了 $+gt$ marker」的信息量
- $\mathrm{JSD}_\alpha(\pi_T^{-gt},\pi_s) = I_\alpha(V;\,C^-)$ —— $V$ 对「是否插入了 $-gt$ marker」的信息量

### 3.3 CDM 损失的语义

$$\boxed{\;L_t \;=\; I_\alpha(V;\,C^+) \;-\; \lambda\, I_\alpha(V;\,C^-)\;}$$

最小化等价于同时做两件事：

- **降低** $I(V;C^+)$ → 让 token 对「是否做了 +gt 干预」**不可辨**（充分统计量被吃掉）→ $\pi_s$ 学到「看起来就像被告诉答对了」。
- **抬高** $I(V;C^-)$ → 让 token 对「是否做了 −gt 干预」**易辨**（充分统计量被放大）→ $\pi_s$ 学到「看起来不像被告诉答错了」。

合起来：**student 把自己的采样分布摆到 verdict 后验空间里 $+gt$ 那一侧**。这是判别式（discriminative）目标，不是单纯模仿 $\pi_T^{+gt}$。

### 3.4 GT-copy 捷径为何结构性消去

把 student 的失败模式分解：

$$\pi_s = (1-\eta)\,\pi_s^{\text{honest}} + \eta\,\pi_s^{\text{copy-gt}}$$

由于 $\pi_T^{+gt}$ 和 $\pi_T^{-gt}$ 都看到了 GT 字符串，两边都会把质量加在「抄 GT」的 token 上，得到同一个 lift $\Delta_\text{copy}$：

$$
\begin{aligned}
\mathrm{JSD}(\pi_T^{+gt}, \pi_s) &= J^+_\text{honest} + \Delta_\text{copy} \\
\mathrm{JSD}(\pi_T^{-gt}, \pi_s) &= J^-_\text{honest} + \Delta_\text{copy}
\end{aligned}
$$

差分：

$$L \;=\; (J^+_\text{honest} - \lambda\, J^-_\text{honest}) \;+\; (1-\lambda)\,\Delta_\text{copy}$$

**$\lambda = 1$ 时第二项归零** —— 「抄 GT」这部分 marker-confound 被完全切掉，只剩「$+$verdict vs $-$verdict 的真信号差」。这正是 CDM 设计 dual-marker 的核心目的。

### 3.5 与原始 $-\sum p_s\log_\text{LR}$ 的关系

v1 损失（已被否决）展开是

$$L_{v1} \;=\; -\,\mathbb{E}_{V\sim\pi_s}\!\left[\log\dfrac{p(V^+\mid x,V)}{p(V^-\mid x,V)}\right]$$

读起来很 Bayes：**期望的 posterior log-odds**。但它只用了一阶矩（期望对数 BF），把分布形状信息全扔了，等价于「把所有质量打到 BF 最大的那个 token」—— 没有熵正则，必塌。曲线印证：见 §1.2。

JSD 形式把目标从「最大化期望 log-BF」升级到「**整体分布要贴近 +gt 教师、远离 −gt 教师**」。在 χ² 二阶近似下：

$$\mathrm{JSD}(p,q) \;\approx\; \tfrac{1}{8}\,\chi^2(p,q) \;\approx\; \tfrac{1}{2}\,d_\text{Fisher}^2(p,q)$$

所以

$$L \;\approx\; \tfrac{1}{2}\bigl[d_\text{Fisher}^2(\pi_T^+,\pi_s) \;-\; \lambda\, d_\text{Fisher}^2(\pi_T^-,\pi_s)\bigr]$$

—— **在 Fisher-Rao 流形上做几何 LDA**：把 $\pi_s$ 摆到「离 +gt 教师近、离 −gt 教师远」的判别面 $+gt$ 一侧。

### 3.6 λ 的 Bayes 解读 + 有界性

$\lambda$ 是 MI 减号项的权重：

| λ | 含义 |
| --- | --- |
| 0 | 退化为 SDPO 单教师 JSD（仅 +gt 拉力，等价于 `gt_marker` 模式） |
| 1 | 对称：拉力 = 推力 |
| <1 | 保守：以「贴近 +gt」为主 |
| >1 | 激进：以「远离 −gt」为主 |

由于 $\mathrm{JSD}_\alpha \in [0, \ln 2]$（α=0.5 时上界严格为 $\ln 2$，其他 α 取值上界更小），所以

$$L \;\in\; [-\lambda \ln 2,\; \ln 2]$$

**严格有界**。这是相对 v1 的关键改进：v1 的 $\sum p_s \log_\text{LR}$ 因 log-BF 无界 → 损失无下界 → 学生狂奔向局部 BF 极值；JSD 差分自带闭合区间，policy 走不远。

## 4. 外审风险与 λ 取值指南

外部 review 提了一条非常关键的反直觉点:**「推离一个分布」比「拉向一个分布」危险得多**(已落 [[feedback-contrastive-push-fluency-risk]])。CDM v2 数学上自洽,但 λ 的工程含义需要单独警惕。

### 4.1 Fluency Destruction 风险

$\pi_T^{-gt}$ 虽然在「答得对不对」这件事上是错的,但它说的依然是**人话**——语法、连词、标点都是自然语言。当 $-\lambda\cdot\mathrm{JSD}_\alpha(\pi_T^{-gt},\pi_s)$ 把 student 推离 $\pi_T^{-gt}$ 时,如果 λ 过大,student 会发现一条**捷径**:

> 逃离 $\pi_T^{-gt}$ 最快的方式不是「说更对的话」,而是**说胡话**——发散到长尾 / 乱码 token,因为这些 token 在 $\pi_T^{-gt}$ 上的概率最低。

数学直觉:$\pi_T^{-gt}$ 把质量摊在一个语言流形上;长尾乱码区在该流形外、$\pi_T^{-gt}$ 几乎为零;student 只要把质量挪过去,$\mathrm{JSD}(\pi_T^{-gt},\pi_s)$ 直接顶到 ln 2 上限,损失被一阶刷掉。但 $\pi_T^{+gt}$ 也在语言流形上,所以 $\mathrm{JSD}_\text{pos}$ 同时被推爆。**净 loss 仍可能下降**,因为 $\lambda$ 大、推力主导。

### 4.2 λ 推荐区间

| λ | 行为 | 状态 |
| --- | --- | --- |
| 0.1–0.3 | 拉力主导,推离作为辅助约束 | 文献常用安全区(对比解码 / unlearning) |
| **0.5** | 拉/推 ≈ 2:1,首选实验值 | **推荐起点** |
| 1.0 | 数学上对称,GT-copy 完美抵消 | 压力测试值,易触发 fluency destruction |
| >1.0 | 推力主导 | 高风险,需严密监控 |

CDM 设计的「λ=1 时 $\Delta_\text{copy}$ 完美归零」是个数学美感,但工程上 student 会先撞上 fluency 悬崖。先用 λ=0.5(推力打一个 2:1 折扣)换掉一小部分理论 GT-copy 抵消,换取更稳的语言流形约束。

### 4.3 监控组合(早期报警雷达)

不需要新加指标——以下三组现有 metric 的**联合趋势**足以诊断 fluency destruction:

| 信号组合 | 解读 |
| --- | --- |
| `actor/entropy` ↗ + `cdm_jsd_neg_mean` ↗ → ln 2 + `cdm_jsd_pos_mean` 也 ↗ | **fluency destruction**:student 跑去长尾乱码区,两个 JSD 同时被顶大,熵反弹 |
| `actor/entropy` ↘ → 0 + `cdm_jsd_pos_mean` ↘ + `cdm_jsd_neg_mean` ↘ | **GT-copy 塌陷**(v1 的失败模式):student 收敛到一个尖峰,两侧都靠近 |
| `actor/entropy` 平稳 + `cdm_jsd_pos_mean` ↘ + `cdm_jsd_neg_mean` 持平/↗ | **CDM 信号生效**:student 向 +gt 教师靠拢、与 −gt 拉开距离,这是目标轨迹 |
| `actor/ppo_kl` 突然飙高 | 任意一种崩溃的前兆,降 λ 或降 lr |

## 5. 与 SDPO/OPSD 框架的关系

- 标准 SDPO(OPSD 共享参数)的损失:`JSD_α(π_T, π_s)`,单教师,α=0.5。
- CDM = 双教师 JSD 差分。当 `cdm_neg_weight=0` 时退化为「只用 +gt 标记的标准 SDPO」(这正好可以做消融基线)。
- 这套写法**是纯损失替换**(pure loss replacement),没有把 sample-level R 当 PG advantage 去乘 belief delta,因此**不会撞到 belief × PG 的符号陷阱**(参 [[feedback-belief-pg-sign-check]])。

## 6. 还需要回答的问题

1. λ=1 是否合适？较小的 λ（0.3–0.5）可能更稳，因为 `JSD_neg` 本身可能噪声较大；λ 太大会让损失被「推离 −gt」主导。需要对比 λ ∈ {0.5, 1.0, 2.0}。
2. α=0.5 vs α=1.0 / 0.0：这是 forward / reverse / 对称 JSD 之争，跟同一组 baseline 的 SDPO sweep 已经测过 α=0.5 比较稳，沿用就好。
3. **抄 GT 捷径的取消程度怎么验证**：监控 `JSD_pos − JSD_neg`；如果 GT-copy 完美对称取消，这个差应该是「正确性 vs 错误性」的纯净信号。可以做一个故意把 +gt 和 −gt 都设成相同模板的「null 实验」，期望 `JSD_diff` 围绕 0。

## 7. 下一步

- 把第一版 CDM 的两个 stale run 停掉(已经训了 30+ step 但 loss 形式错误)。
- 用新 loss 跑(顺序按风险递增):
  - `cdm-jsd / topk=100`、**λ=0.5**(主力,fluency 安全区,§4.2 推荐起点)
  - `cdm-jsd / topk=100`、λ=0.3(更保守,验证「推离信号是否还够用」)
  - `cdm-jsd / topk=100`、λ=1.0(压力测试,看 fluency destruction 是否真的发作)
- 监控按 §4.3 三组联合趋势判断:
  - 如果出现 fluency destruction 模式(熵反弹 + 两个 JSD 同时↗),直接砍 λ。
  - 如果 CDM 信号生效模式持续,跑到 250 step 看 sciknoweval acc。
- Null 实验(外审建议):用同一模板同时当 +gt 和 −gt(`m^0 = "Reference answer is <gt>."`),期望 `cdm_jsd_diff_mean` 钉死在 0 附近 —— sanity check CDM 信号是真来自 verdict 极性而不是模板差异。
