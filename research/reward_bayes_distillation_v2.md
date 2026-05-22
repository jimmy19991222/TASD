# Reward-Bayes Distillation: 把 Verified Reward 直接焊进 Loss 的两条路径

> **状态**: 方法升级草案, 2026-05-22, 月丘
> **目的**: 修复 VC-OPSD v1 草稿中 "R 只当 sign / 当锚" 的弱融入问题,给出两条把 verifier reward 通过 Bayes 直接构造进 loss 的方法。
> **关联**: 主稿 [vc_opsd_paper_draft.md](vc_opsd_paper_draft.md) / [vc_opsd_paper_draft_zh.md](vc_opsd_paper_draft_zh.md), 推导起点 [verdict_credit_assignment.md](verdict_credit_assignment.md)。
> **代号**: 本文给出的两个方法分别简称 **OPD-Bayes**(Outcome-Posterior Distillation)与 **VEC**(Verified-Evidence Credit)。最终入选哪个作主线需结合 §6 实证。

---

## 0. 问题诊断:VC-OPSD v1 里 R 的融入太弱

回顾 v1 草稿 §4.4:
$$
\mathcal{L}_{\text{VC-OPSD}}^{(v1)} \;=\; -\,\mathbb{E}_\tau\!\left[ (2R - 1) \cdot \sum_t \mathrm{sg}(\Delta V^*_t) \cdot \log p_s(s_t) \right] \;+\; \lambda \cdot \mathcal{L}_{\text{calib}}.
$$

R 出现两次:

1. $(2R-1) \in \{-1, +1\}$ —— 只决定 PG 的**符号**,与 magnitude 无关。
2. $\mathcal{L}_{\text{calib}}$ —— 整轨迹层面的锚,不影响 per-token magnitude。

**问题**: 当 verifier reward 为 1 vs 0 时,advantage 的*大小*完全一样,只是方向反转。从信息论看,R 这 1 bit 的内容**没有进入 per-token credit 的量级**。审稿人会立刻问: "如果 R 只是 sign,你和 GRPO 配对 ΔV* baseline 有什么本质区别?"

下面两条路径都把 R 作为**贝叶斯观测事件**直接进 loss,使其同时定义信号的 **方向 + 量级 + 形状**。

---

## 1. 路径 A — Outcome-Posterior Distillation(OPD-Bayes)

### 1.1 核心构造

定义 **outcome-posterior 教师分布**: 当外部 verifier 告知 outcome 将为 R 时,Bayes-最优的 student 在每一步应该取什么分布?

$$
\boxed{\;
\pi^*(v \mid s_{<t}; R) \;:=\; \frac{\pi_\theta(v \mid s_{<t}) \cdot p(R \mid v, s_{<t})}{p(R \mid s_{<t})}.
\;}
$$

这是**严格的贝叶斯更新**: 先验 $\pi_\theta$、似然 $p(R | v, \cdot)$、证据 $p(R | \cdot)$。$\pi^*$ 即"假如学生提前知道 outcome 是 R,它应该相信的下一 token 分布"。

### 1.2 损失函数

KL 投影把 $\pi_\theta$ 拉向这个**自身条件后验**:

$$
\boxed{\;
\mathcal{L}_{\text{OPD-Bayes}} \;=\; \mathbb{E}_{s_{<t}}\!\Big[\, \mathrm{KL}\!\big(\pi^*(\cdot \mid s_{<t}; R) \,\big\|\, \pi_\theta(\cdot \mid s_{<t})\big) \,\Big] \;+\; \lambda \cdot \mathcal{L}_{\text{calib}}.
\;}
$$

展开:

$$
\mathrm{KL}(\pi^* \| \pi_\theta) \;=\; \sum_v \pi^*(v) \log \frac{\pi^*(v)}{\pi_\theta(v)} \;=\; \sum_v \frac{\pi_\theta(v) p(R|v,\cdot)}{p(R|\cdot)} \log \frac{p(R|v,\cdot)}{p(R|\cdot)}.
$$

即:**对每个候选 token $v$,按"它能多大程度上把 verifier 推向观察到的 R"做加权,然后让 $\pi_\theta$ 模仿这个加权分布**。

### 1.3 怎么估 $p(R | v, s_{<t})$

这是全方法唯一的"未知量"。利用对比 forward + Bayes 反推:

$$
\frac{p(R{=}1 \mid v, s_{<t})}{p(R{=}0 \mid v, s_{<t})} \;=\; \underbrace{\frac{p_T^+(v \mid s_{<t})}{p_T^-(v \mid s_{<t})}}_{\text{对比比率}} \cdot \underbrace{\frac{\pi(\text{right} \mid s_{<t})}{\pi(\text{wrong} \mid s_{<t})}}_{\text{先验赔率}}.
$$

**推导**: 由 Bayes,
$p_T^\pm(v|s_{<t}) = p(v | s_{<t}, Y{=}\text{r/w}) = \frac{p(v|s_{<t}) \cdot p(Y{=}\text{r/w} | v, s_{<t})}{p(Y{=}\text{r/w}|s_{<t})}$,
两式相除即得。

**先验赔率**怎么取:
- **静态**: 从 batch base-rate 估,$\hat\pi(\text{right}) = $ rollout 的平均 R。
- **动态**: 由 calibration loss 学习的一个 learnable scalar logit。
- **保守**: 直接取 0.5/0.5,放弃 prior 信息,容忍轻微 bias 换稳定。

加 sigmoid 把 odds 化为概率:
$$
\hat{p}(R{=}1 \mid v, s_{<t}) \;=\; \sigma\!\Big(\big[\log p_T^+(v) - \log p_T^-(v)\big] + \log \hat\pi(\text{r}) - \log \hat\pi(\text{w})\Big).
$$

### 1.4 计算路径

每 step:
1. Student forward: $\pi_\theta(\cdot|s_{<t})$ (vocab 分布)。
2. 两次 teacher forward: $p_T^\pm(\cdot|s_{<t})$ (vocab 分布,与 student 共参数,只是 prefix 不同)。
3. 算 $\hat{p}(R|v,\cdot)$, 构造 $\pi^*(\cdot|R)$。
4. KL loss,反传穿过 $\log \pi_\theta(v)$ 项;$\pi^*$ 端 stop-gradient。

**算力**: 与 v1 VC-OPSD 同(2 teacher + 1 student forward)。KV-cache 复用同样适用。

### 1.5 为什么比 v1 强

| 维度 | v1 VC-OPSD | OPD-Bayes |
|---|---|---|
| R 进 loss 的方式 | sign + 锚 | 直接定义 target 分布 |
| Per-token magnitude 是否反映 R | 否(±1 翻转) | 是(Bayes factor) |
| 与 SDPO 的关系 | 是 SDPO + 替换 r | SDPO ⊂ OPD-Bayes(取 $r$ 为 $c^\pm$、忽略 calibration) |
| 与 GRPO 的关系 | (2R−1) 是 sign | GRPO 是 OPD-Bayes 在 $p(R|v,\cdot)$ 近似为常数时的退化 |
| Noisy verifier 行为 | sign 翻成噪声 | $\hat{p}(R|v) \to 0.5$, 信号自动衰减 |

### 1.6 与 §3 的衔接

v1 §3 的"SDPO loss family = $\mathbb{E}_q[\Delta W]$"是教师不动、改采样分布的视角。**OPD-Bayes 把这一族再统一一层**: $\pi^* = \pi_\theta \cdot p(R|v) / p(R)$ 本身是一个新的 teacher,所有 vocab-KL 形式(rKL / fKL / JSD)都可以重新对它写一次。

具体:
- rKL: $\mathrm{KL}(\pi_\theta \| \pi^*) = -\mathbb{E}_{v \sim \pi_\theta}[\log p(R|v) - \log p(R)]$ —— *student-sampled R-Bayes factor*。
- fKL: $\mathrm{KL}(\pi^* \| \pi_\theta)$ —— 即上述 OPD-Bayes loss。
- JSD: 在 $\pi^*$ 与 $\pi_\theta$ 之间插值,工程意义为 bounded variant。

也就是说 v1 §3 的 4 行表可以**升级为一张 8 行表**:每个 SDPO loss 变体都得到一个 verifier-conditioned 对偶版本。

---

## 2. 路径 B — Verified-Evidence per-token Credit(VEC)

### 2.1 核心构造

不构造 teacher,直接把"verifier 已经观察到 R"翻译成**每 token 的真实价值增量**:

$$
\boxed{\;
\widehat{\Delta V}^R_t \;:=\; \log \hat{p}(R \mid s_{\le t}) - \log \hat{p}(R \mid s_{<t}).
\;}
$$

其中 $\hat{p}(R | s_{\le t})$ 用 sequence-level 对比构造:
$$
\hat{p}(R{=}1 \mid s_{\le t}) \;=\; \sigma\!\Big(\log p_T^+(s_{1:t}) - \log p_T^-(s_{1:t}) + \log \hat\pi(\text{r/w})\Big).
$$

这里 $\log p_T^\pm(s_{1:t}) = \sum_{i \le t} \log p_T^\pm(s_i | s_{<i})$ 是前缀联合 log-likelihood。

### 2.2 关键守恒律(Telescoping)

按定义,
$$
\sum_{t=1}^T \widehat{\Delta V}^R_t \;=\; \log \hat{p}(R \mid \tau) - \log \hat{p}(R \mid \emptyset).
$$

**每 token credit 总和 = 整轨迹 log-evidence**。这是 v1 没有的性质:GRPO 给整条轨迹 1 个 R,VEC 把这 1 bit **保守地分摊**到每个 token,sum 严格守恒到 log-evidence。

### 2.3 损失函数

PG 形式:
$$
\boxed{\;
\mathcal{L}_{\text{VEC}} \;=\; -\,\mathbb{E}_\tau\!\left[ \sum_t \mathrm{sg}\!\big(\widehat{\Delta V}^R_t\big) \cdot \log \pi_\theta(s_t \mid s_{<t}) \right] \;+\; \lambda \cdot \mathcal{L}_{\text{calib}}.
\;}
$$

注意与 v1 的关键差异:**$\widehat{\Delta V}^R_t$ 把 R 编进 magnitude**,而不是单独乘 $(2R-1)$ 在前面。

- 若 R=1: $\widehat{\Delta V}^R_t$ 在"模型相信此 token 是 right 的"位置上为正,反之为负。
- 若 R=0: 符号自然颠倒,且 magnitude 反映"模型当时多自信"。
- 若 $\hat{p}(R|s_{\le t}) \approx 0.5$(模型分不出): $\widehat{\Delta V}^R_t \approx 0$,信号自然衰减 —— 没有人为 gate。

### 2.4 noisy verifier 的 graceful degradation

设 verifier 有噪 $\varepsilon$: $R = Y \oplus \text{Bern}(\varepsilon)$。
- GRPO: advantage 直接被翻转,梯度方向错误。
- v1 VC-OPSD: $(2R-1)$ 仍是 ±1,sign 错就全错。
- **VEC**: $\hat{p}(R|s_{\le t})$ 会反映 calibration 后的不一致 —— 模型很自信 $V^*$ 在右、但 R=0 时,$\widehat{\Delta V}^R_t$ 给出**小幅**的负 credit(而非 GRPO 的全幅反转)。

这是 VEC 在 noisy-verifier niche 里的核心优势。

### 2.5 与 PRM 的关系

VEC 的 $\widehat{\Delta V}^R_t$ 本质上是一个**隐式 PRM**:它就是"该 token 对最终 R 的边际证据贡献"。但它**不是单独训练**的网络,而是由 $\pi_\theta$ 通过对比 forward 自动 induce 出来。这是 OpenAI o1 / PRM800K 路线的"无标注、零额外参数"版本。

---

## 3. 两条路径的关系

| | OPD-Bayes | VEC |
|---|---|---|
| 单位 | KL 散度(vocab 维) | log-evidence 增量(scalar) |
| 信号来源 | $\pi^* = \pi_\theta \cdot p(R\|v) / p(R)$ target | $\widehat{\Delta V}^R_t$ 标量 advantage |
| 与 SDPO 的关系 | SDPO 的 vocab-KL 在 $r=c^\pm$ 时退化 | SDPO 的 token-PG 用 $\widehat{\Delta V}^R_t$ 替代 $\Delta W_t$ |
| 与 GRPO 的关系 | $p(R\|v) \to p(R)$ 时退化 | $\widehat{\Delta V}^R_t$ 跨步求和 = trajectory log-evidence,与 GRPO 的 R 同构 |
| 反传形态 | vocab-级 distill | per-token PG |
| 实现复杂度 | 同 v1(替换 target) | 同 v1(替换 advantage 公式) |
| 与 §3 论证的契合 | 把 SDPO loss family 升级为 R-conditioned 双倍版本 | 把 token-PG 路线推到 telescoping-守恒的极限 |

**等价性观察**: 当 $\pi_\theta$ 已经达到"完美 calibration"时,OPD-Bayes 的 KL 梯度恰好等于 VEC 的 PG 梯度按 expected-token 写出的形式。两者是同一对象在 vocab-level / token-level 的两个投影。这一等价性可在 Appendix 中作为统一性 corollary。

### 3.1 推荐定位

- **OPD-Bayes** 适合作主线方法,因为它直接套进 §3 的统一框架,reviewer 容易看出"原来 SDPO 的整个 loss family 都有 verifier-conditioned 对偶版"。
- **VEC** 作为 token-level 工程实现 + telescoping 守恒律(单独一节强调 attribution 性质),并提供 noisy verifier 的实验侧重。
- v1 的 **VC-OPSD = $(2R-1) \cdot \Delta V^*$** 降级为 ablation baseline,显示"只用 R 当 sign"和"用 R 进 Bayes"的差距。

---

## 4. 新的论文结构提案

```
§1 Intro   —— P1/P2/P3 + 引入 R-as-event 视角
§2 Background —— OPD/OPSD 区分, SDPO loss family
§3 Bayes Unification of SDPO        ← v1 §3 保留
   3.1 共享参数恒等式 + ΔW
   3.2 token-PG/vocab-KL 在 ΔW 下统一
   3.3 (新)R-conditioned 对偶视角: 把 r 一般化为 verdict 事件
§4 Reward-Bayes Distillation        ← 全新
   4.1 OPD-Bayes(主线)
   4.2 VEC(token-level 推论)
   4.3 估 p(R|v): 对比 forward + calibration loss
   4.4 算法 + 算力分析
   4.5 与 GRPO / SDPO / PRM 的关系图谱
§5 Diagnostics(ΔW SNR + verdict AUC)← v1 §5 保留
§6 Experiments
   主表: GRPO vs SDPO-best vs (v1 VC-OPSD ablation) vs OPD-Bayes vs VEC
   noisy-verifier 实验: 在 verifier flip rate ∈ {0, 5, 10, 20}% 下扫
   ablation: 先验赔率 / calibration 权重 / EMA / KL 方向
§7 Related Work
§8 Discussion
§9 Conclusion
```

---

## 5. 实现 TODO(替代旧的 VC-OPSD 实现路径)

按依赖顺序:

- [ ] **A. Calibration AUC 已在跑**(2026-05-22 sweep) —— 不变,继续等。AUC ≥ 0.8 即可上 OPD-Bayes / VEC。
- [ ] **B. `compute_opd_bayes_loss`**: 在 `verl/trainer/ppo/core_algos.py` 加新函数,签名:
  ```
  inputs: student_logits, teacher_pos_logits, teacher_neg_logits,
          R (scalar per rollout), prior_logit (learnable or static)
  outputs: loss_per_token (vocab-KL form), aux_metrics
  ```
  估计 ~150 LoC,复用现有 two-forward 路径。
- [ ] **C. `compute_vec_loss`**: 同上但 token-PG 形式。复用 B 的 `hat_p_R_given_v` 张量,做 prefix-cumsum 得 $\widehat{\Delta V}^R_t$。估计 ~80 LoC。
- [ ] **D. Calibration prior learnable scalar**: 在 actor config 加 `verdict_prior_logit_lr`,joint optimize。
- [ ] **E. Noisy-verifier 实验脚手架**: 在 verifier wrapper 加 `--flip_rate` 参数,offline mode 即可。
- [ ] **F. v1 VC-OPSD 作为 ablation baseline 保留**: 标 `--method=vc_opsd_sign`,与 OPD-Bayes / VEC 同框对比。

---

## 6. 风险与备选

- **风险 1**: $\hat{p}(R|v, s_{<t})$ 在低 calibration 下方差大 → loss 噪声。**缓解**: 加 clamping 或 EMA 平滑;若仍不稳,fallback 到 sequence-level $\hat{p}(R|s_{\le t})$ 然后做 token-PG(即 VEC 路线)。
- **风险 2**: KL 方向选错。OPD-Bayes 默认 $\mathrm{KL}(\pi^* \| \pi_\theta)$ (mass-covering),mode-seeking 反向(rKL)在 calibration 不准时可能更稳。在 §6 ablate 两个方向。
- **风险 3**: 先验赔率漂移。training 初期 base-rate 估计不准 → loss 偏。**缓解**: warm-up 阶段固定 prior=0.5,之后切到 batch-base-rate 估计。
- **风险 4**: noisy-verifier 实验"赢得太干净"reviewer 会怀疑数据生成。**缓解**: 用真实 noisy verifier(judge LLM 已知有 noise)+ 合成 flip 两个 setting 双线验证。

---

## 7. 一句话定位

> **R 不是 sign,R 是被观测到的 Bayes 事件;让它去定义 target 分布(OPD-Bayes)或定义 token-level evidence increment(VEC),才是把 verified reward 真正焊进 self-distillation 的方式。**
