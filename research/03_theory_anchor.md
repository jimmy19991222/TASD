# 03 — 理论锚点: 5 篇 Online DPO 论文 + 7 设计原则

> 完整深度分析见 [`OPD_Deep_Analysis.html`](OPD_Deep_Analysis.html) "Online DPO 理论锚点" 章节 (本 repo 内,2607 行)。本文是**精简本**便于快速参考。

---

## 5 篇核心论文

我们 DPO-TGS 的每个设计选择都对应这 5 篇论文中的一个 insight。

### Paper A: OAIF — *Direct LM Alignment from Online AI Feedback* (Guo et al. 2024)

**核心问题**: DPO 数据集是 **offline** 的,但 policy 是 **online** 演化的。

OAIF 给出 DPO 训练中**两层 distribution shift**的清晰分类:
- **初始 shift**: 预收集数据 $\mathcal{D} \sim \rho \neq \pi_{\theta_0}$
- **渐进 shift**: 训练中 $\pi_{\theta_t}$ 持续漂离 $\pi_{\theta_0}$

**方法**: 每 step 从当前 $\pi_{\theta_t}$ 采两 response,LLM annotator 打偏好,**on-policy** DPO update。

**实证**: human eval 显示 OAIF 对 offline DPO 胜率 **63.74%** (TL;DR);58% 对 SFT。但**显著长度偏置** (response 从 ~50 涨到 ~70 tokens)。

**对 DPO-TGS 的启发**: on-policy 是命脉。我们 chain rollout 完全 on-policy;**与 OAIF 不同的是**:
- 把 "annotator" 换成 verifiable env reward (无 LLM judge 偏置)
- 把 "采样" 换成 "teacher counterfactual" (有因果定位,不是独立 i.i.d.)

### Paper B: OFS-DPO — *Online Fast-Slow Chasing DPO* (Qi et al. 2024)

**核心问题**: 标准 DPO 有**梯度消失**。

**Proposition 3.3.1**: $\forall \epsilon > 0$, 训练后期 $\|\nabla_\theta \mathcal{L}_{\text{DPO}}(\theta)\| < \epsilon$ 几乎处处成立。

**直观原因**: DPO loss 含 $\sigma(\beta \cdot \text{margin})$,margin 大 → $\sigma'$ 饱和 → 梯度趋零。

**方法**: 两个 LoRA module (fast / slow) 模拟 *intraspecific competition*,新增 regularizer $\mathcal{L}_{\text{DPO-FS}}$ 测两模块偏好概率 gap,**互相追逐**。

**对 DPO-TGS 的启发**:
- v1: **α-mix GRPO fallback** 防梯度消失 (pair-free 样本走 GRPO,保证 entropy)
- v2 计划: 我们的 `self_distillation` 基础设施已维护一个 **EMA teacher** (rate=0.05,约 20 step lag),正好是天然的 "slow module" — 可加 chase regularizer (近零成本)

### Paper C: Samplers-in-DPO — *The Crucial Role of Samplers in Online DPO* (Shi et al. ICLR 2025)

**核心问题**: 不同 sampler 决定 DPO 的**收敛速度**。

**Theorems**:
| Sampler | 定义 | 收敛速度 |
|---|---|---|
| `DPO-Unif` (vanilla) | $\pi^{s_1} = \pi^{s_2} = \text{Uniform}$ | **linear** ($\propto 0.588^T$) |
| `DPO-Mix-R` (reward-guided) | $\pi^{s_1} \propto \exp(r)$, $\pi^{s_2} \propto \exp(-r)$ | **quadratic** ($\propto 0.5^{2^T-1}$) |
| `DPO-Mix-P` (policy-difference) | $\pi^{s_1} \propto (\pi_\theta/\pi_{\text{ref}})^\beta$ | **quadratic** ($\propto 0.611^{2^T-1}$) |

**实证**: Safe-RLHF +7.4%, Iterative-Prompt +5.4% vs vanilla DPO。

**对 DPO-TGS 的启发**: **OPSD teacher 是 reward-aware sampler 的工程实现**:
- 标准 DPO-Mix-R 需要 reward oracle 来构造 $\exp(\pm r)$ — 不实际
- 我们的 OPSD teacher (含 ref answer) 是 *privileged-info-aware* sampler,logits 隐式编码 "什么 token 更可能通往高 reward"
- → **DPO-TGS V2.5 ② Teacher-Anchored ref 直接接 ICLR-25 quadratic 收敛理论**

### Paper D: RPO — *Reflective Preference Optimization* (Zhao & Li, arXiv 2512.13240, Dec 2025)

> ⚠️ **DPO-TGS 最相近的先前工作** — 必须仔细差异化

**核心问题**: 证明 self-evolution DPO 的**信号缺陷**。

**Proposition** (RPO Eq 4): 当 chosen $y^+$ 与 rejected $y^-$ 都从同一 $\pi_\theta$ 采时:
$$\mathbb{E}[\Delta\ell(x)] \to 0,\quad \text{Var}[\Delta\ell(x)] \to 2\sigma^2,\quad \text{KL}(\pi_\theta(\cdot|x,y^+) \| \pi_\theta(\cdot|x,y^-)) \ll 1$$

→ **自蒸馏 DPO 的期望梯度信号为 0、方差爆炸**。

**方法**: hint-guided pipeline
1. $y^- \sim \pi_\theta(\cdot|x)$
2. $h = C_\phi(x, y^-)$ (外部 LMM 如 GPT-4V 生成 hint)
3. $y^+ \sim \pi_\theta(\cdot|x, h)$ (**同 policy**,但条件加入 hint)
4. 对 $(y^+, y^-)$ 做 DPO,权重 $W_{\text{hal}}$ 来自 hallucination severity

复合损失: $\mathcal{L}_{\text{RPO}} = \mathcal{L}_{\text{Pref}} + \lambda_1 \mathcal{L}_{\text{RD}} + \lambda_2 \mathcal{L}_{\text{Anc}}$

### ⚔ DPO-TGS 与 RPO 的关键差异 (论文必须强调)

| 维度 | RPO | **DPO-TGS** |
|---|---|---|
| Teacher 信号粒度 | 自然语言 hint (sequence-level) | **单 token decoding** (token-level) |
| 条件注入位置 | 拼到 input prompt | **拼到 response prefix** (causal splice) |
| (chosen, rejected) 关系 | 独立 regenerate,可能结构性发散 | **共享 prefix + 仅 1 token 分叉** → gradient causal-localized |
| Teacher 来源 | GPT-4V 等闭源 LMM (付费 API,不可微) | **同模型 + OPSD ref-answer context** (free, 可 EMA 自演化) |
| Pair 选择依据 | hint 是否触发 hallucination 修正 | **verifiable env reward $R(y_i) > R(y_{i-1})$ 才保留** |
| Sample 数 | 每 prompt 1 个 pair | **每 prompt chain_length 个 sample → 多个潜在 pair** |
| 目标领域 | LVLM hallucination | 通用 RL reasoning (math/biology/code) |
| 对 teacher 质量敏感性 | 高 (Table 4: QwenVL < GLM-4V < GPT-4V) | **低 (reward 兜底,弱 teacher 也能筛出有信号 pair)** |

**定位**: RPO 与 DPO-TGS 站在同一山头的两个朝向:
- RPO 朝**多模态 hallucination + 自然语言 hint**
- DPO-TGS 朝**通用 reasoning + token 级 causal intervention**
- 技术上 DPO-TGS 比 RPO 更细粒度 (token vs sequence),工程上更便宜 (同模型 OPSD vs 外部付费 API)
- 但 RPO 的 $\mathcal{L}_{\text{RD}}$ (Reflective-Distillation) 与 $\mathcal{L}_{\text{Anc}}$ (Anchored Regularization) 是 v2 可吸收的两个独立组件

### Paper E: Meta Bridging — *Bridging Offline and Online RL for LLMs* (Lanchantin et al., FAIR Meta, Jun 2025)

**核心实证**: 系统扫描 offline / semi-online (s=5/10/100) / online (s=1) × DPO/GRPO × verifiable/non-verifiable 全配置。

| Setting | Math500 | NuminaMath | AMC23 | AlpacaEval LC (GPT-4o) |
|---|---|---|---|---|
| Offline DPO ($s=\infty$) | 53.7 | 36.4 | 28.8 | 38.3 |
| Semi-online DPO ($s=100$) | **58.9** | 39.3 | **35.1** | 59.4 |
| Online DPO ($s=1$) | 58.7 | **39.6** | 32.9 | 60.1 |
| GRPO | 58.1 | 38.8 | 33.6 | 55.0 |

**3 个跨页关键发现**:
1. **Semi-online ≈ online ≈ GRPO**: 三者全部持平,且都显著 > offline。**"完全 on-policy" 不是必要条件,周期性 sync 足够好**。
2. **Reference model sync 才是关键**: 不 sync ref → response 长度崩溃 + reward 退化 (Figure 2)。
3. **Entropy collapse 普遍发生**: 所有 online DPO 变体训练中 entropy 都崩 (Figure 3)。

**Best-vs-worst pool pair** (与 DPO-TGS 高度相关):
- Meta 的 verifiable 任务 DPO pair: rollout N → 分 correct/incorrect pool → 随机各挑一个做 pair → **全 correct 或全 incorrect 的 prompt 直接 skip**
- ↔ 与我们 `all_failed_strategy='skip'` 完全一致
- 还试了 GroupDPO (所有 correct × incorrect 配对): **无显著提升 vs 单 pair** → 我们 chain_consecutive (单 pair per chain edge) 不需要扩成 group 形式

**对 DPO-TGS 启发** (3 点):
1. **论文 framing 调**: "完全 on-policy" 不是核心卖点。**真正差异化在 token-level teacher intervention**
2. 新增 `pair_strategy = 'hybrid_init_chain'` 验证 best-vs-worst init pair + chain pair 的叠加 (论文 ablation)
3. 加 entropy collapse 监控 metric (`actor/entropy_chosen` vs `actor/entropy_rejected`);Adam epsilon 调大也是 Meta 推荐的 tuning

---

## 5 篇 → DPO-TGS 7 设计原则

| # | 原则 | 来源 | DPO-TGS 实现位置 |
|---|---|---|---|
| ① | on-policy 采样不可妥协 | OAIF | Phase 1-3 chain rollout 全程从 $\pi_{\theta_t}$ 采 |
| ② | 用 verifiable reward 替代 LLM judge | OAIF | reward_fn 来自 sciknoweval verifier (无 judge 偏置) |
| ③ | 防止梯度消失 | OFS-DPO | v1 α-mix GRPO fallback;v2 EMA teacher 作 slow module |
| ④ | Teacher-guided ≈ quadratic-convergence sampler | Samplers ICLR-25 | OPSD teacher 是 reward-aware sampler 工程实现 → V2.5 ② |
| ⑤ | Chain depth ≈ logit mixing 的迭代版本 | Samplers ICLR-25 | n_attempts ablation (chain_length k = k-1 次 implicit logit mixing) |
| ⑥ | Token-level intervention 比 sequence-level hint 更细 | RPO | single-token splice + 共享 prefix causal-localization → V2.5 ① |
| ⑦ | best-vs-worst init pair + skip degenerate prompt | Meta Bridging | `pair_strategy='hybrid_init_chain'` + `all_failed_strategy='skip'` |

---

## 5 方法的统一视角 (vs DPO-TGS)

| 方法 | Pair 来源 | Sampler | 梯度稳定化 | Reward 信号 | Compute |
|---|---|---|---|---|---|
| Offline DPO | 预收集 (offline) | $\rho$ (任意) | 无 | 人工/RM | 低 |
| OAIF (Online DPO) | $\pi_{\theta_t}$ × 2 独立采样 | $\pi_{\theta_t}$ × 2 (Uniform 类) | 无 | LLM judge | +rollout |
| OFS-DPO | 预收集 或 $\pi_{\theta_t}$ | 同 base DPO | Fast/Slow LoRA chase | 同 base DPO | +LoRA |
| Samplers-DPO | $\pi_\theta^{3/2} \pi_{\text{ref}}^{-1/2} \times \pi_\theta^{1/2} \pi_{\text{ref}}^{1/2}$ | **DPO-Mix-P (quadratic)** | 无 | RM | +logit mixing |
| RPO (LVLM) | $y^- \sim \pi_\theta$ + $y^+ \sim \pi_\theta(\cdot\|x, h_{\text{GPT-4V}})$ | sequence-level hint 注入 | $\mathcal{L}_{\text{RD}} + \mathcal{L}_{\text{Anc}}$ | 外部 critique LMM | +API 调用 |
| **DPO-TGS (本工作)** | **chain 内 $R_i > R_{i-1}$ 自然 pair (共享 prefix)** | **OPSD teacher token splice (reward-aware, implicit Mix-R)** | **α-mix GRPO + (v2) EMA chase + RPO-style RD/Anc** | **verifiable env reward** | **+chain rollout (同模型)** |

---

## 论文定位句

> DPO-TGS 是 5 篇核心思想的**极简交集**:
> - OAIF 的 on-policy (chain rollout)
> - Samplers-DPO 的 reward-aware sampler (OPSD teacher)
> - OFS-DPO 的梯度稳定化 (α-mix GRPO,v2 EMA chase)
> - RPO 的 "外部信号注入但保持同模型" (teacher token splice)
> - Meta Bridging 的 best-vs-worst init pair + skip 策略
>
> **关键差异化**:
> - 用 verifiable env reward 替代 LLM judge / RM / hint quality
> - 用 token-level causal splice 替代独立采样 / hint 注入 / logit mixing
> - 用同模型 OPSD context 替代付费外部 API
>
> 这是 **chain rollout 时代的 online DPO**。

---

## 与 OPD 系列论文的关系

OPD 综述 84+ 篇论文,token-level signal 演化路径:

```
①  Distribution matching     MiniLLM, GKD, OPSD, SDPO       D_KL(π_T || π_S) 直接当 loss
   ↓
②  Heuristic salience        TIP, SCOPE, SelecTKD            entropy + divergence 手工组合
   ↓
③  Statistical evidence ratio RLSD, AlignDistil               P_T/P_S 调制 magnitude
   ↓
④  Sample-level routing      SRPO                            按 outcome reward 路由整条样本
   ↓
⑤  Token-level causal credit (TCCA-Lite, 我们之前的方法)        ΔR causal counterfactual
   ↓
⑥  Pairwise causal preference (DPO-TGS, 当前主推)             因果反事实 pair + DPO loss
```

**⑥ 阶段 = ⑤ 的演化 + ④ 的简洁度**:
- 保留 ⑤ 的因果反事实信号 (token splice + ΔR)
- 抛弃 ⑤ 的 advantage modulation (λ_div tuning 复杂)
- 走 ④ 的简洁路径 (DPO loss 替代 PPO advantage 计算)

详见 HTML "★ TCCA Token signal evolution path" 表格。
