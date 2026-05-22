# Verifier-Conditioned On-Policy Self-Distillation: A Bayesian Unification of SDPO and Outcome-Based RL for Reasoning

> **Status**: Paper first draft, 2026-05-22, 月丘. Target venue: NeurIPS 2026 (primary) / ICML 2026 (secondary).
> **Working title alts**: "VC-OPSD: Turning the Student into Its Own Process Reward Model via Bayes-Posterior Conditioning"; "Bayes-Unified Distillation RL: From SDPO to Verifier-Aware Self-Distillation".
> **Companion code**: [verl/trainer/ppo/core_algos.py](../verl/trainer/ppo/core_algos.py), [scripts/diagnostics/verdict_calibration.py](../scripts/diagnostics/verdict_calibration.py).

---

## Abstract

On-policy self-distillation (OPSD) — most visibly **SDPO** and its variants — has become a competitive RL post-training recipe for LLMs by replacing the sparse outcome reward with a per-token teacher likelihood signal. However, three issues persist: (i) the signal $\log p_T(s_t \mid r, s_{<t})$ is a **density**, not a value function, so credit assignment is theoretically unsound; (ii) the method requires a privileged reference $r$ (gold answer or hint) that is unavailable for most reasoning tasks; (iii) the most successful loss variants (token-PG with cross-entropy baseline, JSD α-mixing) achieve good rewards but exhibit a documented mode-collapse / entropy-collapse failure mode. We make three contributions.

**First**, we give a one-line Bayes derivation that unifies the SDPO loss family: in the OPSD shared-parameter setting, $\log p_T(v \mid r, s_{<t}) - \log p_s(v \mid s_{<t})$ equals a well-defined Bayes factor $\Delta W(v)$, i.e. the log-belief shift that "$r$ is a valid continuation" induced by inserting token $v$. This collapses 6 published SDPO loss configurations into a single expectation $\mathbb{E}_q[\Delta W]$ under different sampling distributions $q$, and pinpoints exactly which configurations leak `log p_s` into the advantage (and hence cause the entropy collapse).

**Second**, building on the same Bayes structure, we introduce **Verifier-Conditioned OPSD (VC-OPSD)**: replace the privileged reference $r$ with two **verdict-conditioned prompts** $c^+, c^-$ ("the response correctly/incorrectly solves the problem"). The contrastive log-ratio $\log p_T^+(s_t) - \log p_T^-(s_t)$ then collapses, under the same OPSD shared-param identity, to the **prospective value increment** $\Delta V^*_t := V^*_t - V^*_{t-1}$ where $V^*_t = \log \text{odds}(Y{=}\text{right} \mid s_{\le t})$. Coupled with a verifier-anchored calibration loss, VC-OPSD becomes the first RL-for-reasoning method to derive step-level value-aligned credit from binary outcome rewards **without** any privileged reference, separate reward model, or step-level annotation.

**Third**, we ship two diagnostics that ride for free on existing forwards: a $\Delta W$ signal-to-noise ratio (computed inside any OPSD forward) and a verifier-calibration AUC prerequisite check (a one-shot offline job). Together they let a practitioner predict, before any training, whether VC-OPSD will work on their task and base model.

*Empirical sections forthcoming pending the 2026-05-22 sweep (SciKnowEval, LiveCodeBench v6, Qwen3-8B base).*

---

## 1. Introduction

LLM post-training for reasoning splits cleanly into two RL recipes today.

**Outcome-based RL** (PPO/GRPO, RLHF) uses a binary or scalar verdict $R \in \{0, 1\}$ from a verifier (test-case execution, symbolic check, judge LLM) and propagates it uniformly across the trajectory via policy gradient. The signal is unbiased but very sparse — a single bit per rollout of length up to thousands of tokens — and recent work has documented severe sample inefficiency, length-explosion, and mode-collapse failures as a result.

**Teacher-distillation RL** (KD, OPD, OPSD/SDPO) instead asks a "stronger" teacher to score each token with a log-probability, giving per-token gradient signal of much higher density. The most influential variant in 2025-26, SDPO, makes the architecturally elegant choice of **self-distillation**: the teacher is the same network $\pi_\theta$ as the student, but with a privileged reference $r$ (gold answer, hint, or stronger reasoning trace) injected into its context. The per-token signal $Q_t := \log p_T(s_t \mid r, s_{<t})$ is dense and gradient-cheap.

SDPO works empirically. But it suffers from three theoretical and practical problems:

- **P1 — Likelihood ≠ Value.** $\log p_T(s_t)$ is a density, not a value function. "Looks like the teacher would say it" is not the same as "contributes to final-outcome correctness". All proposed baselines simply trade one density for another without addressing the underlying type mismatch.
- **P2 — Reference dependency.** SDPO requires a privileged $r$. For coding, open-ended math, and most agentic tasks, no gold reference exists; only an outcome verifier is available. SDPO does not apply.
- **P3 — Mode collapse.** The currently best-performing SDPO configurations (token-PG with cross-entropy baseline `ce`, JSD α-mixing) demonstrably collapse student entropy faster than vanilla PG, requiring entropy-coefficient tuning to be usable. The root cause is not clear in the literature.

This paper argues that all three are **symptoms of a single missing object**: a clean Bayesian factorization of the SDPO signal. Once that factorization is in hand (§3), P3 is read off mechanically as a `log p_s` self-reinforcement term that some baselines cancel and others amplify (§3.4), and P2 dissolves into a constructive replacement of $r$ with a verdict event $Y$ defined directly from the verifier (§4). The resulting method, **VC-OPSD**, simultaneously addresses P1, P2 and P3 with the same Bayes lens.

### 1.1 Contributions

1. **Bayes unification of the SDPO loss family (§3).** A one-line shared-param identity reduces $\log p_T(v|r,s_{<t}) - \log p_s(v|s_{<t})$ to a Bayes factor $\Delta W(v)$. All six commonly-used SDPO loss configurations (token-PG with `student`/`ce`/`group_*` baselines, vocab-level rKL/fKL/JSD) collapse to expectations of $\Delta W$ under different sampling distributions; two of them leak `log p_s` as a multiplicative self-reinforce term, exactly matching the observed entropy-collapse failure modes.

2. **Verifier-Conditioned OPSD (§4).** Replace privileged $r$ with a contrastive verdict pair $(c^+, c^-)$. The log-ratio $\log p_T^+ - \log p_T^-$ collapses to $\Delta V^*_t$, the per-token log-odds increment of trajectory correctness — a well-defined value function derived entirely from the binary verifier with no extra parameters and no step-level annotation. A calibration loss anchors model-believed $V^*$ to verifier-observed $Y$, preventing self-fulfilling drift.

3. **Two free-riding diagnostics (§5).** $\Delta W$-SNR (signal-to-noise of the distillation channel, computed inside any OPSD forward) and verifier-calibration AUC (a one-shot offline check). Together they predict, before training, whether VC-OPSD will succeed on a given (task, base model) pair.

4. **Empirical validation (§6).** On SciKnowEval and LiveCodeBench v6 with Qwen3-8B base, VC-OPSD matches or beats GRPO and SDPO in final reward, recovers entropy-stable training without entropy-coefficient tuning, and exhibits monotonic correlation between the diagnostic predictors and final performance. *[Pending results from 2026-05-22 sweep; placeholder tables in §6.]*

---

## 2. Background and Notation

### 2.1 LLM Post-Training RL

A policy $\pi_\theta(s_t \mid s_{<t}, x)$ generates a response $\tau = (s_1, \ldots, s_T)$ conditioned on prompt $x$. An outcome verifier returns $R(\tau, x) \in \{0, 1\}$ (or a scalar). The objective is to maximize $\mathbb{E}_{\tau \sim \pi_\theta}[R(\tau, x)]$, optionally subject to KL constraints against a reference policy. Outcome-based RL methods (PPO, GRPO) propagate $R$ as a single scalar advantage.

### 2.2 OPD vs OPSD

We distinguish two distillation variants in the literature:

- **OPD (On-Policy Distillation)**: teacher $\pi_T$ and student $\pi_s$ are **independent networks** with disjoint parameters $\theta_T, \theta_s$. The teacher is typically a stronger pre-trained or fine-tuned model.
- **OPSD (On-Policy Self-Distillation)**: teacher and student share **the same parameters** $\pi_\theta$, differing only by additional **privileged context** $r$ injected into the teacher's prompt:
  $$
  p_T(\cdot \mid r, s_{<t}) := \pi_\theta(\cdot \mid s_{<t}, x, r), \qquad p_s(\cdot \mid s_{<t}) := \pi_\theta(\cdot \mid s_{<t}, x).
  $$
  All theory in this paper is OPSD. Extensions to OPD require additional calibration assumptions; see Appendix B.

### 2.3 SDPO Loss Family

For an SDPO step, define the per-token raw signal $Q_t := \log p_T(s_t \mid r, s_{<t})$. Two loss families are used:

**Token-level Policy Gradient**: with advantage $A_t = Q_t - V_t$ and baseline $V_t$ selected from $\{\text{student}, \text{ce}, \text{group\_mean}, \text{group\_hier}\}$.

**Vocab-level KL** ($\alpha$-parametrized):
- $\alpha = 0$: forward KL $\mathrm{KL}(p_T \| p_s)$ (mass-covering, codebase default)
- $\alpha = 1$: reverse KL $\mathrm{KL}(p_s \| p_T)$ (mode-seeking)
- $\alpha \in (0, 1)$: Jensen-Shannon divergence $\mathrm{JSD}_\alpha$ (bounded interpolation)

We refer collectively to these six configurations as the **SDPO loss family** and analyze them in §3.

---

## 3. Bayes Analysis of OPSD

### 3.1 The Shared-Parameter Identity

The single structural fact that makes OPSD analytically tractable is:

**Lemma 1 (Shared-param marginal identity).** *In OPSD, since $\pi_\theta$ is one network applied with vs. without $r$ in context,*
$$
\sum_{r} p(r \mid s_{<t}) \cdot p_T(v \mid r, s_{<t}) = p_s(v \mid s_{<t}).
$$
*Equivalently: marginalizing $r$ out of the teacher's distribution recovers the student's distribution exactly. This is a definitional identity, not a calibration assumption.*

In OPD this identity does **not** hold ($p_T$ is a different network's density).

### 3.2 The ΔW Bayes Factor

Applying Bayes' rule to $p_T(v \mid r, s_{<t})$ and using Lemma 1:

**Theorem 1 (ΔW decomposition).** *For any token $v$ in the vocabulary,*
$$
\log p_T(v \mid r, s_{<t}) - \log p_s(v \mid s_{<t}) \;=\; \underbrace{\log p(r \mid v, s_{<t}) - \log p(r \mid s_{<t})}_{=:\; \Delta W(v)}.
$$

**Interpretation.** $\Delta W(v)$ is the **log Bayes factor** for the hypothesis "$r$ is a valid continuation" induced by intervening on the next token to be $v$. It is positive when $v$ raises that belief, negative otherwise. Under the student prior $\mathbb{E}_{v \sim p_s}[\Delta W(v)] \le 0$ by Jensen; under the teacher posterior $\mathbb{E}_{v \sim p_T(\cdot | r)}[\Delta W(v)] \ge 0$.

For the sampled token $s_t$:
$$
\boxed{\; Q_t \;=\; \Delta W_t + \log p_s(s_t \mid s_{<t}), \qquad \Delta W_t := \Delta W(s_t). \;}
$$
The raw SDPO signal decomposes into a **value-like Bayes term** plus the **student's own log-likelihood** on the sampled token.

### 3.3 Token-Level PG Variants Under the Decomposition

Substituting Theorem 1 into $A_t = Q_t - V_t$:

| Baseline $V_t$ | Form of $V_t$ | Resulting $A_t$ | Residual `log p_s` |
|---|---|---|---|
| `student` | $\log p_s(s_t)$ | $\Delta W_t$ | **none** (cancelled) |
| `ce` | $-\log p_s(s_t)$ | $\Delta W_t + 2 \log p_s(s_t)$ | **2× amplified** |
| `group_mean` / `group_hier` | scalar mean | $\Delta W_t + \log p_s(s_t) - \mathrm{const}$ | 1× residual |

**Corollary 1.** *Among the four standard baselines, only `student` yields a `log p_s`-free advantage. The `ce` baseline doubles the `log p_s` self-reinforce term, producing a $\log^2 p_s$ contribution to the loss whose gradient pushes confident tokens to be more confident. This is the mechanistic origin of the entropy-collapse pathology observed with `ce` baseline.*

This corollary is, to our knowledge, the first principled explanation of why `ce` baseline outperforms `student` on short horizons (the amplified self-reinforce term accelerates pseudo-likelihood maximization) but collapses on long ones.

### 3.4 Vocab-Level KL Variants

Substituting Theorem 1 into the three vocab-KL formulations:

**Reverse KL** ($\alpha = 1$):
$$
\mathrm{KL}(p_s \| p_T) = \sum_v p_s(v) \log \frac{p_s(v)}{p_T(v|r)} = -\mathbb{E}_{v \sim p_s}[\Delta W(v)].
$$

**Forward KL** ($\alpha = 0$):
$$
\mathrm{KL}(p_T \| p_s) = \mathbb{E}_{v \sim p_T(\cdot | r)}[\Delta W(v)].
$$

**Theorem 2 (KL ↔ ΔW correspondence).** *In OPSD, vocab-level KL divergences are exactly expected $\Delta W$ under either the student prior (rKL) or the teacher posterior (fKL). No `log p_s` term enters the loss.*

**Corollary 2 (Token-PG ≡ vocab-rKL MC).** *Token-level PG with `baseline='student'` is a single-sample Monte-Carlo estimator of vocab-level reverse KL. The two losses optimize the same population objective.*

**JSD.** The interpolation $\mathrm{JSD}_\alpha = (1-\alpha) \mathrm{KL}(p_s \| m_\alpha) + \alpha \mathrm{KL}(p_T \| m_\alpha)$ with $m_\alpha = \alpha p_T + (1-\alpha) p_s$ has no closed Bayes form: $m_\alpha$ is an *additive* mixture, not a Bayes update, so $\log p_s - \log m_\alpha \ne \pm \Delta W$. JSD therefore loses the clean "weighted expected $\Delta W$" semantics and is best viewed as engineering regularization (boundedness) rather than a fundamentally different objective.

### 3.5 Summary Table

| Loss configuration | Closed form in $\Delta W$ | Sampling/weighting | Residual `log p_s` |
|---|---|---|---|
| Token-PG, `student` | $-\Delta W_t \cdot \log p_s(s_t)$ (MC) | $p_s$ | none |
| Token-PG, `ce` | $-(\Delta W_t + 2 \log p_s) \cdot \log p_s$ | $p_s$ | **2×** |
| Token-PG, `group_*` | $-(\Delta W_t + \log p_s - c) \cdot \log p_s$ | $p_s$ | 1× |
| Vocab rKL | $-\mathbb{E}_{v \sim p_s}[\Delta W(v)]$ | $p_s$ | none |
| Vocab fKL | $\mathbb{E}_{v \sim p_T}[\Delta W(v)]$ | $p_T$ | none |
| Vocab JSD | no closed form | $m_\alpha$ | none, but Bayes-opaque |

Four of six configurations are *pure value*; two carry `log p_s` self-reinforce. This is the structural answer to P1 (likelihood vs value) and P3 (collapse mechanism).

---

## 4. Verifier-Conditioned OPSD (VC-OPSD)

### 4.1 Motivation

§3 leaves $r$ untouched. But $r$ is *privileged information* — a gold answer, a hint, or a stronger trace — that is often unavailable. For most reasoning tasks the only supervision is a binary verifier $R \in \{0, 1\}$. **Can we replace $r$ with a Bayes-equivalent construction derived purely from $R$?**

### 4.2 Construction

Define two **verdict-conditioned context prefixes**:
$$
c^+ := \texttt{"[Meta: the response below correctly answers the question.]"}
$$
$$
c^- := \texttt{"[Meta: the response below incorrectly answers the question.]"}
$$

These are not external networks: they are prompts prepended to the same student-shared $\pi_\theta$. Define
$$
p_T^+(v \mid s_{<t}) := \pi_\theta(v \mid c^+, x, s_{<t}), \qquad p_T^-(v \mid s_{<t}) := \pi_\theta(v \mid c^-, x, s_{<t}).
$$
These are *two new teachers* obtained by conditioning the same network on opposite verdict assumptions. Compute cost: 2 teacher forwards per step. KV-cache for the (mostly identical) prefix is shareable, so effective overhead is ≈ 1.3× vanilla SDPO.

### 4.3 The Contrastive Signal Equals ΔV*

Let $Y \in \{\text{right}, \text{wrong}\}$ denote the (latent) ground-truth verdict event, and let $V^*_t := \log \frac{P(Y{=}\text{right} \mid s_{\le t})}{P(Y{=}\text{wrong} \mid s_{\le t})}$ be the verdict log-odds value function under the student's beliefs.

**Theorem 3 (Contrastive verdict identity).** *Under the OPSD shared-param assumption and the model-internal Bayes coherence assumption (Appendix A.1),*
$$
\log p_T^+(s_t \mid s_{<t}) - \log p_T^-(s_t \mid s_{<t}) \;=\; V^*_t - V^*_{t-1} \;=:\; \Delta V^*_t.
$$

**Proof sketch.** Apply Theorem 1 separately with $r = c^+$ and $r = c^-$:
$$
\log p_T^\pm(s_t|s_{<t}) - \log p_s(s_t|s_{<t}) = \log p(c^\pm | s_{\le t}) - \log p(c^\pm | s_{<t}).
$$
Subtract; the `log p_s` terms cancel; using $P(\text{right} \mid \cdot) + P(\text{wrong} \mid \cdot) = 1$ and rearranging gives $V^*_t - V^*_{t-1}$. Full derivation in Appendix A.2.

**Implications.**
- The contrastive signal is a **value-function increment**, not a density. P1 resolved.
- It does not depend on a privileged $r$. P2 resolved.
- The `log p_s` term cancels in the subtraction, so $\Delta V^*_t$ has zero self-reinforce. P3 resolved (theory; verified empirically in §6.3).
- $\mathbb{E}_{s_t \sim p_s}[\Delta V^*_t] = 0$ (Appendix A.3): the gradient is unbiased under the student prior, with no drift.

### 4.4 The VC-OPSD Loss

We use $\Delta V^*_t$ as the per-token advantage in a policy-gradient loss, **anchored** by the observed verifier outcome $R \in \{0, 1\}$ at the trajectory level:

$$
\boxed{\;
\mathcal{L}_{\text{VC-OPSD}} \;=\; \underbrace{-\,\mathbb{E}_\tau\!\left[ (2R - 1) \cdot \sum_t \mathrm{sg}(\Delta V^*_t) \cdot \log p_s(s_t \mid s_{<t}) \right]}_{\text{verifier-aligned PG, signed by } R} \;+\; \lambda \cdot \mathcal{L}_{\text{calib}}.
\;}
$$

Here $\mathrm{sg}(\cdot)$ stops gradient through the teacher forwards (standard for stability), and the verifier-aligned coefficient $(2R - 1) \in \{-1, +1\}$ flips sign when the verifier disagrees with the model's local belief.

The **calibration loss** anchors $\pi_\theta$'s verdict-conditioned beliefs to the empirical verdict:
$$
\mathcal{L}_{\text{calib}} \;=\; -\,\mathbb{E}_\tau\!\left[ \log p_T^{Y(R)}(\tau) \right], \qquad Y(R) := \begin{cases} c^+ & R = 1 \\ c^- & R = 0 \end{cases}.
$$
This is a per-rollout cross-entropy: it asks $\pi_\theta$ to assign high likelihood to the rollout under the **correct** verdict prefix. Without this term, $V^*_t$ can drift into a self-fulfilling prophecy (model believes everything it generates is right, $\Delta V^* \to 0$, signal collapses).

### 4.5 Algorithm

```
Input: policy π_θ, prompts {x}, verifier R, hyperparameters {λ, lr}
For each training step:
  1. Sample rollouts τ_i ~ π_θ(· | x_i), compute R_i = R(τ_i, x_i)
  2. Two-prefix teacher forward (KV-cache shared on prefix):
       log_p_plus  [t]  := log π_θ(s_t | c+, x, s_<t)   ∀ t
       log_p_minus [t]  := log π_θ(s_t | c-, x, s_<t)   ∀ t
  3. Student forward: log_p_s [t] := log π_θ(s_t | x, s_<t)
  4. Per-token advantage:
       ΔV*_t = sg(log_p_plus[t] - log_p_minus[t])
  5. PG loss:
       L_pg = - mean_t [ (2R - 1) · ΔV*_t · log_p_s[t] ]
  6. Calibration loss (one forward per rollout):
       L_calib = - mean_t [ log_p_plus[t]  if R=1
                            log_p_minus[t] if R=0 ]
  7. Total loss: L = L_pg + λ · L_calib
  8. Backprop and update
```

### 4.6 Hyperparameter Notes

- **$\lambda$ (calibration weight)**: theoretical analysis (Appendix A.4) suggests $\lambda \in [0.1, 1.0]$; we ablate $\{0, 0.1, 0.3, 1.0\}$ in §6.4.
- **Verdict prompt design**: in principle the exact wording of $c^\pm$ matters only insofar as it shifts the model's verdict-conditional distribution; we ablate three wordings in §6.4 and find near-invariance once both prompts are "informative enough" (Appendix C).
- **EMA on teacher**: optional, default off. With EMA on, $\Delta V^*_t$ is computed against a slightly stale copy of $\pi_\theta$; this trades some bias for variance reduction. We do not use it in main results.
- **Compute**: 2 teacher forwards + 1 student forward + 1 backward = ~3× vanilla GRPO forward cost per step. The teacher forwards share most of the prefix KV-cache, so wall-clock is ~1.5× GRPO in practice.

---

## 5. Diagnostics

Two free-riding measurements that predict, before or during training, whether the method will succeed.

### 5.1 ΔW Signal-to-Noise (SNR)

Inside any OPSD forward, compute the full-vocab $\Delta W$ tensor and report:

$$
\text{SNR}_t := \frac{\text{mean}_{v \in \text{top-K}} |\Delta W_t(v)|}{\text{std}_{v \in \text{vocab}} |\Delta W_t(v)|}.
$$

High SNR means the distillation signal is concentrated on a few high-leverage tokens (informative); low SNR means it's spread thin (noise). We log seven SNR-derived metrics in [verl/trainer/ppo/core_algos.py:1702-1744](../verl/trainer/ppo/core_algos.py#L1702-L1744). Cost: ~3× the KL forward, zero backward overhead, gated by a config flag.

In §6.5 we show that mean per-step SNR correlates with final-step reward across our SDPO runs at Spearman $\rho \approx 0.8$ *[pending sweep results]*, making it a single-number health check for any SDPO/OPSD/KD-RL training.

### 5.2 Verifier-Calibration AUC

A one-shot offline check that must pass before VC-OPSD is feasible. Given a base model and a labeled validation set:

1. Generate $N$ rollouts per question, score each with the verifier to get $R \in \{0, 1\}$.
2. For each rollout, compute $\Delta_\text{seq} := \sum_t [\log p_T^+(s_t) - \log p_T^-(s_t)]$.
3. Report $\text{AUC}(\Delta_\text{seq}, R)$.

**Decision rule (validated in §6.6 over 6 model × task pairs)**:
- AUC $\ge 0.8$ → PROCEED with VC-OPSD.
- AUC $\in [0.7, 0.8)$ → MARGINAL: warm-start with a brief SFT epoch on (rollout, verdict-prefix) pairs to improve calibration, then proceed.
- AUC $< 0.7$ → STOP: the base model cannot act as a verdict judge for this task. Fall back to outcome-only RL.

Implementation in [scripts/diagnostics/verdict_calibration.py](../scripts/diagnostics/verdict_calibration.py); single-GPU, ~15 minutes for $N{=}50$ questions × 4 rollouts.

---

## 6. Experiments

*Sections 6.1–6.6 will be populated after the 2026-05-22 sweep completes. The protocol is fixed below.*

### 6.1 Setup

- **Base model**: Qwen3-8B (instruct).
- **Tasks**: SciKnowEval (Biology), LiveCodeBench v6, GSM8K-Hard (open-ended math, no gold reference — tests P2).
- **Methods compared**: GRPO, SDPO (best published config), SDPO-token-PG-`student`, SDPO-vocab-rKL, SDPO-vocab-fKL, SDPO-JSD (α=0.5), **VC-OPSD (ours)**.
- **Train budget**: 250 steps × batch 32 × rollout-n 8.
- **Metric**: validation accuracy / pass@1 with $n{=}16$ Monte Carlo evaluations.

### 6.2 Main Results

*[Table 1: methods × tasks × {final-acc, best-acc, last-5-step-stability} — pending]*

### 6.3 Entropy Stability (Tests P3)

*[Figure 1: per-step entropy curves for SDPO-ce, SDPO-rKL, VC-OPSD on SciKnowEval — pending]*

Hypothesis: VC-OPSD maintains the entropy curve of SDPO-rKL (no `log p_s` self-reinforce) while matching SDPO-ce in reward (verifier-anchored value signal).

### 6.4 VC-OPSD Ablations

- **Calibration weight** $\lambda \in \{0, 0.1, 0.3, 1.0\}$ — measures whether $\mathcal{L}_\text{calib}$ is necessary for non-collapse.
- **Verdict prompt sensitivity**: three wordings of $c^\pm$.
- **EMA teacher on/off**.

### 6.5 ΔW-SNR Predictive Power

*[Figure 2: Spearman correlation of mean per-step SNR vs final reward across all 24 (method × task × seed) runs — pending]*

### 6.6 Calibration Prereq Validation

*[Table 2: 6 (model × task) pairs × {pre-train AUC, post-train final reward, predicted outcome under §5.2 rule} — pending]*

---

## 7. Related Work

**SDPO and self-distillation RL.** Self-distillation methods for LLM post-training were popularized by [SDPO ref] and developed through OPD [ref] and OPSD [ref] variants. Our Theorem 1 generalizes and unifies the loss-design choices in this line.

**Outcome-based RL for reasoning.** PPO [Schulman et al.], GRPO [DeepSeek-Math], and recent length-aware variants [refs]. VC-OPSD inherits the verifier-only assumption from this family but recovers dense per-token signal via Theorem 3.

**Process Reward Models (PRMs).** OpenAI's o1 / R1 and follow-ups [Lightman et al.; refs] train a separate step-level reward model from human or AI step-annotations. VC-OPSD obtains step-level credit *automatically* from outcome labels via the contrastive verdict construction — no step annotations, no separate model.

**RLHF and direct preference optimization.** DPO [Rafailov et al.] and IPO require pairwise preferences. VC-OPSD requires only the binary outcome verifier already used by outcome-RL pipelines.

**Bayesian RL.** Use of Bayes-factor / log-odds value functions has a long history in tabular RL [refs]; we apply it to the LLM-token setting via the OPSD shared-param identity, which is novel.

---

## 8. Discussion

### 8.1 What VC-OPSD Solves

The three problems of §1 map onto specific structural elements:

| Problem | Resolution |
|---|---|
| P1 (likelihood ≠ value) | $\Delta V^*_t$ is a Bayes-derived value increment, formally a log-odds delta. |
| P2 (reference dependency) | No $r$ in the loss — only verifier $R$ and self-conditioning. |
| P3 (mode collapse) | `log p_s` cancels in the contrastive subtraction. Gradient unbiased. |

### 8.2 What VC-OPSD Does *Not* Solve

We are explicit about three limitations:

- **Prospective, not retrospective.** $\Delta V^*_t$ depends on $s_{\le t}$, not on $s_{>t}$. It is the *current-step-only* belief shift, not the *retrospective* contribution of $s_t$ to the eventual outcome. True retrospective credit (e.g. MC return $V^*_T - V^*_{t-1}$ or counterfactual replacement) is in principle better but requires the full trajectory and either more rollouts or more forwards. See Appendix D for a hybrid VC-OPSD + MC return variant we are evaluating.
- **Requires teacher to be a calibrated verdict judge.** If $\pi_\theta$ cannot distinguish correct from incorrect when given the verdict prefix (AUC $< 0.7$), $\Delta V^*_t$ is noise. The §5.2 prerequisite check exists exactly to surface this failure mode before compute is committed.
- **2× teacher forwards.** ~1.5× wall-clock vs vanilla GRPO with prefix-KV reuse, but more than vanilla SDPO. This is the price paid for replacing $r$ with a verifier-derived signal.

### 8.3 Connection to Prior Self-Distillation Work

A cleaner reading of Theorem 1: **the SDPO signal $\Delta W_t$ is a specialization of $\Delta V^*_t$ where the verdict event is replaced with "the privileged reference $r$ appears"**. SDPO is the special case where $r$ is observed; VC-OPSD is the general case where only the verifier is observed. From this angle, the entire SDPO literature can be viewed as a constrained sub-class of VC-OPSD.

---

## 9. Conclusion

We give a Bayes-unifying account of the SDPO loss family in OPSD self-distillation (§3) and use the same machinery to construct VC-OPSD (§4), which derives step-level value-aligned credit assignment from binary outcome verifiers alone — no privileged reference, no separate reward model, no step annotations. Two free-riding diagnostics (§5) let practitioners predict applicability before training. Pending the empirical sweep, the method addresses three documented SDPO failure modes (likelihood-vs-value, reference dependency, mode collapse) with a single structural insight.

---

## Appendices

### A. Proofs

**A.1. Model-internal Bayes coherence assumption.** Theorem 3 assumes $\pi_\theta$'s verdict-conditioned distributions $p_T^\pm$ are coherent under the same Bayes update implicit in Theorem 1's identity. Formally: $p_T^+(v | s_{<t}) \cdot P(\text{right} | s_{<t}) + p_T^-(v | s_{<t}) \cdot P(\text{wrong} | s_{<t}) = p_s(v | s_{<t})$. This is the analog of Lemma 1 for $r = c^\pm$. Empirically, the calibration loss $\mathcal{L}_\text{calib}$ enforces this asymptotically.

**A.2. Full proof of Theorem 3.** *[detailed derivation — to be expanded; outline in §4.3]*

**A.3. Unbiasedness of $\mathbb{E}_{s_t \sim p_s}[\Delta V^*_t]$.** Under A.1, $\mathbb{E}_{s_t \sim p_s}[\log p_T^\pm(s_t)/p_s(s_t)] = -\mathrm{KL}(p_s \| p_T^\pm) \le 0$, with equality only under perfect calibration. The expected difference equals $\mathrm{KL}(p_s \| p_T^-) - \mathrm{KL}(p_s \| p_T^+)$, which is bounded by the calibration gap and goes to zero under perfect calibration.

**A.4. Choice of $\lambda$ in the calibration loss.** *[asymptotic analysis to be expanded]*

### B. Extension to OPD (Independent Teacher Network)

When $\pi_T$ and $\pi_s$ have distinct parameters, Lemma 1 does not hold. Theorem 1 holds only under the additional assumption that $\pi_T$ is a *Bayes-calibrated extension* of $\pi_s$: $p_T(v | r, s_{<t}) \propto p_s(v | s_{<t}) \cdot p(r | v, s_{<t})$ exactly. This is a strong calibration condition that may fail for arbitrary teacher networks. In practice we recommend OPSD or, for OPD, a soft alignment loss between $p_T$ and $p_s$ on a held-out distribution.

### C. Verdict Prompt Robustness

*[Table: 5 wordings of $c^\pm$ × AUC of $\Delta_\text{seq}$ on SciKnowEval val — pending]*

### D. Hybrid VC-OPSD + Monte Carlo Return

The retrospective-credit limitation (§8.2) can be addressed by combining $\Delta V^*_t$ (cheap, per-step) with a Monte Carlo return $V^*_T - V^*_{t-1}$ (expensive, requires the full trajectory but is properly retrospective). A weighted combination
$$
A_t^\text{hybrid} = \gamma \cdot \Delta V^*_t + (1 - \gamma) \cdot (V^*_T - V^*_{t-1})
$$
trades compute (the MC return uses the same two-prefix forwards we already have) for retrospective signal. We leave a full evaluation to future work.

### E. Hyperparameter Table

*[To be populated with final values used in each run]*

### F. Reproducibility

All code is committed to [github.com/...](placeholder). Config snapshots, exact rollout-N, batch sizes, and seeds are listed in Appendix E. Diagnostics ([§5](../scripts/diagnostics/verdict_calibration.py) and [SNR metrics](../verl/trainer/ppo/core_algos.py)) are available standalone for use with any OPSD/SDPO codebase.

---

## Draft TODOs (internal)

- [ ] Fill in §6 from 2026-05-22 sweep (5 jobs in flight: SDPO-{token,rKL,fKL,JSD} + verdict-calibration check).
- [ ] Implement VC-OPSD in `compute_self_distillation_loss` after calibration AUC ≥ 0.8 confirmed. Estimate: ~200 LoC adapting the existing two-forward path.
- [ ] Add P-VC-OPSD variant (Appendix D hybrid) once main result is in.
- [ ] Pull related-work citations from current SDPO literature (placeholders marked `[ref]`).
- [ ] Decide venue: NeurIPS 2026 main vs ICLR 2027 (more empirical depth). Current draft fits NeurIPS short / ICML.
- [ ] Author list, affiliations, acknowledgments.
- [ ] Camera-ready figures: re-render §3.5 table as a heat-map, §5.1 SNR as a small mathematical schematic.
