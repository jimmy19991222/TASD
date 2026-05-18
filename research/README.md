# Research Documentation — DPO-TGS

> **当前主推方法**: **DPO-TGS V2.5** (On-Policy DPO + Teacher-Guided Sampling)
> **V2.5 spec freeze**: `299eec7` (2026-05-17) — 3 loss innovations + 16 metrics 完工
> **最新代码**: [`teacher-guided-intervention`](https://github.com/jimmy19991222/TASD/tree/teacher-guided-intervention) @ `560c761` (2026-05-18) — 含 FSDP chunk padding + ray/NCCL 鲁棒化 8 个 fix
> **最近 commit**: `fix(dpo_tgs): notebook NCCL bootstrap loopback + ray IP detection fallback`

---

## 📚 阅读顺序 (5 篇 active 文档)

| 阅读顺序 | 文件 | 时长 | 内容 |
|---|---|---|---|
| 1 | [README.md](README.md) | 3 min | 入口,当前状态速览 (本文) |
| 2 | [01_evolution.md](01_evolution.md) | 10 min | **来龙去脉**: TGDI → TCCA → TCCA-Lite → DPO-TGS V2.5 |
| 3 | [02_dpo_tgs_design.md](02_dpo_tgs_design.md) | 20 min | **当前方法完整 spec**: V2 adaptive rollout + V1 线性化 DPO + V2.5 三个 loss innovations |
| 4 | [03_theory_anchor.md](03_theory_anchor.md) | 10 min | **5 篇理论锚点**摘要 + 7 设计原则 (OAIF + OFS-DPO + Samplers + RPO + Meta) |
| 5 | [04_experiments.md](04_experiments.md) | 10 min | **实验状态**: 已提交 4 nebula tasks + 待跑 + baseline 数据 + 风险 |
| ref | [submission_guide.md](submission_guide.md) | 5 min | nebula 提交流程 |
| ref | [`OPD_Deep_Analysis.html`](file:///Users/ljm/lazada/papers/raw/opd_papers/OPD_Deep_Analysis.html) | 30 min | OPD 综述 + 5 篇 Online DPO 论文深度分析 (本地 obsidian wiki) |

---

## 🎯 论文一句话 (one-liner)

> **DPO-TGS**: 把 DPO 的 (chosen, rejected) pair 来源从"人工标注 / 独立采样"换成 **on-policy chain rollout 中 teacher 单 token 干预产生的因果反事实样本**,并在 loss 层面提出 3 个针对 teacher-guided 设定的创新 (Causal-Localized + Teacher-Anchored + ΔR-Weighted)。

**核心差异化** (vs OPD/Online DPO 现有工作):
- vs **OAIF**: 用 verifiable env reward 替代 LLM judge,用 teacher counterfactual 替代独立采样
- vs **OFS-DPO**: 用 α-mix GRPO fallback 防梯度消失,v2 可加 EMA chase
- vs **Samplers-DPO** (ICLR-25): OPSD teacher = privileged-info-aware sampler 的工程实现 → 直接获得 quadratic 收敛理论
- vs **RPO**: token-level vs sequence-level 信号粒度;同模型 OPSD 替代付费 GPT-4V
- vs **TCCA-Lite (我们之前的方法)**: pairwise preference 替代 advantage modulation,消除 λ_div tuning

---

## 🧭 当前状态速览 (2026-05-18)

| 维度 | 状态 |
|---|---|
| **代码** | ✅ V2.5 完整实现 (`560c761`),含 V2 adaptive rollout + 3 loss innovations + 16 metrics + DPO val mode + FSDP chunk padding |
| **本地测试** | ⏳ tiny / smoke / innov 三模式可选 (1-GPU/4-GPU);见 `run_notebook_dpo_tgs.sh` |
| **Nebula 任务** | 🔄 第 4 批 3 task 已排队 (commit `560c761`,FSDP padding bug 修后重交,见 [04_experiments.md](04_experiments.md) §2) |
| **Baseline 数据** | ⚠️ 单 seed 方差 7.5% 是论文级风险;biology GRPO best=0.660 (length=17 蒙对) vs 0.585 (健康) |
| **5 papers 理论锚点** | ✅ 已在 HTML 摘要 + [03_theory_anchor.md](03_theory_anchor.md) 本地版 |
| **风险** | 见 [04_experiments.md §6 论文级风险](04_experiments.md) |

## 🔑 凭证

`.env` 在项目根 (gitignore'd)。提交 nebula 前:

```bash
set -a; source .env; set +a
```

需要 `OPENLM_TOKEN`, `OSS_ACCESS_ID`, `OSS_ACCESS_KEY`, `SWANLAB_API_KEY`。

## 📂 代码位置 (DPO-TGS V2.5)

| 模块 | 文件 |
|---|---|
| **adaptive rollout** (Phase 1-4) | [`verl/trainer/ppo/dpo_tgs/adaptive_rollout.py`](../verl/trainer/ppo/dpo_tgs/adaptive_rollout.py) |
| **DPO loss + 3 innovations** | [`verl/trainer/ppo/dpo_tgs/dpo_loss.py`](../verl/trainer/ppo/dpo_tgs/dpo_loss.py) |
| **pair collector** (chain_consecutive / hybrid_init_chain) | [`verl/trainer/ppo/dpo_tgs/pair_collector.py`](../verl/trainer/ppo/dpo_tgs/pair_collector.py) |
| **ray_trainer dispatch + DPO val mode** | [`verl/trainer/ppo/ray_trainer.py`](../verl/trainer/ppo/ray_trainer.py) |
| **Hydra config** (16 个 knob,默认 v1 backwards compat) | [`verl/trainer/config/dpo_tgs.yaml`](../verl/trainer/config/dpo_tgs.yaml) |
| **nebula 提交** | [`nebula_scripts/dpo_tgs/`](../nebula_scripts/dpo_tgs/), [`nebula_scripts/submit_dpo_tgs_sweep.sh`](../nebula_scripts/submit_dpo_tgs_sweep.sh) |
| **本地 smoke (1-GPU tiny / 4-GPU)** | [`run_notebook_dpo_tgs.sh`](../run_notebook_dpo_tgs.sh) (tiny / smoke / innov / pair / full 五模式) |

## 🗄️ Archive (历史/弃用方案)

[archive/](archive/) 保留所有历史设计文档:

| 文件 | 时期 | 弃用原因 |
|---|---|---|
| [archive/tgdi_design_history.md](archive/tgdi_design_history.md) | 2026-05-14 早期 | TGDI 单步设计已被 TCCA / TCCA-Lite 取代 |
| [archive/tcca_v2_design.md](archive/tcca_v2_design.md) | 2026-05-16 | chain rollout 8 步串行,工程复杂度高,pivot 到 TCCA-Lite |
| [archive/tcca_lite_paper_idea.md](archive/tcca_lite_paper_idea.md) | 2026-05-16 21:50 | TCCA-Lite paper idea,后 pivot 到 DPO-TGS (pairwise preference 替代 advantage mod) |
| [archive/tcca_lite_progress.md](archive/tcca_lite_progress.md) | 2026-05-16 22:00 | TCCA-Lite 实验进展,DPO-TGS 后被 [04_experiments.md](04_experiments.md) 取代 |
| [archive/dpo_tgs_v1_design.md](archive/dpo_tgs_v1_design.md) | 2026-05-17 早期 | DPO-TGS V1 设计文档,V2.5 update 后被 [02_dpo_tgs_design.md](02_dpo_tgs_design.md) 取代 |
| [archive/self_teacher_advantage.md](archive/self_teacher_advantage.md) | 2026-05-07 | mode collapse 死掉 |
| [archive/on_policy_sd_dpo.md](archive/on_policy_sd_dpo.md) | 2026-05-12 | 早期 token-level on-policy DPO 路线,已弃 |
| [archive/proposal_review.md](archive/proposal_review.md) | 2026-05-12 | 早期 proposal,已演化 |
| [archive/quick_submit_guide.md](archive/quick_submit_guide.md) | 2026-05-07 | 已合并到 submission_guide.md |
| [archive/experiment_submission_guide.md](archive/experiment_submission_guide.md) | 2026-05-07 | 已合并到 submission_guide.md |

## 📊 SwanLab Projects

| Project | 内容 |
|---|---|
| **DPO-TGS** | 当前主项目,Phase B 4 tasks (commit `299eec7`) |
| `TGDI-Tier3` | TCCA-Lite 时代的实验 (历史) |
| `TGDI-local` | 本地 smoke (历史) |
| `PriorShift-Tier1` | Prior-Shift 实验 (上游 baseline) |
| `Baselines_v2`, `Baselines_v3` | GRPO/SDPO baseline (双 seed,单 seed 方差 7.5%) |
