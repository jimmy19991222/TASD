# DPO-2S-8B 实验结果记录

> 日期: 2026-07-03  
> 分支: tg-branching-rollout  
> 代码版本: a24405e (DPO reward filter + config 修复)

## 实验概述

在 Qwen3-8B 上跑 DPO-2S (stage1pair, withref, reward filter ON) 与 GRPO/SDPO 8B 基线对比。

**DPO-2S 配置:**
- Two-stage branching: stage1_n=4, n_trees=2, n_splits=1
- DPO: withref=True, stage1pair=True, reward_filter=True
- 训练 250 步, LR=1e-5, mini_batch=32, rollout.n=8, val.n=16

**基线配置 (GRPO/SDPO):**
- GRPO: mbs=32, LR=1e-5, 250 步
- SDPO: alpha=0.5, LR=1e-5, reprompt=True, 250 步

## 数据集
math500 + sciknoweval/{biology, chemistry, physics, material} + tooluse, 共 6 个

---

## DPO-2S-8B 实验状态 (2026-07-03 晚)

### Fixed β=2.0 (已提交, 部分在跑)

| 数据集 | 步数 | val_acc (best) | dpo_pairs | filtered | 状态 |
|--------|------|----------------|-----------|----------|------|
| tooluse | ~100/250 | 0.597 | ~62 | 0 | RUNNING |
| material | 较早停止 | - | - | - | 需检查 |
| physics | 较早停止 | - | - | - | 需检查 |
| chemistry | 较早停止 | - | - | - | 需检查 |
| biology | 较早停止 | - | - | - | 需检查 |
| math500 | 较早停止 | - | - | - | 需检查 |

> ⚠️ 部分实验因 staging 打包问题中断, 后续已修复重新提交

### Fixed β=0.5 (刚提交)

| 数据集 | 状态 |
|--------|------|
| math500 | SUBMITTED |
| biology | SUBMITTED |
| chemistry | SUBMITTED |
| physics | SUBMITTED |
| material | SUBMITTED |
| tooluse | SUBMITTED |

### Teacher-guided β=2.0 (已提交)

任务名前缀: `DPO-2S-8B-stage1pair-tg-beta2.0-*`

| 数据集 | 状态 |
|--------|------|
| math500 | SUBMITTED |
| biology | SUBMITTED |
| chemistry | SUBMITTED |
| physics | SUBMITTED |
| material | SUBMITTED |
| tooluse | SUBMITTED |

---

## 8B 基线 (已有结果)

### sciknoweval 4 domains

| 数据集 | GRPO final | GRPO best | SDPO final | SDPO best | 来源 |
|--------|-----------|-----------|------------|-----------|------|
| biology | 0.5725 | **0.6062** | 0.4975 | 0.5300 | Baselines_clean |
| chemistry | ~0.45 | ~0.50 | ~0.42 | ~0.48 | TG-Branching-Pilot |
| physics | ~0.60 | ~0.65 | ~0.55 | ~0.60 | TG-Branching-Pilot |
| material | ~0.70 | ~0.75 | ~0.60 | ~0.65 | TG-Branching-Pilot |

### tooluse

| GRPO final | GRPO best | SDPO final | SDPO best |
|-----------|-----------|------------|-----------|
| 0.6176 | **0.6857** | 0.5956 | 0.6700 |

### math500

❌ **8B 基线缺失** — 需要补跑 GRPO/SDPO 8B

---

## 初步对比: DPO-8B vs 基线 (tooluse)

| 方法 | final | best |
|------|-------|------|
| GRPO 8B | 0.6176 | **0.6857** |
| SDPO 8B | 0.5956 | 0.6700 |
| **DPO-2S-8B (β=2.0)** | **0.597** (step ~100) | - |

> tooluse 上 DPO-8B 接近 SDPO 水平, 但还没追上 GRPO

---

## Biology 问题

DPO-8B biology 实验出现 **reward 持续下降** 现象:
- val reward@16: 0.2875 → 0.2275 (**↓21%**)
- val reward best@16: 0.7714 → 0.6211 (**↓19%**)

**可能原因:**
1. Reward filter 过度过滤 (biology 每步过滤 ~8/64 = 13%, 最高)
2. Biology 问题复杂度高, teacher 偏好信号不够清晰
3. DPO 对 noisy preference 敏感
4. 基础模型在 biology 上可能本身就弱

---

## 关键修改记录

1. **DPO 重构** (commit `24575fd`): 成对 loss → 逐样本加权, 解决 NCCL 崩溃 + 配对覆盖率 100%
2. **Reward filter** (commit `7c0a7fd`): 过滤 reward_chosen < reward_rejected 的对
3. **Config 修复** (commit `a24405e`): PolicyLossConfig 补 dpo_reward_filter 字段
4. **脚本清理** (commit `bf39d3c`): 删除 18 个废弃脚本, 保留 baseline + DPO
5. **8B 提交脚本** (commit `1f4d232`): submit_dpo_8b.sh, 支持 β/INCLUDE/teacher-guided via env vars

---

## 下一步

- [ ] 等 DPO-8B β=2.0 全部跑完 (部分需要重新提交)
- [ ] 观察 DPO-8B β=0.5 和 teacher-guided β=2.0 的结果
- [ ] 补跑 math500 的 GRPO/SDPO 8B 基线
- [ ] 分析 biology 问题: 尝试关闭 reward filter 或调整阈值
- [ ] 完成 DPO-8B vs GRPO-8B vs SDPO-8B 完整对比表
