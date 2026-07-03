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

## 8B 基线汇总 (Best Peak, mean@16 / tooluse mean@4)

跨多个项目 (Baselines_clean, TG-Branching-Pilot, Baselines_v2, Baselines_v3, Baselines, TASD_param_search) 汇总最佳结果：

| Dataset | GRPO Peak | SDPO Peak | GRPO Final | SDPO Final | #GRPO runs | #SDPO runs |
|---------|:---------:|:---------:|:----------:|:----------:|:----------:|:----------:|
| **biology** | **0.6600** | 0.5900 | 0.6600 | 0.5787 | 14 | 15 |
| **chemistry** | **0.7795** | 0.7476 | 0.7795 | 0.7438 | 6 | 6 |
| **physics** | **0.7820** | 0.6539 | 0.7672 | 0.6508 | 6 | 6 |
| **material** | 0.7959 | **0.7999** | 0.7899 | 0.7739 | 6 | 6 |
| **tooluse** | **0.6012** | 0.5800 | 0.5350 | 0.5112 | 3 | 3 |
| **math500** | ❌ Missing | ❌ Missing | - | - | - | - |

**关键发现：**
- **GRPO 在 4/5 数据集上优于 SDPO**（biology +7%, chemistry +3%, physics +13%, tooluse +2%）
- **material 是唯一 SDPO 略胜的数据集**（0.800 vs 0.796，差距极小）
- **math500 两个 baseline 都缺失**，需要补跑

---

## 初步对比: DPO-8B vs GRPO-8B vs SDPO-8B (tooluse)

| 方法 | final | best |
|------|-------|------|
| GRPO 8B (baseline) | 0.5350 | **0.6012** |
| SDPO 8B (baseline) | 0.5112 | 0.5800 |
| **DPO-2S-8B (β=2.0)** | **0.597** (step ~100) | - |

> tooluse 上 DPO-8B 接近 SDPO 水平, 但还没追上 GRPO

---

## Biology 问题

DPO-8B biology 实验出现 **reward 持续下降** 现象:
- val reward@16: 0.2875 → 0.2275 (**↓21%**)
- val reward best@16: 0.7714 → 0.6211 (**↓19%**)

**对比 8B baseline (Biology):**
| 方法 | Best Peak |
|------|:---------:|
| GRPO 8B | **0.6600** |
| SDPO 8B | 0.5900 |
| DPO-2S-8B (β=2.0, step 30) | 0.2875 ⚠️ |

**差距巨大**：DPO-8B 在 biology 上的峰值只有 GRPO 的 44%，SDPO 的 49%。

**可能原因:**
1. Reward filter 过度过滤 (biology 每步过滤 ~8/64 = 13%, 最高)
2. Biology 问题复杂度高, teacher 偏好信号不够清晰
3. DPO 对 noisy preference 敏感
4. 基础模型在 biology 上可能本身就弱
5. **β=2.0 可能过大**，导致训练不稳定

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

---

## 综合分析

### 8B Baseline 格局

从 6 个项目的历史数据来看，**GRPO 在 4/5 数据集上优于 SDPO**：

| Dataset | GRPO Best | SDPO Best | 差距 |
|---------|:---------:|:---------:|:----:|
| biology | **0.660** | 0.590 | GRPO +12% |
| chemistry | **0.780** | 0.748 | GRPO +4% |
| physics | **0.782** | 0.654 | GRPO +20% |
| material | 0.796 | **0.800** | SDPO +0.5% |
| tooluse | **0.601** | 0.580 | GRPO +4% |
| math500 | ❌ | ❌ | - |

**结论**：
- Physics 上 GRPO 优势最大（+20%），说明 reward signal 清晰时 GRPO 更强
- Material 是唯一 SDPO 略胜的数据集，但差距极小（0.4%）
- 整体而言 GRPO 更稳定，SDPO 在 noisy 环境下表现不佳

### DPO-8B 初步表现

| Dataset | DPO β=2.0 (当前) | GRPO Baseline | SDPO Baseline | 评估 |
|---------|:----------------:|:-------------:|:-------------:|:----:|
| biology | 0.288 (step 30) | 0.660 | 0.590 | ❌ 严重落后 |
| tooluse | 0.597 (step ~100) | 0.601 | 0.580 | ⚠️ 接近 SDPO |

**Biology 问题严重**：DPO-8B 只有 baseline 的 44-49%，需要排查。

### 建议方向

1. **降低 β**：β=2.0 可能过大，导致 training instability，尝试 β=0.5
2. **关闭 reward filter**：biology 过滤比例最高（13%），可能丢失有价值的 learning signal
3. **检查 teacher quality**：biology 的 teacher preference 可能不够 reliable
4. **等待 β=0.5 和 teacher-guided 结果**：看是否有改善
5. **如果持续不佳**：考虑在 biology 上放弃 DPO，专注 GRPO
