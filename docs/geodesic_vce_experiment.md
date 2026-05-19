# Geodesic-VCE 融合实验：技术文档

## 📋 实验概览

本实验旨在验证 **Geodesic（测地线）约束** 能否解决 **V_CE（Cross-Entropy Baseline）** 导致的熵崩溃（Entropy Collapse）问题，并对比 SDPO 与 V_CE 两种范式在引入 Geodesic 后的性能表现。

---

## 🎯 核心科学问题

### 问题 1: V_CE 是否真的导致 Entropy Collapse？

**背景：**
- V_CE 使用 token-level 的交叉熵期望作为 advantage baseline
- 公式：`A_raw = Q - V_CE = log π_teacher(y_t) - E_{π_s}[log π_teacher]`
- 当 student 分布集中（低熵）时，V_CE ≈ Q，导致 advantage 消失
- 模型失去探索动力，发生熵崩溃

**验证方法：**
- 对比实验 A（SDPO baseline）和实验 B（V_CE baseline）
- 监控指标：`actor/entropy`（策略熵）、`actor/adv_std`（advantage 标准差）
- 预期：B 的熵值会显著低于 A，且随训练步数持续下降

---

### 问题 2: Geodesic 能否免疫 V_CE 的熵崩溃？

**理论依据：**
- Geodesic 约束在概率流形上做 Natural Gradient 更新
- Fisher 度量加权：`manifold_weight = 1 / F ≈ 1 / [p(1-p)]`
- 当 token 概率 `p → 0` 或 `p → 1`（低熵）时，Fisher 度量 `F → 0`
- 导致 `manifold_weight → ∞`（经 trust region 截断后为有限大值）
- **效果：** 低熵 token 获得更大的梯度更新，强制模型"反思"过度自信的区域

**验证方法：**
- 对比实验 B（V_CE baseline）和实验 C（V_CE + Geodesic）
- 预期：C 的熵值稳定或上升，advantage 标准差保持健康水平

---

### 问题 3: SDPO+Geodesic vs VCE+Geodesic 哪个更强？

**理论对比：**
- **SDPO:** 直接用 KL/JSD 蒸馏损失（不需要 advantage）
  - 优点：稳定，不依赖 advantage 估计
  - 缺点：可能缺乏对"困难 token"的针对性优化
  
- **V_CE:** 用 advantage + policy gradient
  - 优点：能针对性优化高 advantage 的 token
  - 缺点：advantage 估计噪声大，容易熵崩溃

**验证方法：**
- 对比实验 C（V_CE + Geodesic）和实验 D（SDPO + Geodesic）
- 预期：D 在稳定性上占优，C 在收敛速度上可能更快

---

## 🔬 实验设计

### 四组实验矩阵（Seed=42 严格控制）

| 实验 | 名称 | Loss Mode | use_vce | use_geodesic | 核心目的 |
|------|------|-----------|---------|--------------|----------|
| A | SDPO baseline | `sdpo` | False | False | 对照组 |
| B | V_CE baseline | `self_teacher` | True | False | 验证熵崩溃复现 |
| C | V_CE + Geodesic | `self_teacher` | True | True | 验证 Geodesic 免疫熵崩溃 |
| D | SDPO + Geodesic | `sdpo` | False | True | 验证 Geodesic 在 SDPO 上的效果 |

### 固定超参（所有实验共享）

```yaml
# 数据集
dataset: "sciknoweval/biology"

# 训练参数
seed: 42
lr: 1e-5
train_batch_size: 32
rollout_n: 8
max_steps: 250

# 模型
model_name: "Qwen3-8B"

# SDPO 参数
alpha: 0.5  # Jensen-Shannon divergence
distill_topk: 100
dont_reprompt_on_self_success: True

# V_CE 参数
clip_value: 3.0  # advantage clip threshold
adv_std_floor: 0.0

# Geodesic 参数
geodesic_trust_region: 5.0  # Trust Region 阈值
geodesic_beta_scale: 0.5    # β 缩放因子
```

---

## 🛠️ 核心算法实现

### 1. Geodesic 约束（Fisher 度量加权）

**位置：** `verl/trainer/ppo/core_algos.py` 的 `compute_self_distillation_loss`

**核心代码：**

```python
# ===== Geodesic SDPO: Fisher 度量加权 (流形约束) =====
use_geodesic = getattr(self_distillation_config, "use_geodesic", False)
if use_geodesic:
    # 计算 Fisher 对角近似 F ≈ p(1-p)
    if student_all_log_probs is not None:
        # Full logit 模式：在 vocab 维度计算 Fisher
        p_student = torch.exp(student_all_log_probs.detach())  # (B, T, V)
        fisher_metric = (p_student * (1.0 - p_student)).sum(-1)  # (B, T)
    else:
        # Token-level 模式：使用采样 token 的概率
        p_student = torch.exp(student_log_probs.detach())  # (B, T)
        fisher_metric = p_student * (1.0 - p_student)
    
    fisher_metric = fisher_metric + 1e-5
    
    # 计算原始 manifold weight
    raw_manifold_weight = 1.0 / fisher_metric
    
    # Trust Region 裁剪（几何 Huber-style，比 clamp 更平滑）
    trust_region_scale = getattr(self_distillation_config, "geodesic_trust_region", 5.0)
    manifold_weight = torch.where(
        raw_manifold_weight > trust_region_scale,
        trust_region_scale + torch.log(raw_manifold_weight - trust_region_scale + 1.0),
        raw_manifold_weight
    )
    
    # 应用流形权重
    per_token_loss = manifold_weight * per_token_loss
    
    # 记录指标
    metrics["actor/geodesic_mean_weight"] = manifold_weight.mean().item()
    metrics["actor/geodesic_max_weight"] = manifold_weight.max().item()
```

**理论解释：**

1. **Fisher 对角近似：** `F ≈ p(1-p)`
   - 当 `p → 0.5`（高熵/不确定）时，`F → 0.25`（最大值）
   - 当 `p → 0` 或 `p → 1`（低熵/确定）时，`F → 0`（最小值）

2. **Manifold Weight：** `w = 1/F`
   - 高熵 token：`w ≈ 4`（小权重，保持探索）
   - 低熵 token：`w → ∞`（大权重，强制反思）

3. **Trust Region 裁剪：**
   ```python
   w_clipped = {
       w,                      if w ≤ 5.0
       5.0 + log(w - 5.0 + 1), if w > 5.0
   }
   ```
   - 平滑缩放，不突然截断梯度方向
   - 保留微分几何的连续性

---

### 2. β 缩放因子补偿

**问题：** Manifold weight 会放大有效梯度（尤其低置信度 token）

**解决方案：**

```yaml
# 配置文件：verl/trainer/config/sdpo.yaml
geodesic_beta_scale: 0.5  # β 缩放因子
```

**效果：**
- 原始 DPO loss: `L = -logsigmoid(β · Δr)`
- Geodesic loss: `L = manifold_weight · [-logsigmoid(β · 0.5 · Δr)]`
- 相当于将 β 减半，补偿 manifold weight 的放大效应

---

## 📊 监控指标

### 核心指标（SwanLab）

| 指标 | 含义 | 预期趋势（A→B→C→D） |
|------|------|---------------------|
| `actor/entropy` | 策略熵（越高越好） | A: 稳定, B: 下降, C: 稳定, D: 稳定 |
| `actor/adv_std` | Advantage 标准差 | B: 趋近 0, C: 健康水平 |
| `actor/geodesic_mean_weight` | 平均流形权重 | C, D: 动态变化 |
| `actor/geodesic_max_weight` | 最大流形权重 | C, D: 反映极端 low-entropy token |
| `val/accuracy` | 验证集准确率 | 对比最终性能 |
| `actor/loss` | 训练损失 | 对比收敛速度 |

### 辅助诊断

```python
# 在训练日志中观察
- "Epistemic Collapse" 信号：entropy 持续下降 + adv_std 趋近 0
- "Geodesic 保护" 信号：entropy 稳定 + geodesic_max_weight 周期性飙升
- "Reward Hacking" 信号：loss 下降但 accuracy 不升
```

---

## 🚀 提交与运行

### 提交命令

```bash
# 设置环境变量
export OPENLM_TOKEN="YOUR_TOKEN"
export OSS_ACCESS_ID="YOUR_ID"
export OSS_ACCESS_KEY="YOUR_KEY"
export SWANLAB_API_KEY="YOUR_KEY"

# 提交实验（4 组）
bash nebula_scripts/submit_geodesic_vce_ablation.sh

# Dry-run 模式（只打印命令，不提交）
bash nebula_scripts/submit_geodesic_vce_ablation.sh --dry-run
```

### 任务命名规则

```
GVCE-{dataset}-{exp_name}
示例：
  GVCE-sciknoweval_biology-sdpo_baseline
  GVCE-sciknoweval_biology-vce_baseline
  GVCE-sciknoweval_biology-vce_plus_geodesic
  GVCE-sciknoweval_biology-sdpo_plus_geodesic
```

### 预期运行时间

- 数据集：sciknoweval/biology（约 1000 样本）
- Max steps：250
- 预估时间：约 6-8 小时/实验（H20 GPU）
- 总时间：约 24-32 小时（4 实验串行）

---

## 🔍 结果分析指南

### 场景 1: V_CE 确实导致熵崩溃

**信号：**
- 实验 B 的 `actor/entropy` 从 3.5 下降到 1.0 以下
- 实验 B 的 `actor/adv_std` 从 0.5 下降到 0.05 以下
- 实验 B 的 `val/accuracy`  plateau 或下降

**结论：**
- V_CE 的 token-level baseline 确实会导致 advantage 消失
- 需要引入 Geodesic 或其他熵保护机制

---

### 场景 2: Geodesic 成功免疫熵崩溃

**信号：**
- 实验 C 的 `actor/entropy` 稳定在 2.5-3.5
- 实验 C 的 `actor/geodesic_max_weight` 周期性飙升到 5.0（trust region 阈值）
- 实验 C 的 `val/accuracy` 持续上升

**结论：**
- Fisher 度量加权确实能强制模型反思低熵区域
- Trust Region 裁剪有效防止梯度爆炸
- Geodesic 约束在概率流形上保持了拓扑结构

---

### 场景 3: SDPO+Geodesic 最优

**信号：**
- 实验 D 的 `val/accuracy` 最高
- 实验 D 的 `actor/loss` 收敛最平滑
- 实验 D 的 `actor/entropy` 稳定在健康水平

**结论：**
- SDPO 的 KL 蒸馏损失比 V_CE 的 advantage 估计更稳定
- Geodesic 约束在 SDPO 上同样有效
- 推荐使用 **SDPO + Geodesic** 作为默认配置

---

## 📝 代码改动清单

### 修改的文件

| 文件 | 改动 | 行数 |
|------|------|------|
| `verl/trainer/ppo/core_algos.py` | 添加 Geodesic Fisher 度量加权逻辑 | +35 |
| `verl/trainer/config/sdpo.yaml` | 添加 `use_geodesic`, `geodesic_trust_region`, `geodesic_beta_scale` | +5 |
| `nebula_scripts/sdpo/geodesic_vce_ablation_parametric.sh` | 统一参数化脚本（支持 SDPO 和 V_CE） | 新增 |
| `nebula_scripts/submit_geodesic_vce_ablation.sh` | 4 组实验提交脚本 | 新增 |
| `docs/geodesic_vce_experiment.md` | 本文档 | 新增 |

### Git 分支

```bash
# 当前分支
git branch: geodesic-sdpo

# 基于
git log --oneline -1
# 输出: xxxxxxx feat: add Geodesic-VCE ablation experiment scripts (4-way comparison)
```

---

## 🎓 理论背景

### 1. 为什么 V_CE 会导致熵崩溃？

**数学推导：**

V_CE baseline 的计算：
```
V_CE[t] = E_{y~π_s}[log π_teacher(y)] = Σ_k π_s(k) · log π_T(k)
```

Advantage：
```
A[t] = log π_T(y_t) - V_CE[t]
     = log π_T(y_t) - Σ_k π_s(k) · log π_T(k)
```

当 student 分布集中（熵低）时：
- `π_s(y_t) → 1`，其他 `π_s(k) → 0`
- `V_CE[t] → log π_T(y_t)`
- `A[t] → 0`（advantage 消失）

**后果：**
- Policy gradient: `∇L = -A[t] · ∇log π_s(y_t)`
- 当 `A[t] → 0`，梯度消失
- 模型失去优化动力，熵进一步下降（正反馈）
- 最终导致 **Epistemic Collapse**

---

### 2. 为什么 Geodesic 能免疫熵崩溃？

**信息几何视角：**

概率分布构成一个统计流形（Statistical Manifold），其度量张量是 Fisher 信息矩阵：
```
F_ij(θ) = E_{x~π_θ}[∂_i log π_θ(x) · ∂_j log π_θ(x)]
```

Natural Gradient 更新：
```
Δθ = F^{-1} · ∇_θ L
```

**对角近似：**
```
F ≈ p(1-p)  （Categorical 分布的 Fisher 对角元）
```

**效果：**
- 当 `p → 0.5`（高熵）：`F → 0.25`，`F^{-1} → 4`（小更新）
- 当 `p → 0` 或 `p → 1`（低熵）：`F → 0`，`F^{-1} → ∞`（大更新）

**几何解释：**
- 在高熵区域（探索区），流形曲率小，更新步长小（保持探索）
- 在低熵区域（塌陷区），流形曲率大，更新步长大（强制逃离）
- **本质：** 沿测地线（Geodesic）流动，保持流形拓扑结构

---

### 3. Trust Region 裁剪的必要性

**问题：** 纯理论 Natural Gradient 在 `p → 0` 时会导致 `F^{-1} → ∞`，引发梯度爆炸。

**解决方案：** 几何 Huber-style 裁剪
```python
w_clipped = {
    w,                      if w ≤ τ
    τ + log(w - τ + 1),     if w > τ
}
```

**优势：**
1. **平滑性：** `w_clipped` 在 `w = τ` 处连续可微
2. **有界性：** 对数增长比线性增长慢，防止极端权重
3. **方向保持：** 不改变梯度方向向量，只缩放幅度

**对比硬截断（clamp）：**
```python
# ❌ Clamp：突然截断，改变梯度方向
w_clipped = min(w, τ)

# ✅ Trust Region：平滑缩放，保持方向
w_clipped = τ + log(w - τ + 1)  if w > τ
```

---

## 📚 参考文献

1. **Why Self-Distillation Degrades LLM Reasoning** (2025)
   - 发现 SDPO 导致 Epistemic Collapse
   - 提出不确定性 verbalization 缺失是根因

2. **Natural Gradient Descent** (Amari, 1998)
   - 信息几何视角的优化方法
   - Fisher 度量作为流形曲率

3. **Geodesic Flow on Statistical Manifolds** (Ay et al., 2017)
   - 统计流形上的测地线方程
   - 保持拓扑结构的优化路径

4. **Trust Region Policy Optimization** (Schulman et al., 2015)
   - 信任区域约束防止策略更新过大
   - 启发我们的 Trust Region 裁剪设计

---

## 🤝 贡献者

- **算法设计:** Jimmy (loujieming.ljm)
- **代码实现:** Jimmy
- **实验提交:** Nebula 集群（lazada_llm_ad_h20 队列）
- **文档编写:** Jimmy + Qoder

---

## 📅 时间线

- **2026-05-19:** 创建 `geodesic-sdpo` 分支，实现 Geodesic 约束
- **2026-05-19:** 设计 4 组消融实验，提交到 Nebula
- **2026-05-20:** 预期实验完成，开始分析结果
- **2026-05-21:** 根据结果调整超参（trust_region, beta_scale）

---

**祝实验顺利！🚀**
