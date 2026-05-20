# Geodesic-VCE 融合实验：理论动机与实现文档

**分支**: `geodesic-sdpo`  
**创建时间**: 2026-05-19  
**实验状态**: 🚀 待提交（Nebula 环境配置修复中）  
**Seed**: 42（严格控制，保证可复现性）

---

## 📋 TL;DR

本文档记录了 **Geodesic SDPO**（测地线偏好优化）与 **V_CE**（Cross-Entropy Baseline）的融合实验设计与实现。核心科学问题是：

1. **V_CE 是否真的导致 entropy collapse？**（复现历史问题）
2. **Geodesic 约束能否免疫 V_CE 的熵崩溃？**（理论验证）
3. **SDPO+Geodesic vs VCE+Geodesic 哪个更强？**（算法对比）

实验采用 **4 组消融设计**，在 `sciknoweval/biology` 数据集上严格统一 seed=42 和所有超参，唯一变量是算法设计。

---

## 🎯 一、理论动机

### 1.1 V_CE 的熵崩溃问题

在 `self-teacher-advantage` 分支的研究中，我们发现 V_CE（Cross-Entropy Baseline）会导致严重的 **entropy collapse**：

```python
# V_CE 的核心公式
V_CE[i,t] = Σ_k π_student(k) · log π_teacher(k)  # 期望值
A_raw[i,t] = Q[i,t] - V_CE[i,t]                   # advantage
```

**崩溃机制**：
- 当 student 分布集中（低熵）时，V_CE ≈ log π_teacher(y_t)
- 导致 A_raw → 0，**advantage 消失**
- 模型失去探索动力，进一步固化低熵分布 → **正反馈循环**

### 1.2 Geodesic SDPO 的内生熵保护

Geodesic SDPO 基于 **信息几何**（Information Geometry）理论，在概率流形上做 Natural Gradient 更新：

```python
# Fisher 对角近似
F ≈ p(1-p)  # p = π_student(y_t)

# 流形权重（Trust Region 平滑裁剪）
manifold_weight = torch.where(
    1/F > trust_region,
    trust_region + torch.log(1/F - trust_region + 1.0),
    1/F
)

# 外部缩放 loss
loss = manifold_weight × base_loss
```

**保护机制**：
| Student 置信度 $p$ | Fisher $F=p(1-p)$ | manifold_weight $1/F$ | 效果 |
|-------------------|-------------------|----------------------|------|
| $p \to 1$（过度自信） | $F \to 0$ | $\to \infty$（截断后很大） | **强烈惩罚** |
| $p \to 0$（完全不确定） | $F \to 0$ | $\to \infty$ | **鼓励探索** |
| $p = 0.5$（最大熵） | $F = 0.25$ | $1/F = 4$ | 正常更新 |

### 1.3 融合假设

**核心假说**：Geodesic 约束可以在 **loss 层面** 提供熵保护，而 V_CE 在 **advantage 层面** 做 variance reduction，二者正交互补。

---

## 🔬 二、实验设计

### 2.1 实验矩阵（4 组）

| 实验编号 | 名称 | loss_mode | use_vce | use_geodesic | 核心验证目标 |
|---------|------|-----------|---------|--------------|-------------|
| **A** | SDPO baseline | `sdpo` | False | False | 对照组（标准 SDPO） |
| **B** | V_CE baseline | `self_teacher` | True | False | 复现熵崩溃 |
| **C** | V_CE + Geodesic | `self_teacher` | True | True | 验证 Geodesic 免疫熵崩溃 |
| **D** | SDPO + Geodesic | `sdpo` | False | True | 验证 Geodesic 在 SDPO 上的效果 |

### 2.2 控制变量

所有实验严格统一以下超参：

```yaml
# 训练参数
seed: 42
lr: 1e-5
train_batch_size: 32
rollout_n: 8
model_name: Qwen3-8B
total_training_steps: 250

# SDPO 参数
alpha: 0.5  # Jensen-Shannon divergence
distill_topk: 100

# V_CE 参数
clip_value: 3.0  # advantage clip threshold
adv_std_floor: 0.0

# Geodesic 参数
geodesic_trust_region: 5.0
geodesic_beta_scale: 0.5
```

### 2.3 预期结果

| 对比 | 预期现象 | 科学意义 |
|------|---------|---------|
| **B vs A** | B 的 entropy 快速下降，accuracy 初期上升后崩塌 | 复现 V_CE 熵崩溃 |
| **C vs B** | C 的 entropy 稳定，accuracy 持续上升 | Geodesic 免疫熵崩溃 ✓ |
| **D vs C** | 对比最终 accuracy 和收敛速度 | 哪种范式更强？ |

---

## 🛠️ 三、代码实现

### 3.1 核心修改文件

#### 1. `verl/trainer/ppo/core_algos.py`（Geodesic 约束实现）

**修改位置**: `compute_self_distillation_loss` 函数（~L1179）

```python
# ===== Geodesic SDPO: Fisher 度量加权 (流形约束) =====
use_geodesic = getattr(self_distillation_config, "use_geodesic", False)
if use_geodesic:
    # 计算 Fisher 对角近似
    if student_all_log_probs is not None:
        # Full logit 模式：在 vocab 维度计算 Fisher
        p_student = torch.exp(student_all_log_probs.detach())
        fisher_metric = (p_student * (1.0 - p_student)).sum(-1)
    else:
        # Token-level 模式
        p_student = torch.exp(student_log_probs.detach())
        fisher_metric = p_student * (1.0 - p_student)
    
    fisher_metric = fisher_metric + 1e-5
    
    # Trust Region / Geometric Huber-style clipping
    trust_region_scale = getattr(self_distillation_config, "geodesic_trust_region", 5.0)
    manifold_weight = torch.where(
        1.0 / fisher_metric > trust_region_scale,
        trust_region_scale + torch.log(1.0 / fisher_metric - trust_region_scale + 1.0),
        1.0 / fisher_metric
    )
    
    # 外部缩放
    per_token_loss = per_token_loss * manifold_weight
    
    # 记录指标
    metrics["actor/geodesic_mean_weight"] = manifold_weight.mean().detach().item()
    metrics["actor/geodesic_max_weight"] = manifold_weight.max().detach().item()
```

**关键设计决策**：
- ✅ 使用 `detach()` 阻断计算图（不优化 Fisher 本身）
- ✅ Trust Region 平滑裁剪（而非硬 clamp，保留梯度方向）
- ✅ 外部缩放（不改变 BT 模型最优点）

#### 2. `verl/trainer/config/sdpo.yaml`（配置开关）

```yaml
actor_rollout_ref:
  actor:
    self_distillation:
      use_geodesic: False  # 默认关闭
      geodesic_trust_region: 5.0
      geodesic_beta_scale: 0.5
```

#### 3. `nebula_scripts/sdpo/geodesic_vce_ablation_parametric.sh`（统一参数化脚本）

支持两种 loss_mode 的动态切换：

```bash
if [ "$LOSS_MODE" = "sdpo" ]; then
    # SDPO 模式：self_distillation loss
    HYDRA_ARGS+=(
        actor_rollout_ref.actor.policy_loss.loss_mode=sdpo
        actor_rollout_ref.actor.self_distillation.use_geodesic=${USE_GEODESIC}
        ...
    )
elif [ "$LOSS_MODE" = "self_teacher" ]; then
    # Self-Teacher 模式：advantage + policy gradient
    HYDRA_ARGS+=(
        actor_rollout_ref.actor.policy_loss.loss_mode=vanilla
        algorithm.adv_estimator=self_teacher
        algorithm.use_vce=${USE_VCE}
        ...
    )
fi
```

#### 4. `nebula_scripts/submit_geodesic_vce_ablation.sh`（标准化提交脚本）

遵循 `submit_baseline_sweep.sh` 的 nebulactl 标准格式：

```bash
nebulactl run mdl \
    --force \
    --engine=xdl \
    --queue=$QUEUE \
    --entry=nebula_scripts/entry.py \
    --user_params="--script_path=${SCRIPT_PATH} --world_size=${WORLD_SIZE} --job_name=${JOB_NAME} ${USER_PARAMS}" \
    --worker_count=$WORLD_SIZE \
    --file.cluster_file=$CLUSTER_FILE \
    --job_name=$JOB_NAME \
    --env=OPENLM_TOKEN=$OPENLM_TOKEN \
    --env=OSS_ACCESS_ID=$OSS_ACCESS_ID \
    ...
```

### 3.2 Git 提交记录

| Commit | 说明 |
|--------|------|
| `53bd131` | feat: implement Geodesic SDPO with Fisher metric weighting |
| `1e98dea` | refactor: replace clamp with trust region clipping for Geodesic SDPO |
| `610fd20` | feat: add Geodesic-VCE ablation experiment scripts (4-way comparison) |
| `77ac99c` | fix: standardize nebulactl submission format |

---

## 📊 四、监控与诊断

### 4.1 SwanLab 关键指标

| 指标名称 | 含义 | 预期行为 |
|---------|------|---------|
| `actor/entropy` | Student 策略熵值 | C/D 应显著高于 B |
| `actor/geodesic_mean_weight` | 平均流形权重 | 应稳定在 2-8 之间 |
| `actor/geodesic_max_weight` | 最大流形权重 | 验证 trust_region 截断 |
| `val/accuracy` | 验证集准确率 | C/D 应持续上升，B 后期崩塌 |
| `actor/kl_divergence` | Student-Teacher KL | 监控分布匹配程度 |

### 4.2 异常诊断清单

| 现象 | 可能原因 | 解决方案 |
|------|---------|---------|
| `geodesic_mean_weight` → ∞ | trust_region 设置过大 | 降低到 3.0 |
| entropy 仍快速下降 | beta_scale 过小 | 增加到 0.7-1.0 |
| accuracy 不收敛 | lr 过大或 mini_batch 过小 | 检查配置对齐 |
| NaN loss | Fisher 计算溢出 | 检查 `detach()` 是否正确 |

---

## 🔮 五、后续扩展

### 5.1 短期（本周）

- [ ] 完成 4 组实验提交并监控训练动态
- [ ] 对比 entropy 曲线验证 Geodesic 保护效果
- [ ] 分析 accuracy 收敛速度差异

### 5.2 中期（下周）

- [ ] 在 `sciknoweval/chemistry` 上复现（跨领域验证）
- [ ] 增加 `geodesic_trust_region` sweep（3.0/5.0/8.0）
- [ ] 尝试 Geodesic-VCE 融合（V_CE advantage + Geodesic loss weighting）

### 5.3 长期（论文方向）

- [ ] 理论证明：Geodesic 约束下的熵下界
- [ ] 扩展到 DPO-TGS（On-Policy DPO with Teacher-Guided Sampling）
- [ ] 与 Sinkhorn Credit Assignment 结合（最优传输视角）

---

## 📚 六、理论背景

### 6.1 Natural Gradient 与 Fisher 信息矩阵

Natural Gradient 的核心思想是在 **统计流形**（Statistical Manifold）上做优化，而非欧氏参数空间：

$$\tilde{\nabla}_\theta L = F^{-1} \nabla_\theta L$$

其中 $F$ 是 Fisher 信息矩阵。对于 Categorical 分布，对角近似为：

$$F_{ii} \approx p_i(1-p_i)$$

### 6.2 Trust Region 与几何 Huber 裁剪

标准的 Natural Gradient 在 $p \to 0$ 或 $p \to 1$ 时会导致梯度爆炸。我们提出 **几何 Huber 裁剪**：

$$w(p) = \begin{cases} 
\frac{1}{F(p)} & \text{if } \frac{1}{F(p)} \leq \tau \\
\tau + \log\left(\frac{1}{F(p)} - \tau + 1\right) & \text{otherwise}
\end{cases}$$

这比硬截断（clamp）更平滑，保留了梯度方向向量。

### 6.3 与 PPO Clip 的关系

PPO 的 clip 机制本质上也是一种 trust region：

$$L^{CLIP} = \min(r_t A_t, \text{clip}(r_t, 1-\epsilon, 1+\epsilon) A_t)$$

Geodesic 的 trust region 是在 **概率流形** 上，而 PPO clip 是在 **重要性采样比率** 上。二者互补而非替代。

---

## 📝 七、实验日志

### 2026-05-19

- ✅ 完成 Geodesic SDPO 核心实现（core_algos.py）
- ✅ 添加配置开关（sdpo.yaml）
- ✅ 创建统一参数化脚本（geodesic_vce_ablation_parametric.sh）
- ✅ 创建标准化提交脚本（submit_geodesic_vce_ablation.sh）
- ⚠️ 修复 nebulactl 格式问题（缺少 worker_count 参数）
- 🚀 准备提交 4 组实验到 Nebula

---

## 🔗 八、相关文件

| 文件路径 | 说明 |
|---------|------|
| `verl/trainer/ppo/core_algos.py` | Geodesic 约束核心实现 |
| `verl/trainer/config/sdpo.yaml` | 配置开关与超参 |
| `nebula_scripts/sdpo/geodesic_vce_ablation_parametric.sh` | 统一参数化训练脚本 |
| `nebula_scripts/submit_geodesic_vce_ablation.sh` | Nebula 提交脚本 |
| `research/01_evolution.md` | OPD 演进文献综述 |

---

**文档维护者**: Jimmy  
**最后更新**: 2026-05-19  
**下次审查**: 实验结果出来后更新预期 vs 实际对比
