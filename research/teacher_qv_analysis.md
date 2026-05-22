# Teacher-QV Framework 实验分析报告

## 概述

本报告分析了Teacher-QV框架不同baseline类型的实验结果，包括：
- `qv_student`: SDPO sampled-token baseline
- `qv_ce`: True V_CE baseline (E_{y~p_s}[log p_t(y)])
- `qv_ce_geodesic`: V_CE + Geodesic constraint
- `qv_student_geodesic`: SDPO + Geodesic constraint
- `qv_group_mean`: Batch-mean baseline
- `qv_group_hier`: Hierarchical (two-level) normalization baseline
- `qv_group_hier_no_std`: Hierarchical baseline without std normalization

## 实验设计

### 基线类型对比
1. **student**: V_t = log p_s(y_t) (SDPO原生baseline)
2. **ce**: V_t = E_{y~p_s}[log p_t(y)] (True cross-entropy baseline)
3. **group_mean**: V = mean(log p_teacher) across batch
4. **group_hier**: Two-level normalization (within-seq + across-seq)

### 梯度模式
- **sampled**: Monte Carlo estimation on single sampled token
- **full_logit**: Vocabulary-summed policy gradient (recommended to prevent entropy collapse)

### Geodesic约束
- **use_geodesic=True/False**: 是否启用Fisher metric约束的自然梯度

## 主要发现

### 1. Entropy Collapse 问题
- **sampled模式**: 所有baselines都出现entropy collapse
  - 原因: Reverse KL (mode-seeking)性质 + 采样估计的高方差
  - 结果: 模型快速收敛到低熵状态

- **full_logit模式**: 有效防止entropy collapse
  - 原因: Vocabulary-summed PG提供更好的探索压力
  - 结果: 维持健康的entropy水平

### 2. V_CE Baseline 行为
- **预期**: V_CE作为零均值baseline，advantage应近似为0
- **观察**: 在sampled模式下出现entropy collapse
- **验证**: V_CE = E_{y~p_s}[log p_t(y)]确实会导致collapse

### 3. Geodesic 约束效果
- **V_CE + Geodesic**: 成功缓解entropy collapse
- **SDPO + Geodesic**: 提供稳健性改进
- **机制**: Fisher metric约束防止过度更新

### 4. Group Baselines 性能
- **group_mean**: 全局批次均值作为baseline
  - 优点: 简单，提供批次级去偏
  - 缺点: 不考虑序列内结构
  
- **group_hier**: 两层标准化
  - 优点: 更好的数值稳定性，考虑序列内变异
  - 结果: 相比group_mean更稳定的训练动态

## 关键指标分析

### 主要监控指标
1. `actor/entropy`: 衡量策略多样性
2. `actor/teacher_qv_adv_mean`: advantage均值，理想为0
3. `actor/teacher_qv_adv_std`: advantage方差
4. `actor/pg_loss`: 政策梯度损失（应为实数值，非NaN）
5. `val-core/sciknoweval/acc/mean@16`: 终端准确率

### Advantage范围分析
对于`qv_group_hier_no_std`等baseline类型，advantage的范围通常较窄（如-0.875到0.875）是因为：
- Advantage A_t = Q_t - V_t，其中Q_t = log p_teacher(y_t)，V_t是baseline
- 对于group_hier_no_std：A_t = (log p_teacher - per_seq_mean(log p_teacher)) - per_group_mean(per_seq_mean(log p_teacher))
- 这是双重中心化的结果，不是原始的log_teacher_prob值
- 相比之下，原始log_teacher_prob可能有更大的绝对值范围（如-10到0）
- 通过减去baseline，advantage被校准到接近0的小范围内

### Teacher Log Probability vs Verified Reward
需要澄清一个重要概念混淆：
- Teacher log probability (log p_teacher(y_t)) 是teacher模型对生成token的对数概率值，通常是负值（如-2到-10）
- Verified reward通常指经过验证的真实奖励值，范围是0~1之间的浮点数
- 在Teacher-QV框架中，我们使用log p_teacher(y_t)作为Q值，**而不是直接的verified reward**
- 这意味着我们实际是在优化与teacher模型预测的一致性，而不是直接优化verified reward
- Advantage A_t = log p_teacher(y_t) - V_t 基于teacher对数概率的差值，而非verified reward的差值
- 由于Q和V都是基于类似的负值log概率，A = Q - V的结果可能在0附近分布（正负皆有可能）

### 潜在的实现问题
从你的观察（advantage范围在-0.875到0.875，而非总是负数）可以看出，这里存在概念混淆：
- 如果Q_t = log p_teacher(y_t)（负数）和V_t（也是基于类似log概率）都为负数
- 则A_t = Q_t - V_t = (负数) - (负数)，可能为正也可能为负
- 这实际上衡量的是"teacher对token y_t的偏好程度相对于baseline的偏离"，而非传统意义上的reward信号
- 这可能与我们想要优化的verified reward（0~1）不是直接对应关系

### NaN 问题解决
- **根本原因**: `agg_loss`函数中除零导致0/0 NaN
- **解决方案**: `_safe_divisor`函数确保除数≥1
- **影响**: 修复前所有baselines的loss都为NaN

## 实验结论

### 1. 最佳配置
- **推荐**: `full_logit` gradient mode + `geodesic=True`
- 理由: 防止entropy collapse + 提供稳健性

### 2. Baseline选择
- **sampled模式**: `qv_student` (original SDPO)表现最佳
- **full_logit模式**: `qv_ce` + `geodesic`组合最优
- **group baselines**: 适合需要更强正则化的场景

### 3. V_CE假说验证
- ✅ **确认**: V_CE baseline确实会导致entropy collapse
- ✅ **验证**: Geodesic约束能够缓解collapse问题
- ❌ **否定**: V_CE并非理想的zero-mean baseline在实践中

## 后续方向

1. **算法改进**: 完善Geodesic约束的自适应参数
2. **Baseline优化**: 探索更robust的group-level baselines
3. **应用扩展**: 在更多数据集上验证泛化性
4. **理论分析**: 深入理解不同baseline的理论性质