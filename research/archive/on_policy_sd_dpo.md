这份 Proposal 旨在将 **Direct Preference Optimization (DPO)** 的对比学习机制引入到 **On-policy（同策略）** 训练闭环中，并结合 **Self-Teacher（自教师）** 信号，解决推理任务中信度分配（Credit Assignment）模糊和思维退化的问题。

---

# 提案：基于自教师引导的 Token 级同策略 DPO (Token-level On-policy DPO)

## 一、 研究背景与动机 (Motivation)

在推理任务（RLVR）的强化学习中，现有的方案存在以下局限：

1. **GRPO/PPO**：依赖序列级奖励，导致“连坐效应”，即正确轨迹中的平庸 Token 被奖励，错误轨迹中的正确步骤被惩罚。
2. **SDPO (Self-Distilled PO)**：本质是行为克隆（Behavioral Cloning），只教模型“该写什么”，不教“不该写什么”，容易导致模型多样性丧失和思维退化。
3. **Offline DPO**：依赖静态数据集，无法处理模型在训练过程中新产生的逻辑漏洞（Distribution Shift）。

**核心创新点**：提出 **On-policy DPO**。模型在采样过程中，利用带有特权信息的 **Self-Teacher** 实时识别“逻辑分歧点”，构建“即时偏好对”（Token-level Pair），通过对比学习同时实现“拉近正确”与“推离错误”。

---

## 二、 核心算法框架

### 2.1 闭环流程 (The Loop)

1. **On-policy Rollout**：当前策略 $\pi_\theta$ 采样生成回答 $\hat{y}$。
2. **环境验证**：获取序列奖励 $R_{seq}$（如代码通过率或数学正确性）。
3. **Self-Teacher 介入 (仅针对错误轨迹 $R_{seq}=0$)**：
* 将错误轨迹喂给 Teacher（当前模型 + 正确答案作为 Context）。
* **定位分歧点 (Divergence Point)**：找到学生概率 $\pi_\theta$ 高但教师概率 $\pi_{teacher}$ 低的 Token 位置。


4. **构建 Token 偏好对**：
* **Chosen Token ($a^+$)**：教师在该位置预测的 Top-1 Token。
* **Rejected Token ($a^-$)**：学生实际采样的那个导致后续崩盘的 Token。


5. **DPO 更新**：执行对比损失优化。

### 2.2 损失函数 (Objective Function)

在确定的分歧点 $t$，应用 Token 级 DPO 损失：


$$\mathcal{L} = -\mathbb{E} \left[ \log \sigma \left( \beta \log \frac{\pi_\theta(a^+ | x, y_{<t})}{\pi_{ref}(a^+ | x, y_{<t})} - \beta \log \frac{\pi_\theta(a^- | x, y_{<t})}{\pi_{ref}(a^- | x, y_{<t})} \right) \right]$$

---

## 三、 关键实现细节

### 3.1 动态分歧门控 (Dynamic Divergence Gating)

为了避免对每一个 Token 进行无效的 DPO 更新（如标点、简单连词），我们引入门控机制：

* **触发条件**：只有当 $D_{KL}(\pi_{teacher} || \pi_\theta) > \tau$ 时，才激活 DPO 更新。
* **意义**：这确保了模型只在“真正的逻辑误区”进行高强度学习，保护了非关键 Token 的信息熵。

### 3.2 防止信息泄漏 (Anti-Leakage)

参考 RLSD 论文，Teacher 因为看到了正确答案，其分布可能包含“答案暗示”。

* **策略**：对 Teacher 的输出进行温度平滑 (Temperature Smoothing)，或在构建 $a^+$ 时，强制要求 $a^+$ 必须是能推导出逻辑步骤的 Token，而非直接泄露结果。

### 3.3 负样本挖掘 (Negative Mining)

在 On-policy 采样中，如果同一 Prompt 下多个轨迹都在某处出错，我们会将这些 $a^-$ 聚类。

* **强化惩罚**：对于高频出现的错误 Token，加大 DPO 中的 $\beta$ 权重，实现“精准排雷”。

---

## 四、 方案对比优势

| 维度 | GRPO | SDPO | **On-policy DPO (本方案)** |
| --- | --- | --- | --- |
| **信号粒度** | 序列级 (粗糙) | Token 级 (精细) | **Token 级 (精细且具对比性)** |
| **学习本质** | 试错法 | 模仿法 | **复盘纠错法** |
| **多样性保持** | 好 | 差 (容易收敛至单一点) | **好 (通过对比而非单纯模仿)** |
| **纠错能力** | 慢 (依赖稀疏奖励) | 中 (只看正确路径) | **极强 (显式推离错误动作)** |

---

## 五、 预期风险与应对 (Risk Management)

1. **采样成本**：On-policy DPO 需要 Teacher 频繁进行 Forward。
* *应对*：采用 **Selective Forward**，仅对 $R=0$ 的轨迹和检测到的分歧点进行 Teacher 计算。


2. **梯度不稳定**：Token 级的对比可能导致 Log-ratio 剧烈波动。
* *应对*：引入你之前提出的 **V-EMA Baseline** 对 Advantage 进行平滑处理，或使用 Clipping 限制梯度大小。


3. **参考模型漂移**：
* *应对*：定期同步 $\pi_{ref}$，或使用 EMA 方式缓慢更新参考模型，确保对比的基准线是稳健的。



---

## 六、 实验计划 (Timeline)

1. **阶段 1 (Baseline 搭建)**：在简单的数学任务 (GSM8K) 上验证 Token 对构建的准确性。
2. **阶段 2 (消融实验)**：对比“仅拉近正样本 (SDPO)”与“对比学习 (DPO)”的收敛速度差异。
3. **阶段 3 (性能飞跃)**：在 AIME、Codeforces 等高难度任务上测试模型对逻辑分歧点的处理能力。

---

### 总结

本 Proposal 的核心在于将 **DPO 的“辨析”能力** 引入到 **RL 的“探索”过程** 中。通过 Teacher 指导“什么是好”，通过模型自身的失败经验定义“什么是不好”，从而在动态的训练流中实现比传统方法更精准、更稳健的逻辑对齐。