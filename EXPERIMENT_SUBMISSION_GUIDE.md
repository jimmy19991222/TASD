# 算法对比实验提交指南

## 📋 实验概览

所有实验使用以下统一配置：
- **数据集**: sciknoweval/biology
- **模型**: Qwen3-8B
- **学习率**: 1e-5
- **Rollout N**: 8
- **SwanLab 组**: Algorithm-Comparison-v1

## 🚀 实验提交顺序

### 1️⃣ GRPO Baseline（2 个实验）

```bash
# 验证
bash nebula_scripts/submit_grpo_comparison.sh --dry-run

# 提交（需要在 Nebula 集群环境中执行）
bash nebula_scripts/submit_grpo_comparison.sh
```

**实验列表**：
- `grpo_offpolicy_mbs8`: off-policy, mini_batch=8, 4步更新
- `grpo_onpolicy_mbs32`: on-policy, mini_batch=32, 1步更新（与 SDPO 公平对比）

### 2️⃣ SDPO Baseline（1 个实验）

```bash
# 验证
bash nebula_scripts/submit_sdpo_comparison.sh --dry-run

# 提交
bash nebula_scripts/submit_sdpo_comparison.sh
```

**实验列表**：
- `sdpo_js_alpha0.5`: on-policy, alpha=0.5 (Jensen-Shannon divergence)

### 3️⃣ FIPO Baseline（1 个实验）

```bash
# 验证
bash nebula_scripts/submit_fipo_comparison.sh --dry-run

# 提交
bash nebula_scripts/submit_fipo_comparison.sh
```

**实验列表**：
- `fipo_offpolicy_mbs8`: off-policy, mini_batch=8（与 GRPO 共享参数）

### 4️⃣ DAPO Baseline（1 个实验）

```bash
# 验证
bash nebula_scripts/submit_dapo_comparison.sh --dry-run

# 提交
bash nebula_scripts/submit_dapo_comparison.sh
```

**实验列表**：
- `dapo_offpolicy_mbs8`: off-policy, mini_batch=8, clip-higher, entropy=0.001

### 5️⃣ Self-Teacher Advantage beta 消融（4 个实验）

```bash
# 验证
bash nebula_scripts/submit_self_teacher_comparison.sh --dry-run

# 提交
bash nebula_scripts/submit_self_teacher_comparison.sh
```

**实验列表**：
- `self_teacher_beta1.0_Vce_only`: beta=1.0（纯 V_CE，横向 baseline）
- `self_teacher_beta0.7_recommended`: beta=0.7（推荐配置，V_CE 主导）
- `self_teacher_beta0.5_equal`: beta=0.5（等权融合）
- `self_teacher_beta0.0_Vema_only`: beta=0.0（纯 V_EMA，纵向 baseline）

## 📊 实验总计

| 算法 | 实验数 | 更新策略 | mini_batch_size |
|------|--------|---------|-----------------|
| GRPO (off-policy) | 1 | off-policy 多步 | 8 |
| GRPO (on-policy) | 1 | on-policy 单步 | 32 |
| SDPO | 1 | on-policy 单步 | 32 |
| FIPO | 1 | off-policy 多步 | 8 |
| DAPO | 1 | off-policy 多步 | 8 |
| Self-Teacher | 4 | on-policy 单步 | 32 |
| **总计** | **9** | - | - |

## ⚠️ 注意事项

1. **运行环境**: 这些脚本需要在 Nebula 集群环境中执行（需要 `nebula` 命令）
2. **环境变量**: 确保以下环境变量已设置：
   - `OPENLM_TOKEN`
   - `OSS_ACCESS_ID`
   - `OSS_ACCESS_KEY`
3. **分支**: 所有实验在 `self-teacher-advantage` 分支上
4. **监控**: 提交后可通过以下命令监控：
   ```bash
   nebula logs <task_id>
   nebula status <task_id>
   ```
5. **SwanLab**: 实验结果查看地址：
   https://swanlab.cn/@oh-my-team/Algorithm-Comparison-v1

## 🔍 关键参数对照

### off-policy 算法（GRPO/FIPO/DAPO）
- `train_batch_size=32`
- `mini_batch_size=8`
- 每个 batch 进行 **4 步梯度更新**

### on-policy 算法（SDPO/Self-Teacher）
- `train_batch_size=32`
- `mini_batch_size=32`
- 每个 batch 进行 **1 步梯度更新**

### SDPO 特殊参数
- `alpha=0.5`: Jensen-Shannon divergence（论文 Table 3）
- `dont_reprompt_on_self_success=True`
- `distillation_topk=100`

### Self-Teacher 特殊参数
- `beta`: V_CE vs V_EMA 融合系数（消融实验变量）
- `ema_alpha=0.9`: V_EMA 衰减系数
- `clip_value=5.0`: advantage clipping 阈值

## 📝 提交日志

建议在提交时记录以下信息：

```bash
# 示例：提交 SDPO 实验
echo "$(date): Submitting SDPO baseline experiment" >> experiment_log.txt
bash nebula_scripts/submit_sdpo_comparison.sh 2>&1 | tee -a experiment_log.txt
```
