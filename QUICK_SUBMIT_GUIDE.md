# 算法对比实验 - 快速提交指南

## ⚠️ 重要说明

提交实验需要使用 `nebulactl` 命令，该命令只在 Nebula 集群环境中可用。

## 🚀 提交步骤

### 方式 1：SSH 到开发机提交

```bash
# 1. SSH 到开发机
ssh <your-dev-machine>

# 2. 进入项目目录
cd /path/to/SDPO

# 3. 激活环境（如果需要）
source nebula_env/bin/activate

# 4. 验证 nebulactl 可用
which nebulactl

# 5. 提交实验（参考下方各算法命令）
```

### 方式 2：直接在集群环境执行

```bash
# 确保在已配置 nebulactl 的环境中
nebulactl --version  # 验证命令可用
```

## 📋 各算法提交命令

### 1️⃣ GRPO（2 个实验）

```bash
# 先验证
bash nebula_scripts/submit_grpo_comparison.sh --dry-run

# 实际提交
bash nebula_scripts/submit_grpo_comparison.sh
```

**实验**：
- `grpo_offpolicy_mbs8`: off-policy, mini_batch=8
- `grpo_onpolicy_mbs32`: on-policy, mini_batch=32

### 2️⃣ SDPO（1 个实验）

```bash
bash nebula_scripts/submit_sdpo_comparison.sh --dry-run
bash nebula_scripts/submit_sdpo_comparison.sh
```

### 3️⃣ Self-Teacher（4 个实验）

```bash
bash nebula_scripts/submit_self_teacher_comparison.sh --dry-run
bash nebula_scripts/submit_self_teacher_comparison.sh
```

## 🔍 验证提交是否成功

```bash
# 查看任务状态
nebulactl list

# 查看日志
nebulactl logs <job_id>

# 查看 SwanLab
# https://swanlab.cn/@oh-my-team/Algorithm-Comparison-v1
```

## 📝 环境变量检查

提交前确保以下环境变量已设置：

```bash
echo $OPENLM_TOKEN      # 应显示 token
echo $OSS_ACCESS_ID     # 应显示 access id
echo $OSS_ACCESS_KEY    # 应显示 access key
```

如果未设置，需要先在开发机上配置：

```bash
export OPENLM_TOKEN="your-token"
export OSS_ACCESS_ID="your-access-id"
export OSS_ACCESS_KEY="your-access-key"
```

## ⚡ 快速一键提交所有实验

```bash
# 创建一个批量提交脚本
cat > submit_all.sh << 'EOF'
#!/bin/bash
echo "=== 开始提交所有算法对比实验 ==="

echo "1/5 提交 GRPO..."
bash nebula_scripts/submit_grpo_comparison.sh
sleep 5

echo "2/5 提交 SDPO..."
bash nebula_scripts/submit_sdpo_comparison.sh
sleep 5

echo "3/5 提交 FIPO..."
bash nebula_scripts/submit_fipo_comparison.sh
sleep 5

echo "4/5 提交 DAPO..."
bash nebula_scripts/submit_dapo_comparison.sh
sleep 5

echo "5/5 提交 Self-Teacher..."
bash nebula_scripts/submit_self_teacher_comparison.sh

echo "=== 所有实验提交完成 ==="
EOF

chmod +x submit_all.sh
bash submit_all.sh
```

## 🐛 常见问题

### Q: `nebulactl: command not found`
A: 需要在 Nebula 集群环境或已安装 nebulactl 的开发机上执行

### Q: `OPENLM_TOKEN not set`
A: 需要先 export 环境变量

### Q: 如何查看任务进度？
A: 
```bash
nebulactl list | grep grpo
nebulactl logs <job_id> | tail -100
```

## 📊 实验总计

| 算法 | 实验数 | 提交脚本 |
|------|--------|---------|
| GRPO | 2 | submit_grpo_comparison.sh |
| SDPO | 1 | submit_sdpo_comparison.sh |
| FIPO | 1 | submit_fipo_comparison.sh |
| DAPO | 1 | submit_dapo_comparison.sh |
| Self-Teacher | 4 | submit_self_teacher_comparison.sh |
| **总计** | **9** | - |
