# 实验提交指南（统一版）

> 合并了旧版 quick_submit_guide.md + experiment_submission_guide.md，并加入 TCCA 系列。
> **提交前**：必须 SSH 到 Nebula 开发机或 notebook（本地 Mac 没有 `nebulactl`）。

---

## 0. 通用前置

```bash
cd /path/to/SDPO              # 或开发机上的 /home/<user>/TASD

# 激活环境 (有 nebulactl)
source sdpo_env/bin/activate   # 或 nebula_env/bin/activate

# 加载凭证 (OPENLM_TOKEN, OSS_ACCESS_*, SWANLAB_API_KEY 等)
set -a && source .env && set +a

# 验证
which nebulactl   # → /path/to/sdpo_env/bin/nebulactl
echo "OPENLM=$OPENLM_TOKEN" | head -c 30
```

**所有 sweep 脚本必先 `--dry-run` 一次** 核对 JOB_NAME、超参再正式提交。

---

## 1. TCCA / 论文主线（**当前重点**）

### Nebula sweep（默认 GRPO + RLSD 两个 base × ΔR=True）

```bash
bash nebula_scripts/submit_intervention_credit_sweep.sh --dry-run    # 先 dry-run
bash nebula_scripts/submit_intervention_credit_sweep.sh              # 正式提交
```

默认产 2 个 job：
- `TGDI-v3-grpo-dR-...` (GRPO + TCCA ΔR causal layer)
- `TGDI-v3-rlsd-dR-...` (RLSD + TCCA ΔR causal layer，论文 headline)

### 关 ΔR 跑纯 baseline（填论文对照空缺）

```bash
IC_ENABLE_INTERVENTION_LIST="False" \
  bash nebula_scripts/submit_intervention_credit_sweep.sh
```

### 单跑 RLSD baseline 完整 250-step

```bash
IC_BASE_ESTIMATORS="rlsd" IC_ENABLE_INTERVENTION_LIST="False" \
  bash nebula_scripts/submit_intervention_credit_sweep.sh
```

### Local 4-GPU smoke

```bash
# Phase 1 (enable_intervention=False, 验证 plumbing, ~15 min)
./run_notebook_intervention_credit.sh smoke

# Phase 2/TCCA (enable_intervention=True, 真路径, ~20 min)
IC_ENABLE_INTERVENTION=True IC_BASE_ESTIMATOR=grpo \
  ./run_notebook_intervention_credit.sh smoke

# t* 策略消融 (3 策略串行 ~45 min)
./run_notebook_intervention_credit.sh tstar

# 完整 250 step (本地等价 Nebula)
./run_notebook_intervention_credit.sh full
```

### 可调 env vars (Nebula sweep)

| env var | 含义 | default |
|---|---|---|
| `IC_BASE_ESTIMATORS` | 空格分隔 base list (grpo/rlsd/prior_shift) | `"grpo rlsd"` |
| `IC_ENABLE_INTERVENTION_LIST` | 空格分隔的 True/False list | `"True"` |
| `IC_DIVERGENCE_METRIC_LIST` | t\* 选择策略 (argmax_excl_eos / argmax_raw / g_t_argmax) | `"argmax_excl_eos"` |
| `IC_TOP_K` | TCCA top-K 位置数 | `3` |
| `IC_LAMBDA_TOKEN_CREDIT` | TCCA c_t 强度 | `1.0` |
| `IC_K` | 每位置 teacher 改写 token 数 | `2` |
| `IC_FAILED_THRESHOLD` | reward 阈值（< 时触发 intervention） | `0.5` |

---

## 2. Prior-Shift（ablation）

```bash
# Nebula sweep (v2a + v2b ablation)
bash nebula_scripts/submit_prior_shift_sweep.sh --dry-run
bash nebula_scripts/submit_prior_shift_sweep.sh

# Local notebook
./run_notebook_prior_shift.sh
```

---

## 3. Baseline 系列

```bash
# GRPO
bash nebula_scripts/submit_grpo_comparison.sh --dry-run
bash nebula_scripts/submit_grpo_comparison.sh

# SDPO
bash nebula_scripts/submit_sdpo_comparison.sh

# RLSD
# 注: RLSD 没有独立 sweep 脚本, 用 intervention_credit_sweep 的
#     base_estimator=rlsd, enable_intervention=False 配置代跑

# TASD
bash nebula_scripts/submit_tasd_simple_sweep.sh
```

---

## 4. 验证 / 查看

```bash
# Job 状态
nebulactl list

# 日志
nebulactl logs <job_id>

# SwanLab projects (常用)
# - https://swanlab.cn/@awesome_jimmy/TGDI-Tier3     (Nebula TCCA / TGDI v3)
# - https://swanlab.cn/@awesome_jimmy/TGDI-local     (本地 smoke)
# - https://swanlab.cn/@awesome_jimmy/PriorShift-Tier1
# - https://swanlab.cn/@awesome_jimmy/Baselines_v2   (历史 GRPO/SDPO)
# - https://swanlab.cn/@awesome_jimmy/Baselines_v3   (重跑 GRPO/SDPO)
```

---

## 5. 环境变量 checklist

提交前确保以下 env vars 已 export（通常通过 `source .env`）：

| 必需 | 用途 |
|---|---|
| `OPENLM_TOKEN` | Nebula auth |
| `OSS_ACCESS_ID` / `OSS_ACCESS_KEY` | OSS 模型/数据访问 |
| `SWANLAB_API_KEY` | SwanLab 上报 |
| `WANDB_API_KEY` | (备用) |

---

## 6. 常见操作

```bash
# 取消某个 job
nebulactl kill <task_id>

# Dry-run 自定义 sweep
IC_DIVERGENCE_METRIC_LIST="argmax_excl_eos argmax_raw g_t_argmax" \
  bash nebula_scripts/submit_intervention_credit_sweep.sh --dry-run

# 单跑某 dataset
DATASET=sciknoweval/chemistry \
  ./run_notebook_intervention_credit.sh smoke
```
