# DPO-2S 实验设置参考

**模型**: Qwen3-4B  
**SwanLab Project**: `awesome_jimmy/DPO-Comparison-4B`  
**日期**: 2026-06-01 ~ 2026-07-01

---

## §1 两套 Hyperparameter Profile

论文中所有实验分属两套 profile，按**任务类型**而非数据集名划分：

### Profile A: Generalization（知识 + 工具使用）

适用数据集：`sciknoweval/*`（chemistry/biology/material/physics）、`tooluse`、`math500`、`gsm8k`、`competition_math`

| 参数 | 值 | 说明 |
|---|---|---|
| LR | **1e-5** | |
| lr_warmup_steps | **10** | |
| GRPO mbs | 32 | `ppo_mini_batch_size` |
| SDPO mbs | 32 | |
| SDPO α | **0.5** | JSD（α=0 forward KL, α=1 reverse KL） |
| SDPO topk | 100 | `distillation_topk` |
| val_n | **16** | `rollout.val_kwargs.n` |
| rollout_n | 8 | `rollout.n`（train） |
| train_batch_size | 32 | |
| total_training_steps | 250（sciknoweval）/ 150（math）/ 300（competition_math） | |
| save_best_metric | `val-core/{dataset}/acc/mean@16` | |
| teacher_update_rate | 默认 | sciknoweval 脚本不显式设置 |

### Profile B: Rich Feedback（代码生成）

适用数据集：`lcb_v6`（LiveCodeBench）

| 参数 | 值 | 说明 |
|---|---|---|
| LR | **1e-6** | 比 generalization 低 10x |
| lr_warmup_steps | **0** | |
| GRPO mbs | 8 | |
| SDPO mbs | **1** | 极小 batch → 更精细的蒸馏 |
| SDPO α | **1.0** | 纯 reverse KL |
| SDPO topk | **20** | |
| val_n | **4** | |
| rollout_n | 8 | |
| train_batch_size | 32 | |
| total_training_steps | 250 | |
| save_best_metric | `val-core/livecodebench/acc/mean@4` / `livecodebench/acc/mean@16` | |
| teacher_update_rate | **0.01** | 更保守的 teacher 更新 |

### ⚠️ 易混淆点

- **Tooluse 属于 Profile A**（generalization），不是 Profile B（rich_feedback）
- 论文 `experiments/generalization/run_baseline_grpo_all.sh` 同时处理 sciknoweval 和 tooluse，使用相同参数
- v1 提交脚本中 tooluse 的 parametric scripts 错误硬编码了 Profile B 的 warmup=0, val_n=4

---

## §2 DPO-2S 方法专属参数

| 参数 | 值 | 说明 |
|---|---|---|
| TWO_STAGE | True | 两阶段架构 |
| STAGE1_N | 4 | Stage 1 独立 rollout 数量 |
| N_SPLITS | 1 | 每棵树分裂次数 |
| N_TREES | 2 | 分裂树数量 |
| BRANCH_TOKEN_LOSS_MODE | suffix | 仅 branch token 之后的 suffix 计算 loss |
| SUCCESS_THRESHOLD | 0.3 | Stage 1 chosen 筛选阈值 |
| DPO_USE_REF | True | with-ref（引入 π_ref 隐式 KL） |
| DPO_COEFFICIENT (β) | 2.0（知识）/ 0.5（推理/代码） | 任务依赖 |
| ENTROPY_COEFF | 0 | DPO-2S 不使用 entropy regularization |
| TWO_STAGE_TEACHER_MODE | ref_or_marker | teacher context 来源 |
| TOP_K | 10 | student top-K 候选 |
| TEACHER_TOP_K | 100 | teacher 选择范围 |
| ENTROPY_WINDOW | 20 | rolling z-score 窗口 |
| ENTROPY_SIGMA_START | 2.0 | 初始 σ 阈值 |
| ENTROPY_SIGMA_FLOOR | 0.5 | σ 下限 |
| ENTROPY_SIGMA_STEP | 0.5 | σ 衰减步长 |
| ADV_STD_FLOOR | 0.05 | advantage 标准差下限 |

---

## §3 脚本 → Profile 映射

### Parametric Scripts（底层执行脚本）

| 脚本 | Profile | 关键硬编码参数 |
|---|---|---|
| `grpo/grpo_sciknoweval_parametric.sh` | A | wu=10, vn=16, 通用 DATASET env |
| `sdpo/sdpo_sciknoweval_parametric.sh` | A | wu=10, mbs=32, topk=100, vn=16 |
| `grpo/grpo_tooluse_parametric.sh` | A ✅ (已修正) | wu=10, vn=16 |
| `sdpo/sdpo_tooluse_parametric.sh` | A ✅ (已修正) | wu=10, mbs=32, topk=100, vn=16 |
| `grpo/grpo_lcb_parametric.sh` | B | wu=0, vn=4, metric=livecodebench |
| `sdpo/sdpo_lcb_parametric.sh` | B | wu=0, mbs=1, topk=20, vn=4, tur=0.01 |
| `grpo/grpo_branching_sciknoweval_parametric.sh` | A | wu=10, vn=16（DPO-2S 共用） |

### Submit Scripts（实验提交脚本）

| 脚本 | 数据集 | Profile | LR | 状态 |
|---|---|---|---|---|
| `submit_tooluse_lcb_experiments.sh` | tooluse + lcb | 混用 | 1e-5 | ❌ v1 已废弃（tooluse baseline 参数错） |
| `submit_tooluse_v2.sh` | tooluse | A ✅ (已修正) | 1e-5 | ✅ 可用 |
| `submit_lcb_v2.sh` | lcb_v6 | B | 1e-6 | ✅ 已提交 3 jobs |
| `submit_competition_math.sh` | competition_math | A | 1e-5 | ✅ 已提交 5 jobs |
| `submit_dpo_comparison.sh` | math500/gsm8k + bio | A | 1e-5 | ✅ 已完成 |
| `submit_teacher_guided_beta.sh` | sciknoweval | A | 1e-5 | 🔄 7 runs running |
| `submit_teacher_qv_anticollapse.sh` | sciknoweval | A | 1e-5 | — |
| `submit_tg_branching_v3.sh` | sciknoweval | A | 1e-5 | — |

---

## §4 实验状态汇总（2026-07-01）

### ✅ 已完成，结果可用

| 数据集 | GRPO | SDPO | DPO-2S marker | DPO-2S withref | 验证 |
|---|---|---|---|---|---|
| sciknoweval × 4 域 | ✅ | ✅ α=0.5 | ✅ β=2.0 | ✅ β=0.5, 2.0 | SwanLab config 匹配 Profile A |
| Math500 | ✅ | ✅ | ✅ | ✅ β=0.5 | SwanLab config 匹配 Profile A, st=150 |
| GSM8K | ✅ | ✅ | ✅ | ✅ β=0.5 | SwanLab config 匹配 Profile A, st=150 |

### ⏳ 已提交，等待结果

| 数据集 | 实验 | Jobs | 提交时间 |
|---|---|---|---|
| Competition Math | GRPO + SDPO + DPO-2S withref β={0.5,2.0} | 4 | 7/1 16:30（β=1.0 已取消，非必要消融） |
| LCB v2 | GRPO + SDPO + DPO-2S withref β=0.5 | 3 | 7/1 16:27 |
| Tooluse v2 | GRPO + SDPO + DPO-2S withref β=0.5 | 3 | 7/1 22:00（修正后重跑） |

### 🔄 运行中

| 实验 | 状态 | 说明 |
|---|---|---|
| Teacher-guided β=0.5 × 4 域 | RUNNING | 500 steps |
| Teacher-guided β=2.0 × 3 域 | RUNNING | 500 steps（缺 chemistry β=2.0） |
| 500-step 延长 × 2 | RUNNING / CRASHED | Biology 过拟合确认 |

### 💀 已失败

| 实验 | 状态 | 说明 |
|---|---|---|
| Entropy coeff=0.01 × 4 | CRASHED | 全部崩溃 |
| 500-step × 3 (Material/Chemistry β=0.5/Physics β=0.5) | CRASHED | — |

---

## §5 性能优化

### 5.1 Tensor Parallelism (TP=2)

**问题**：DPO-2S 在 math/competition_math 上每步 ~600s（sciknoweval 仅 ~200s），原因是 math CoT 长度 500~2000 tokens，vLLM 生成 token-by-token 受限于单卡 memory bandwidth。

**解决方案**：将 `tensor_model_parallel_size` 从硬编码 `1` 改为环境变量 `${TENSOR_MODEL_PARALLEL_SIZE:-1}`，默认值不变（向后兼容）。

**改动文件**：
- `nebula_scripts/grpo/grpo_sciknoweval_parametric.sh`
- `nebula_scripts/sdpo/sdpo_sciknoweval_parametric.sh`
- `nebula_scripts/grpo/grpo_branching_sciknoweval_parametric.sh`

**使用方式**：submit script 中加 `--env=TENSOR_MODEL_PARALLEL_SIZE=2`，例如 `submit_competition_math.sh` 已设置 TP=2。

**预期收益**：单条 rollout 生成速度提升 ~1.8x，整体 step 时间从 ~600s 降到 ~400s（competition_math 300 步节省 ~17 小时）。

### 5.2 Streaming Stage 1 → Stage 2 Overlap

**问题**：原 `_run_two_stage_pipeline` 严格串行——Stage 1 的 4 条 rollout **全部完成后**才选 chosen 并启动 Stage 2。在 math 上 Stage 1 单条 ~50s，4 条并行但最慢的可能拖到 ~200s，期间 GPU 空等。

**解决方案**：改用 `asyncio.as_completed` 流式处理 Stage 1 rollout，**第一条满足 success_threshold 的 rollout 立刻触发 Stage 2**，不等待其余 Stage 1 完成。Stage 2 branching（n_trees=2 并行）与剩余 Stage 1 tail 并发执行。

**改动文件**：`verl/experimental/agent_loop/branching_agent_loop.py`
- 新增 `_generate_one_stage1_rollout` 方法（从 `_generate_stage1_rollouts` 抽出单条逻辑）
- 重写 `_run_two_stage_pipeline`：
  - Stage 1 用 `asyncio.ensure_future` + `asyncio.as_completed` 流式评分
  - 第一条 `score ≥ threshold` 立即 break 并启动 Stage 2
  - Stage 2 用 `asyncio.gather` 并发运行 n_trees 棵分裂树
  - 最后 `asyncio.gather` 收集剩余 Stage 1 输出

**正确性保证**：
- Stage 1 输出顺序变化（按完成时间而非 idx），但 DPO loss 只关心 chosen/lost pair，不依赖顺序
- Stage 2 每棵树用独立 `traj_id` 和 `tree_idx`-based seed offset，并发执行不影响随机性
- vLLM continuous batching 天然支持并发请求，无资源死锁

**预期收益**：Stage 2 提前启动 ~100~150s（取决于 Stage 1 中最快 vs 最慢的 gap），整体 step 时间进一步压缩。与 TP=2 叠加，competition_math 300 步可从 ~50 小时降到 ~30 小时。
