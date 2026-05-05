# TASD Roadmap（v5-baseline 路线图）

> 最近更新：2026-05-05
> 所在分支：v5-baseline（从 52149bd 分叉）
> 配套设计文档：
> - [v14 minimal error_pool](./TASD%20v14%20%E8%AE%BE%E8%AE%A1%E6%96%87%E6%A1%A3-minimal%20error_pool%20on%20v5.md)
> - [B 方案 outcome-conditional teacher reward](./TASD%20B%E6%96%B9%E6%A1%88%E8%AE%BE%E8%AE%A1%E6%96%87%E6%A1%A3-outcome-conditional%20teacher%20reward.md)
> - [C 方案 outcome-based teacher credit assignment](./TASD%20C%E6%96%B9%E6%A1%88%E8%AE%BE%E8%AE%A1%E6%96%87%E6%A1%A3-outcome-based%20teacher%20credit%20assignment.md)

---

## 总览

```
v13 (in flight)  v5 复现 baseline               [单 job, 目标 peak ≥ 0.65]
    ↓ pass
v14              最小 error_pool                [+150 行代码 / 4 job 消融]
    ↓ pass
v15              A 方案 or C 方案（二选一）      [零算力 outcome 融入梯度]
    ↓ pass
v16              B 方案 outcome-conditional     [+teacher forward, token 级 Bellman credit]
    ↓ optional
v17              B + C 融合                     [分段采信: R=1 用 C, R=0 用 B]
```

**设计原则**：
1. 干净基线优先 — v13 复现成功前不叠加任何新机制
2. 代码规模可控 — 每一步 diff ≤ 500 行
3. 零算力方案先行 — C/A 在 B 之前验证 outcome-awareness 有效性
4. 可回滚 — 每个版本作为独立 reward_type / flag，默认关闭

---

## v13 — v5 基线复现 `[in flight]`

- **commit**：`9298637 feat(tasd): v13 v5-replay sweep`
- **脚本**：`nebula_scripts/submit_tasd_v13_v5_replay_sweep.sh`
- **nebula task_id**：`1dfb9284a5b4420e8ff91c3cd2b8e49e`
- **SwanLab project**：`TASD-v13-v5Replay`
- **目标**：单 bio job，严格对齐 v5 实验名超参
- **成功判据**：best_metric@val-core/sciknoweval/acc/mean@16 **≥ 0.65**（v5 参考 0.681）
- **等待**：~4 小时训练结果

**决策分叉**：
- ✅ peak ≥ 0.65 → 开工 v14
- ⚠️  0.55 ≤ peak < 0.65 → 先排查 docker / model 版本差异，再决定
- ❌ peak < 0.55 → 问题不在代码，回查数据 / 环境 / 随机种子

---

## v14 — 最小 error_pool `[设计完成，未开工]`

详见 [v14 设计文档](./TASD%20v14%20%E8%AE%BE%E8%AE%A1%E6%96%87%E6%A1%A3-minimal%20error_pool%20on%20v5.md)。

**代码 TODO**（总 ~200 行 diff）：
- [ ] `verl/trainer/config/tasd_simple.yaml` +8 行：`teacher_context_mode / max_errors_in_pool / error_answer_max_chars / error_pool_format_only`
- [ ] `verl/workers/config/actor.py` +5 行：SelfDistillationConfig 新字段
- [ ] `verl/trainer/ppo/ray_trainer.py` +150 行：5 个辅助方法 + dispatch 分支（不 cherry-pick HEAD，从 v5 重写）
- [ ] `nebula_scripts/tasd_simple/tasd_simple_parametric.sh` +12 行：env → hydra override
- [ ] `nebula_scripts/submit_tasd_v14_error_pool_sweep.sh` +200 行：新建 sweep

**实验矩阵（4 JOB, bio）**：
| Job | teacher_context_mode | format_only | max_errors | 定位 |
|---|---|---|---|---|
| v14-ctrl | per_rollout | — | — | 对照，应复现 v13 peak |
| v14-fmt8 | group_shared | true | 8 | **主推** |
| v14-fmt4 | group_shared | true | 4 | 紧凑池消融 |
| v14-all8 | group_shared | false | 8 | 测 v11 判据 |

**成功判据**：
- v14-ctrl peak ≥ v13 peak × 0.98（per_rollout 不应劣化）
- v14-fmt8 peak **≥ v13 peak + 0.02**（绝对提升）
- `group_with_pool_frac` ∈ [0.3, 0.8]（池既有效触发又不饱和）

---

## v15 — A 方案 或 C 方案 `[设计完成，未开工]`

**二选一决策**（预计 v14 结果出来后定）：

### 选项 1：A 方案 `outcome fusion`
- 代码已在 simplifed 分支 commit 9f12132（可 cherry-pick，约 30 行）
- 公式：`r_t = log P_teacher(y_t) × R`
- 优点：最小实现，只动 reward 计算一行
- 缺点：Σr_t = R·Σ log_p 不是无偏的；outcome=1 时退化为 v5

### 选项 2：C 方案 `teacher credit assignment`（推荐）
详见 [C 方案设计文档](./TASD%20C%E6%96%B9%E6%A1%88%E8%AE%BE%E8%AE%A1%E6%96%87%E6%A1%A3-outcome-based%20teacher%20credit%20assignment.md)。
- 公式：`r_t = R × softmax(log P_teacher)_t × T`
- 优点：Σr_t = R·T 无偏；真正的 credit assignment（不是均匀 scale）
- 缺点：权重函数 f 多了 1 个选型超参（默认 log_prob_softmax）

**代码 TODO**（C 方案，总 ~300 行）：
- [ ] `core_algos.py` +60 行：`compute_teacher_credit_reward` 新函数
- [ ] `tasd_simple.yaml` + `actor.py` +15 行：5 个新字段
- [ ] `dp_actor.py` / `ray_trainer.py` +15 行：dispatch 分支
- [ ] `tasd_simple_parametric.sh` +10 行：env
- [ ] `submit_tasd_v15_credit_sweep.sh` +200 行：6 job 矩阵

**成功判据**：
- v15-Cf2 peak **≥ v14-best peak × 1.02**
- `c_credit_top1_frac` ∈ [0.05, 0.5]（credit 不过度集中也不均匀）

---

## v16 — B 方案 outcome-conditional teacher `[设计完成，未开工]`

详见 [B 方案设计文档](./TASD%20B%E6%96%B9%E6%A1%88%E8%AE%BE%E8%AE%A1%E6%96%87%E6%A1%A3-outcome-conditional%20teacher%20reward.md)。

**代码 TODO**（总 ~500 行）：
- [ ] `compute_teacher_beliefs` 新函数（~100 行）：MCQ 单字母版 + tooluse 完整 GT 版
- [ ] `core_algos.py` +40 行：新 reward_type `teacher_conditional`，`r_t = (V_t - V_{t-1}).clamp`
- [ ] `ray_trainer.py` +80 行：teacher belief 批量并行化 forward + 缓存优化
- [ ] 新 teacher forward pass 打包逻辑（~100 行）
- [ ] `submit_tasd_v16_conditional_sweep.sh` +200 行：4 job 矩阵 (B1-B4)

**成功判据**：
- B1 peak ≥ max(v13, v15) peak × 1.02
- V_t 单调性：correct rollout 上 mean(V_T - V_0) > 0，wrong rollout 上 < 0
- 额外 teacher forward 开销 ≤ 原开销 × 1.5（否则必用缓存优化）

**算力风险**：B 方案是唯一会拉高 teacher forward 开销的方案，需要提前 profile。

---

## v17 — B + C 融合 `[可选，B 验证通过后再决定]`

- 公式：`r_t = α · r_t^B + (1-α) · r_t^C`
- 动机：R=1 时用 C 的零开销 credit，R=0 时靠 B 的边际信念信号
- 实现：加个 `fusion_alpha` 超参 + 两个 reward 加权和
- 实验：α ∈ {0.3, 0.5, 0.7} 扫 best

---

## 当前阻塞 / 等待

| 阻塞项 | 原因 | 预计解除时间 |
|---|---|---|
| v14 开工 | 等 v13 跑完验证 0.65 | +4h |
| v15 方案选型 | 等 v14 结果（A 还是 C） | +1-2 天 |
| v16 算力预算 | 需要 profile teacher forward 批量化方案 | 与 v14 并行做 |

---

## 全周期 DoD（Definition of Done）

| 版本 | 代码 | 实验 | 文档 |
|---|---|---|---|
| v13 | ✅ committed 9298637 | ⏳ in flight | — |
| v14 | ☐ | ☐ 4 job | ✅ 设计 v0.1 |
| v15 | ☐ | ☐ 6 job (C) or 3 job (A) | ✅ C v0.1 |
| v16 | ☐ | ☐ 4 job | ✅ B v0.2 |
| v17 | ☐ | ☐ 3 job | ☐ 设计待写 |

---

_本 TODO 是活文档，每次 nebula job 出结果后需同步更新「当前阻塞」与「DoD」两节。_
