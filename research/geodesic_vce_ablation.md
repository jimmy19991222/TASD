# Q − V 统一框架下的 PG 训练方法

**分支**: `geodesic-sdpo-v2`
**最后更新**: 2026-05-20
**Seed**: 42（所有实验严格统一）

---

## 一、设计原则

把所有正在比较的训练方法**统一到**一个最小架构：

```
loss  = -A_t · log π_θ(y_t)              (PG，沿用 PPO 的 IS clip)
A_t   = Q_t - V_t                         (token-level advantage)
```

不同算法只对应不同的 (Q, V) 选择。这样：
- 每个新算法 = 一行 config，而不是一份新 loss 文件
- 跨方法对比可控（同一份 PG/clip/aggregation/rollout correction）
- Geodesic 等"loss 后处理"（Fisher 加权、IS 截断）和 (Q, V) 正交，可独立开关

### 1.1 Q / V 语义澄清

教科书 RL 里 A = Q − V 的 Q 和 V **都是 return（累计折扣回报的期望）**，不是 reward：

```
V(s_t)     = E[ G_t | s_t ]        state value
Q(s_t,a_t) = E[ G_t | s_t, a_t ]   action value（指定动作 a_t）
A(s_t,a_t) = Q(s_t,a_t) - V(s_t)   "选 a_t 比按策略 π 平均能多带回多少 return"
```

由 V(s)=E_{a~π}[Q(s,a)]，所以 E_{a~π}[A | s] = 0——这是 PG 用 V 做 baseline 减方差的根本理由。

LLM 这套里，token-level 的真实 future return 拿不到（reward sparse），所以我们用各种**代理**：

| 框架角色 | 我们这里的代理 | 是否严格匹配 RL 定义 |
|----------|----------------|---------------------|
| Q_t | `log p_teacher(y_t)` | ❌ 不是 E[future reward \| y_t]，是 teacher 瞬时偏好作为 stand-in |
| V_t (`student`) | log p_s(y_t) | ❌ 不是 E_{a~π}[Q]，是 student 自己对该 token 的瞬时偏好 |
| V_t (`ce`) | Σ_v p_s(v) · log p_t(v) = E_{y~p_s}[Q(y)] | ✅ **严格符合 V = E_a[Q]**——所以 E[A\|s]=0 也严格成立 |
| V_t (`group_mean`) | batch token-mean of Q | ❌ 经验 baseline，不严格符合 V=E[Q]（但 state-conditional → 不引入 bias） |
| V_t (`group_hier`) | within-seq z + across-seq z | 同上 |

**关键含义**：只有 `baseline_type='ce'` 是严格意义上的 V = E[Q]；其它 baseline 都是工程意义上的"任何 state-conditional 量都不引入 bias"的近似。这也是为什么 ce 是"零均值 PG baseline"，其它支只能保证不偏但不一定零均值。如果未来加 critic 网络估计真正的 V(s_t)（GAE 风格），那才是教科书意义的 A=Q−V。

---

## 二、已实现的代码

### 2.1 新增入口：`loss_mode = "teacher_qv"`

[verl/trainer/ppo/core_algos.py](verl/trainer/ppo/core_algos.py#L425) `compute_teacher_qv_advantage`：在 micro-batch 里现算 token-level A = Q − V，复用 vanilla PPO 的 `compute_policy_loss_vanilla`（保留 ratio clip、IS、loss 聚合）。

接入点：[verl/workers/actor/dp_actor.py](verl/workers/actor/dp_actor.py)。复用 SDPO 已有的 teacher forward / EMA / trust-region 基础设施，只换 loss 计算这一步。

### 2.2 baseline_type 四选一

当前 Q 固定为 `log p_teacher(y_t)`（detach）。V 由 `policy_loss.teacher_qv.baseline_type` 选：

**A_t 永远 detached**——整个 `compute_teacher_qv_advantage` 包在 `with torch.no_grad()` 里，跟 `compute_grpo_outcome_advantage` / `compute_self_teacher_advantage` 同样的约定。Advantage 在 PG 里只是个 scalar weight，梯度只从 `-A · log p_s` 里 ∇log p_s 那一支走，A 本身不应该有梯度（如果有，loss 求导多出 `-∇A · log p_s`，结果就不是 PG 梯度）。

| baseline_type | V_t | A_t | 等价于 |
|---------------|-----|-----|--------|
| `student` | log p_s(y_t) | log p_t − log p_s | SDPO sampled-token 形式（on-policy + 无 clip 触发时严格等价 reverse-KL gradient） |
| `ce` | **Σ_v p_s(v) · log p_t(v)** = E_{y~p_s}[log p_t] | log p_t(y_t) + CE(p_s, p_t) | 真正的 V_CE（zero-mean PG baseline） |
| `group_mean` | batch-mean(Q over valid tokens) | (Q − μ) [/ σ if `norm_by_std`] | 单一全局 baseline，类似 outcome-only GRPO 但在 token 维上 |
| `group_hier` | within-seq z(Q) + across-seq z(seq_mean_Q) | GRPO 风格双层 z-score（uid 分组） | 把"token 在 seq 内的相对位置"和"seq 在组内的相对位置"两个尺度叠加 |

> 注意：student / ce 是"等价于 SDPO sampled-token"，但**严格等价的前提是 on-policy（`ppo_epochs=1`）且 PPO clip 未触发**——off-policy 多 epoch 时 teacher_qv 走 vanilla PPO 路径，会多出 importance sampling 项。

实现细节：
- `baseline_type=ce` **需要 full vocab 或 topk-aligned log-probs**。Dp_actor 自动检测并打开 `full_logit_distillation=True, distillation_topk=100`（默认）。topk 模式下用 `_qv_add_tail` 补一个 sum-to-1 的尾桶。
- 所有 baseline 都支持：
  - `norm_by_std`: 是否除以对应尺度的 std
  - `clip_value`: 是否 clamp |A| ≤ c
  - `std_floor`: std 下限防数值崩溃
  - `detach_q / detach_v`: 是否切断对应分支的梯度（默认都切，保持纯 PG 行为）

#### V_CE 公式 + 数值例子（澄清 A 的正负号）

```
V_t = E_{y~p_s(·|ctx_t)}[ log p_t(y|ctx_t) ]    ← 负数（log 概率的期望）
    = -CE(p_s, p_t)                              ← 等价写法
A_t = log p_t(y_t) - V_t
    = log p_t(y_t) + CE(p_s, p_t)                ← 同一个东西的不同写法
```

第三行的 "+CE" 不是"两个正数相加"——`log p_t(y_t)` 是负数，CE 是正数，**符号取决于"采样到的 token 在 teacher 眼里的 log-prob，是否高于 student 分布下 log p_t 的平均水平"**。

例：vocab={A,B,C}，p_s=[0.7, 0.2, 0.1]，p_t=[0.4, 0.5, 0.1]：

```
log p_t       = [-0.92, -0.69, -2.30]
V_t           = 0.7·(-0.92) + 0.2·(-0.69) + 0.1·(-2.30) = -1.012
CE(p_s,p_t)   = +1.012

采样到 B (teacher 喜欢): A = -0.69 - (-1.012) = +0.32  → 提升 student 给 B 的概率
采样到 C (teacher 不喜欢): A = -2.30 - (-1.012) = -1.29  → 压低 student 给 C 的概率
采样到 A (中等): A = -0.92 - (-1.012) = +0.09  → 轻微提升

E[A | s] = 0.7·0.09 + 0.2·0.32 + 0.1·(-1.29) ≈ 0  ✓ (state-dependent zero-mean baseline)
```

### 2.3 Geodesic Fisher manifold weight（已接入 teacher_qv）

[verl/trainer/ppo/core_algos.py](verl/trainer/ppo/core_algos.py) 新增 `compute_geodesic_manifold_weight` 公共 helper。SDPO 和 teacher_qv **共享同一个 Geodesic 开关**：`self_distillation.use_geodesic` + `self_distillation.geodesic_trust_region`（沿用原 SDPO 的 config 字段名，不另立门户）。

公式：
```
F_t = Σ_v p_s(v) · (1 - p_s(v))   （full-logit 模式，对 vocab 求和）
    或 p_s(y_t)·(1-p_s(y_t))      （token-level 模式，仅采样 token）
w_t = trust_region + log(1/F_t - trust_region + 1)   if 1/F_t > trust_region
    = 1/F_t                                          otherwise
```

应用方式：在 teacher_qv 分支里，用 `qv_advantages = qv_advantages * w_t`，再喂给 `compute_policy_loss_vanilla`。由于 w_t > 0 且 PPO loss 对 advantages 是分段线性的，这等价于把整个 per-token loss 乘以 w_t。

记录的指标（前缀 `actor/geodesic_*`）：mean / max / min / std / trust_region_scale / full_logit_mode（0 或 1，反映是否是 full-vocab Fisher）。

### 2.3 涉及的修改

| 文件 | 变化 |
|------|------|
| [verl/workers/config/actor.py](verl/workers/config/actor.py) | 新增 `TeacherQVConfig` dataclass，嵌入 `PolicyLossConfig.teacher_qv` |
| [verl/trainer/ppo/core_algos.py](verl/trainer/ppo/core_algos.py#L425) | 新增 `compute_teacher_qv_advantage`（四种 baseline + metrics） |
| [verl/workers/actor/dp_actor.py](verl/workers/actor/dp_actor.py) | 新增 `loss_mode=teacher_qv` 分支；新增 `_qv_add_tail` |
| [verl/trainer/ppo/ray_trainer.py:680](verl/trainer/ppo/ray_trainer.py#L680) / [main_ppo.py:131](verl/trainer/main_ppo.py#L131) / [fsdp_workers.py:896](verl/workers/fsdp_workers.py#L896) | gate `loss_mode == "sdpo"` → `loss_mode in ("sdpo", "teacher_qv")` |
| [verl/trainer/config/actor/actor.yaml](verl/trainer/config/actor/actor.yaml) | 新增 `policy_loss.teacher_qv` schema 块 |
| [verl/trainer/config/teacher_qv.yaml](verl/trainer/config/teacher_qv.yaml) | 新 Hydra config，继承自 `sdpo` |
| [run_local_teacher_qv_smoke.sh](run_local_teacher_qv_smoke.sh) | 单卡 smoke test，传 baseline_type 即可 |
| [nebula_scripts/sdpo/teacher_qv_sciknoweval_parametric.sh](nebula_scripts/sdpo/teacher_qv_sciknoweval_parametric.sh) | 集群 parametric 脚本 |
| [nebula_scripts/submit_teacher_qv_ablation.sh](nebula_scripts/submit_teacher_qv_ablation.sh) | 4 组 ablation 提交器 |

### 2.4 老 `use_vce` / `use_log_pi_s` 配置项已移除

历史上的 `algorithm.use_vce` 和 `algorithm.use_log_pi_s` 这两个 flag 在 `compute_self_teacher_advantage` 里只 parse 不使用——之前所有标着 "VCE" 的实验跑的都是 sequence-level GRPO + clip（不含任何 V_CE 数学）。已经在 [algorithm.py](verl/trainer/config/algorithm.py)、[sdpo.yaml](verl/trainer/config/sdpo.yaml)、[geodesic_vce_ablation_parametric.sh](nebula_scripts/sdpo/geodesic_vce_ablation_parametric.sh) 里清除掉这两个 flag（保留 `clip_value` / `adv_std_floor` 因为它们真的有用）。

老 `compute_self_teacher_advantage` 函数本身保留，作为 sequence-level GRPO 的别名仍可用。**真 V_CE 完全走 `teacher_qv` 这条新路径。**

### 2.5 还没接的能力（下一步要补）

- ~~**Geodesic manifold weight 没挂到 teacher_qv loss 路径**~~ ✅ 已接，见 2.3
- **`q_source` 维度只有 `log_teacher`**——把 GRPO（Q = outcome reward）、REINFORCE（V = 0）也纳入统一框架还需要再加一个 `q_source` 配置项
- **SDPO 的 full-logit JSD 分支**（alpha ≠ 1）本质是 vocab 维 KL 散度的梯度，不是 token-level PG，不能用 Q−V 表达，需要继续保留独立 loss path

---

## 三、Q − V 表达下的方法对照

| 算法 | q_source | baseline_type | 备注 | 当前是否可跑 |
|------|----------|---------------|------|--------------|
| SDPO sampled-token | log_teacher | student | reverse-KL 在 sampled token 上的展开 | ✅ |
| 真 V_CE | log_teacher | ce | zero-mean baseline | ✅ |
| Token-level GRPO（teacher 提供 Q） | log_teacher | group_mean / group_hier | 不依赖 outcome reward 的"组内归一" | ✅ |
| Outcome GRPO | outcome_reward | group_mean | 经典 GRPO（R_i − μ_g） | ❌ 待加 `q_source` |
| REINFORCE | outcome_reward | zero | 无 baseline | ❌ 待加 `q_source` + `zero` baseline |
| SDPO full-logit JSD | — | — | 不在 Q−V 框架内 | 保留独立路径 |
| SDPO/VCE + Geodesic | 同上 | 同上 + `use_geodesic=True` | Fisher 加权 loss 后处理 | ❌ Geodesic 待移植到 teacher_qv |
| Critic-based PPO | learned_value | bootstrap | 需要 critic 网络 | ❌ 本批不做 |

---

## 四、接下来的实验计划

### Phase 0（先补完缺口）

| 任务 | 状态 | 产出 |
|------|------|------|
| 把 Geodesic manifold weight 移植到 teacher_qv loss | ✅ 完成 | `compute_geodesic_manifold_weight` helper，SDPO 与 teacher_qv 共享同一个开关 |
| 移除 dead `use_vce` / `use_log_pi_s` 配置 | ✅ 完成 | 杜绝"挂了 use_vce=True 但其实跑 GRPO"的老坑 |
| 给 teacher_qv 加 `q_source ∈ {log_teacher, outcome_reward}` | ⏳ Pending | GRPO 落到统一框架，阻塞 Phase 2 |
| 加 `baseline_type=zero` | ⏳ Pending | REINFORCE 落到统一框架 |

### Phase 1: 真·VCE 与 SDPO 对照（先跑这一组）

| 组 | loss_mode | baseline_type | use_geodesic | 期望验证 |
|----|-----------|---------------|--------------|----------|
| A | teacher_qv | student | False | 真 · SDPO sampled-token（基线） |
| B | teacher_qv | ce | False | **真·V_CE，预计观察 entropy collapse**（验证文档第 34-37 行假说） |
| C | teacher_qv | ce | True | V_CE + Geodesic（检验 Geodesic 是否救回崩溃） |
| D | teacher_qv | student | True | SDPO + Geodesic（对照 D vs A 看 Geodesic 净增量） |

控制变量：`Qwen3-8B / sciknoweval-biology / seed=42 / lr=1e-5 / batch=32 / rollout_n=8 / 250 step / distillation_topk=100`。

关键观测：
- `actor/entropy`：B 是否在前 50 步内快速下降到接近 0；C 是否被 Geodesic 拉住
- `actor/teacher_qv_adv_mean`：B 应稳定在 0 附近（zero-mean baseline 的性质），偏离 0 说明 topk=100 近似误差大
- `actor/teacher_qv_adv_std`：A 跟 B 量纲差异
- `val-core/sciknoweval/acc/mean@16`：终端准确率
- `actor/geodesic_mean_weight`（C/D 才有）

### Phase 2: Q 源对照（验证统一框架的覆盖力）

固定 V = group_mean（最稳的 baseline），改 Q：

| 组 | q_source | 等价于 |
|----|----------|--------|
| E | log_teacher | "teacher 信号 + outcome 风格 baseline"（新组合） |
| F | outcome_reward | 经典 GRPO 复刻 |

如果 E、F 都收敛且 E ≥ F，说明 teacher 信号比 sparse outcome reward 更稠密更稳；如果 F 显著好，说明 teacher 信号有偏（teacher 已经收敛后效用衰减）。

### Phase 3: 双层归一 vs 单层（验证 hierarchical 设计）

固定 q_source=log_teacher：

| 组 | baseline_type | norm_by_std | 看什么 |
|----|---------------|-------------|--------|
| G | group_mean | True | 单层全局 z-score |
| H | group_hier | True | within-seq z + across-seq z 叠加 |
| H' | group_hier | False | 只做中心化、不归一 |

预期 H > G（hierarchical 能保留 seq 内 token 的相对结构）；H vs H' 看归一化必要性。

### Phase 4（如果 Phase 1 的 B 真崩溃且 C 救回）

写一个最小的 entropy collapse 复现 case（合成数据 + 玩具 vocab），把崩溃机制讲透，附在论文 appendix。

---

## 五、Q 估计的改进方向（post-Phase 3）

当前 Q_t = log p_teacher(y_t) 只是 **instantaneous proxy**——teacher 对采样 token 的瞬时偏好，**不是教科书 RL 里的 Q(s,a) = E[future return | s, a]**。1.1 节已经讲过，这是 LLM-RL 里普遍的代理选择，但显然不是最优。下面四个方向按"工程改动量"从小到大列，可以作为 teacher_qv 框架的横向扩展。

### A. TD/GAE 多步 teacher 信号（最便宜，回报最大）

把 log p_t 当稠密 token-level reward，做标准 TD/GAE bootstrap：

```
r_t = log p_teacher(y_t)
Q_t = Σ_{k=0}^{T-t-1} γ^k · r_{t+k}              ← N-step return
δ_t = r_t + γ·V_{t+1} - V_t                      ← TD residual
A_t = Σ_{k≥0} (γλ)^k · δ_{t+k}                   ← GAE
```

V_t 仍用现有四种 baseline 之一（student / ce / group_*）。

**改动**：在 `compute_teacher_qv_advantage` 里加一个 `q_aggregation: instant | nstep | gae` 配置，instant 是当前行为；nstep/gae 在 token 维做反向扫描（O(T)，不需要任何额外 forward）。
**优势**：A 立刻变成"token 对未来 teacher confidence 累计的贡献"，跟教科书 PG 形式对齐；γ < 1 自动衰减远期信号。
**风险**：γ / λ 引入新超参；累加后 Q 的量纲会扩大 (1-γ)^(-1) 倍，需要重新调 `clip_value`；teacher 在长 sequence 后段如果置信度异常，会污染前段 A。
**期望收益**：中-高，理论清晰，跟 4 个 baseline_type 都可以正交叠加。

### B. Teacher 价值头（critic on teacher hidden states）

冻结 teacher backbone，在最后一层 hidden state 上挂个小 MLP head：

```
V_φ(h_t) → scalar，目标拟合 sequence terminal verifier reward
critic loss: L_V = (V_φ(h_t) - R_terminal)²       ← 跟 actor loss 一起优化
Q_t = log p_teacher(y_t) + γ·V_φ(s_{t+1})         ← TD-bootstrap 风格
```

**改动**：teacher forward 多 return hidden states；加一个独立的 critic head 模块；在 actor loss 里加 critic loss 项（带系数 c_v）；新增 `q_source = teacher_critic`。
**优势**：把稀疏 outcome reward 通过 critic 反向传到 token 级，bootstrap 比 raw teacher log-prob 更精准；critic head 参数量极小，只训 head 几乎零额外算力。
**风险**：critic 冷启动阶段全是 noise，前期反而拖累训练；需要 critic learning rate / loss coefficient 调参；用 hidden states 把 teacher 和 critic 绑死，teacher 更新（EMA）会让 critic 失配。
**期望收益**：中，依赖 critic 训得好不好；适合作为 A 的下一步增强。

### C. Teacher peek with ground truth（专门针对 verifiable-answer 数据集）

sciknoweval 这类数据每条都有 ground-truth answer。让 teacher 在**已经看到答案**的条件下评估 student 当前 token：

```
Q_t = log p_teacher(y_t | prompt, ground_truth_answer, y_<t)
```

本质是 **inverse RL 视角**：teacher + oracle ≈ optimal policy，optimal policy 给的 log-prob ≈ Q*。

**改动**：复用 SDPO 现有的 reprompt 框架（[`ray_trainer.py:680`](verl/trainer/ppo/ray_trainer.py#L680) `_maybe_build_self_distillation_batch`），把 `solution_template` 从"successful previous attempt"换成"correct answer reference"，构造一个新的 `q_source = teacher_oracle`。
**优势**：理论上**最接近真 Q***，因为 teacher + oracle 信息最完整；不需要训 critic、不需要 rollout。
**风险**：(1) 训练信号里灌入了 ground-truth，泛化性可能下降（student 学到"假设 teacher 看过答案"的捷径）；(2) 跟 SDPO 的 reprompt 路径高度耦合，"Q 估计质量"这个变量不容易单独 ablate；(3) 只能用于有可枚举正确答案的数据集，generalization 受限。
**期望收益**：高（如果泛化性问题能控制住），但**应该单独作为 baseline 跟 A/B 比，不要混进 Phase 1 那种受控对比**。

### D. Teacher rollout completion（最准但成本爆炸）

每个 token 位置 t，让 teacher 从 y_<=t 接着续写到底，跑 verifier 拿 0/1 reward：

```
Q_t = verifier( teacher_complete(prompt, y_<=t) )
```

直接是 terminal-reward 形式的真 Q。但每个 training token 多一次 teacher rollout，**成本 O(T) 倍**——基本不可行，除非：
- 只在 reasoning chunk boundary（每 K token）做一次，token 内插值
- 或者 batch 起来用 vLLM prefill + 短 decode 把 latency 摊下来
- 或者只对 high-uncertainty 位置做（用 student entropy 选样）

**期望收益**：理论最强，工程代价最大。等 A+B 走完且效果还不够时再考虑。

### 优先级建议

| 顺序 | 方向 | 理由 |
|------|------|------|
| 1 | **A (TD/GAE)** | 改动最小，让 Q 真正变成 future return，跟教科书 PG 对齐；可以跟当前 4 个 baseline_type 正交叠加 |
| 2 | **B (critic head)** | A 跑通后的自然延伸：把 outcome reward bootstrap 进 token 级 Q |
| 3 | **C (oracle peek)** | 当作独立研究分支，因为公平对比的难度大，但理论上界最高 |
| 4 | **D (full rollout)** | 等 A+B 跑出明显瓶颈再考虑 |

A 和 B 都可以归到现有 `teacher_qv` 框架的扩展（新增 config 字段），不破坏现有 4 个 baseline_type 的实验设计。C 应当作 SDPO 系列的新变种 (`teacher_qv_oracle`)。D 单独立项。

---

## 六、实验执行约定

- 一次只改一个变量；同时改两个的实验不读
- 每个变量都要有 ON/OFF 两组（否则不构成 ablation）
- Seed 永远 42；要看种子敏感性单独再跑一轮 multi-seed
- 所有 run 通过 `nebula_scripts/submit_teacher_qv_ablation.sh` 提交，保证 `JOB_NAME / GROUP_NAME` 在 SwanLab 自动分组
- 监控阈值：`actor/entropy < 0.1` 或 `grad_norm > 100` 自动钉钉告警
- Phase 1 完成前不动 hyperparams，只改 baseline_type / use_geodesic

---

## 七、未决问题（提醒自己）

1. **VCE 的 zero-mean baseline 在 topk=100 下近似多准？** 如果 `actor/teacher_qv_adv_mean` 系统性偏离 0，需要换 full vocab 或加 add_tail 修正幅度
2. **Geodesic 加到 teacher_qv 后，clip_value 还需不需要？** Geodesic 已经在做 fisher 倒数缩放，可能重复
3. **是否要给 teacher_qv 加 group-relative reward**（即在 V 里加上"组内其他 sample 的 token-level Q 均值"）？这就把 GRPO 的 group 结构搬到 token 级，可能比 group_mean / group_hier 更强
4. **`detach_v=False`（让 V 也参与梯度）是 actor-critic 风格**，可能更不稳但理论上方差更小，留作 Phase 4 之后探索

---

**维护人**：Jimmy
**对应 commit hash**：实现 commit 待 push
