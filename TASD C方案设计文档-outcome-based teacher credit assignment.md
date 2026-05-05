# TASD C 方案设计文档：outcome-based teacher credit assignment

> 版本：v0.1（草稿，未开工）
> 作者触发思路：memory 3596d1d2 / fbcf90ec / 4d94e090 ——"outcome × teacher-allocated credit"
> 位置关系：C 方案 与 A、B 并列但互斥；路线图上位于 v15 候选（与 A 方案二选一）或 v17（A、B 验证失败后回退）

---

## 1. 背景与动机

### 1.1 现有三个方案定位

| 方案 | 公式 | 特点 | 状态 |
|---|---|---|---|
| **v5 baseline** | `r_t = log P_teacher(y_t \| ctx)` | 纯 teacher 信号，outcome 完全不参与梯度 | 已复现（v13 in flight） |
| **A 方案 (v12)** | `r_t = log P_teacher(y_t \| ctx) × outcome` | outcome 作序列级 scale，**均匀缩放**所有 token | 已实现（commit 9f12132），未验证 |
| **B 方案 (v16)** | `r_t = log P_teacher(y* \| ctx_t) - log P_teacher(y* \| ctx_{t-1})` | outcome-conditional telescoping，Σr = V_T−V_0 | 设计文档在手 |
| **C 方案 (本文)** | `r_t = R × w_t, Σ w_t = 1, w_t = f(teacher_t)` | outcome 作 token reward 基底，teacher 作 credit 分配器 | 设计中 |

### 1.2 C 方案解决的问题

**A 方案的短板：均匀 scale，没有 credit assignment**
- `outcome=0` 时整条序列 reward=0（断掉复读自洽 ✅），但也失去了"哪些 token 是错误关键点"的信号；
- `outcome=1` 时只是把 v5 的 teacher reward 整体乘 1，退化为 v5（没有任何新信号）。
- 不满足 `Σ r_t = R`，Σ r_t 随 teacher 分布而变，不利于与 PPO advantage 的无偏估计。

**B 方案的短板：算力翻倍 + 需要 GT**
- 每个 rollout 要 teacher forward 两次（一次带 GT 条件 y*，一次不带），训练吞吐减半；
- 没有 GT 的数据集（如 open-ended 生成）无法使用；
- teacher 自身是 student EMA，对 GT 的 conditional log_prob 不一定稳定。

**C 方案核心设想**：
```
把序列总 reward R = outcome 视为"蛋糕"，teacher 信号视为"分蛋糕的规则"；
每个 token 按 teacher 给的权重 w_t 分得一块 R × w_t，权重和为 1。
```
优点：
- ✅ **Σ r_t = R 无偏**：和 outcome 完全对齐，PPO advantage 期望等价 outcome-level RL；
- ✅ **无需额外 teacher forward**：只用 rollout 时已有的 teacher log_prob / entropy，零算力增量；
- ✅ **不依赖 GT**：任意有 outcome reward 的数据集都可用（包括 open-ended）；
- ✅ **错样本 reward=0**：outcome=0 时整序列 reward=0，天然断开"复读 → teacher追随 → 自洽"循环；
- ✅ **保留 credit 差异**：token 间仍有 w_t 差异，梯度被引导到 teacher 认为关键的位置。

---

## 2. 核心公式

### 2.1 基本形式

```
outcome reward:  R ∈ {0, 1}        (或 shifted: R' = 2R-1 ∈ {-1, 1})
teacher signal:  s_t, t=1..T
normalized weight: w_t = f(s_t) / Σ_{τ=1..T} f(s_τ)     (Σ w = 1)
final token reward: r_t = R × w_t × T    (乘 T 是可选的 scale，见 §2.4)
```

不乘 T 时：`Σ r_t = R`（reward per sequence = R）；乘 T 时：`mean(r_t) = R`，量级和 v5 的 token reward 接近。

### 2.2 权重函数 f 的候选

| 选型 | f(teacher_t) | 含义 | 与 memory 3596d1d2 符合度 |
|---|---|---|---|
| **f1: neg-log-prob** | `-log P_teacher(y_t \| ctx)` | teacher 困惑（NLL 大）= 关键决策点 | ⚠️ 与"reward大=关键"相反 |
| **f2: log-prob** | `log P_teacher(y_t \| ctx)` (shifted non-neg) | teacher 熟练（高 log_prob）= 关键决策点 | ✅ 完全符合 |
| **f3: entropy** | `H_teacher,t = -Σ p log p` | teacher 整体不确定 = 决策分叉点 | 中性 |
| **f4: 1-top1** | `1 - max_k P_teacher(k \| ctx)` | top-1 不占优 = 决策分叉点 | 中性 |
| **f5: uniform** | `1/T` | 均匀分配 = outcome-only RL baseline | （对照组） |

**memory 3596d1d2 的原话**：
> reward大的token是关键决策点，reward小的token是冗余或错误环节

这里的 "reward" 指**teacher 给的 token reward**，即 `log P_teacher(y_t | ctx)`，越大越确定，越是 teacher 认为的"关键决策"。
⇒ **对齐 f2**。

**推荐默认 f2**（log-prob shifted non-neg）：
```
f2(s_t) = log P_teacher(y_t | ctx) + C     # C = |min_t log_prob| + ε，确保非负
       ≈ -NLL + C
```
在 shift 后 softmax 归一化，相当于 `w_t = softmax(log_prob / τ)`（τ=1 时等价于 teacher output prob 的 T-softmax）。

### 2.3 完整算法（teacher_credit_assign reward_type）

```python
# 单条 rollout 内：
seq_logp = [logP_teacher[y_t | ctx_t] for t in range(T)]   # v5 已有
outcome = 1.0 if is_correct else 0.0                        # reward model 提供

# Step 1: 归一化权重（f2: softmax of log_prob）
# 数值稳定：减去 max，softmax 等价
shifted = seq_logp - max(seq_logp)
w = softmax(shifted / temperature)                           # Σ w = 1

# Step 2: credit 分配
r = outcome * w * T                                          # 乘 T 使 mean(r) = outcome
# 注：乘 T 后 Σ r = outcome * T，和 v5 的 Σ r = Σ logP 量级相当

# Step 3: 可选的 shifted outcome（让错样本 reward=-1 而非 0）
if outcome_shifted:
    r = (2*outcome - 1) * w * T
```

### 2.4 乘 T 的 scale 理由

v5 的 token reward 规模 `log P_teacher ∈ [-log|V|, 0]`，序列累加 `Σ r_t ~ -T × avg_NLL`（量级 -10T 到 -5T）。
C 方案 `Σ r_t = R`（0 或 1）量级 << v5；直接用会让 advantage 信号极小，clip_adv 门控几乎总是被触发。
乘 T 后 `Σ r_t = R × T`（量级 T 到 0），与 v5 匹配，不需要改 clip_adv、norm_std 等下游超参。

---

## 3. 与 A、B 方案的数学对比

设 T 个 token，teacher log-prob 为 `l_t`，outcome 为 `R`：

| 方案 | r_t | Σ r_t | credit 差异 | outcome 断反馈 |
|---|---|---|---|---|
| v5 | `l_t` | `Σ l_t` | 有（按 l_t） | ❌ 无 |
| A (multiply) | `l_t × R` | `R × Σ l_t` | 有（按 l_t） | ✅（R=0 时全 0） |
| A (shifted 2R-1) | `l_t × (2R-1)` | `(2R-1) × Σ l_t` | 有（按 l_t） | ✅（错样本反向推） |
| B | `l*_t - l*_{t-1}` | `l*_T - l*_0` | 有（按 V 曲线斜率） | 间接（通过 y*） |
| **C (f2, +T)** | `R × softmax(l)_t × T` | `R × T` | **有（按 l_t 归一化）** | ✅（R=0 时全 0） |

C 和 A（multiply）的关键差别：
- A: `r_t = l_t × R` ⇒ R=1 时 r_t ∝ l_t（正值负值都可能，和 v5 一样）；
- C: `r_t = R × softmax(l)_t × T` ⇒ 恒 ≥ 0（softmax 权重非负）× R ∈ {0,1}，r_t ∈ [0, T]；
- C 和 A 的量级分布不同，但 credit 分配的 *相对顺序* 完全一致（softmax 是单调变换）；
- 核心差别在**尺度锚定**：C 锚定在 outcome（Σr=RT 可预测），A 锚定在 teacher（Σr=R·Σl 波动）。

---

## 4. 算力成本

| 方案 | 额外 teacher forward | 额外显存 | 实现复杂度 |
|---|---|---|---|
| v5 | 0 | 0 | baseline |
| A | 0 | 0 | 低（几行） |
| B | **+100%**（带 GT 的 forward） | +T×|V| for logits cache | 中（需 GT 拼 prompt） |
| **C** | **0** | 0 | 低（一次 softmax + 一次乘法） |

C 方案几乎零算力开销，**可以和 A 叠加**（A 当 fallback：若 C 的 softmax 权重过度集中，可用 α·A + (1-α)·C 混合）。

---

## 5. 数值稳定性

### 5.1 softmax 数值域

`log_prob ∈ [-30, 0]`（tokenizer 词表 50-150K，NLL 常常 5-20）；
softmax 前减 max，数值稳定：
```python
shifted = seq_logp - seq_logp.max()         # ∈ [-30, 0]
w = exp(shifted / τ) / exp(shifted / τ).sum()
```

### 5.2 temperature τ 的作用

- τ → 0+：w 退化为 one-hot（只有 max log-prob 的那个 token 拿满），credit 失衡；
- τ → ∞：w → 1/T（均匀），credit 失效；
- 建议 τ ∈ [0.5, 2.0]，默认 1.0；做消融 τ ∈ {0.5, 1.0, 2.0}。

### 5.3 T 的取值（序列长度）

`T` 是 **有效 response mask 下的 token 数**（不是 response 最大长度）。
若 response_mask 被 entropy_gate 过滤后变短，`T` 同步缩小，保持 `Σ r = R×T` 与 mask 内 token 数一致。

### 5.4 长序列 softmax 损失

T=10000 时，softmax 后大部分权重趋近于 0，真正有效的 credit 只在 top-k 个 token。
可选 **top-k mass preservation**：只保留 top-k 的权重，其余置 0 后重归一化，剪掉"尾巴噪声"。

---

## 6. 与现有 TASD pipeline 协同

### 6.1 与 entropy_gate 的协同

`ENTROPY_GATE` 过滤 mask 位置 → C 方案在 **mask=1 的 token 集合** 上做 softmax 归一化；
mask=0 的 token r_t=0（和现有逻辑一致）。

### 6.2 与 norm_adv_by_std 的协同

C 方案保持 `Σ r = RT`，group 内所有 rollout 的 Σ r 都是 RT（错样本 = 0），std 会很大（包含 0 和 T 两种极值）。
可能需要 `adv_std_floor` 保护（避免过分放大正确样本的 advantage）。

### 6.3 与 success_by_uid 的协同

如果一个 group 8 个 rollout 全错（success_rate=0），`R=0` 对所有 rollout 都成立 ⇒ 全部 r=0 ⇒ 全部 advantage 也 0（group_mean=0，std=0）⇒ 梯度归零。
此时和 A 方案 (multiply) 行为一致——该 group 本轮不参与 loss。需要 `filter_groups` 或 soft filter 覆盖这种情况。

### 6.4 与 repetition_penalty 的协同

若 R=0 则 penalty 失效（r 全 0 乘什么都是 0）。问题不大：repetition 的 penalty 核心是让答对样本别复读；答错样本根本不训，不需要 penalty。

---

## 7. 实验设计（v17 或 v15 可选）

### 7.1 对照矩阵（单数据集 bio，6 JOB）

| JOB | reward_type | f | τ | outcome_fusion_mode | 说明 |
|---|---|---|---|---|---|
| v17-v5ref | teacher_log_prob | — | — | none | 对照：复现 v5 |
| v17-Amul | teacher_log_prob | — | — | multiply | 对照：A 方案 |
| v17-Cf2 | **teacher_credit** | f2 (log-prob softmax) | 1.0 | — | **主推 C 方案** |
| v17-Cf1 | teacher_credit | f1 (NLL softmax) | 1.0 | — | 消融：相反的 credit 方向 |
| v17-Cf5 | teacher_credit | f5 (uniform) | — | — | 消融：等价纯 outcome RL |
| v17-Cτ0.5 | teacher_credit | f2 | 0.5 | — | 消融：更尖锐 credit |

### 7.2 关键观测指标

- `tasd/c_credit_top1_frac`：mean over rollouts of max(w_t)（若 > 0.5 说明过度集中）；
- `tasd/c_credit_entropy`：mean of entropy(w_t) / log(T)（0=集中，1=均匀）；
- `tasd/c_effective_tokens`：mean of `1 / Σ w_t²`（有效 credit 分配的 token 数）；
- `tasd/r_sum_per_seq_mean`：验证 `Σ r = RT`（sanity check）；
- 标准 metrics：acc / response_length / entropy / repetition_rate。

### 7.3 成功判据

- v17-Cf2 **peak ≥ v17-v5ref peak × 1.02**（相对 v5 有提升）
- v17-Cf2 **不崩塌**（last 20 steps 的 acc 不比 peak 低 50%）
- `c_credit_top1_frac` 稳定在 [0.05, 0.5]（既有 focus 又不过度集中）
- `c_effective_tokens` ≥ T/20（至少 5% 的 token 参与 credit）

### 7.4 失败模式
| 现象 | 诊断 | 处置 |
|---|---|---|
| Cf2 比 Amul 差很多 | softmax 过度集中 → credit top-k 个 token 饱和 | 降 τ 或加 top-k mass preservation |
| Cf2 response length 爆炸 | 长尾 token 分到 w=0 → 模型无 penalty 地扩写 | 加 length penalty 或切回 Amul |
| Cf2 早期 acc 上升快后崩塌 | success rollout 太少，R=0 主导，梯度归零 | 配合 B 方案 or teacher EMA 慢更新 |
| Cτ0.5 明显好于 Cτ1 | credit 需要更尖锐 | 继续扫 τ=0.3 |

---

## 8. 风险与边界

### 8.1 主要风险

1. **credit 不稳定 → reward 方差爆炸**
   softmax 在 log-prob 分布不均时会集中，少数 token 拿 90% credit。此时 advantage 基本等同于 "top-1 token × R"，梯度集中在 1-2 个 token 上。
   缓解：监控 `c_credit_top1_frac`，超过 0.5 时 τ ← τ × 1.5 自适应。

2. **全错 group → 梯度归零**
   见 §6.3。缓解：combined with `filter_groups` 或 soft filter。

3. **teacher 是 student EMA，C 方案用 teacher 做 credit**
   当 student 复读时 teacher 也在复读，teacher log-prob 对复读 token 反而给高分 → C 把 credit 分给复读 token → 正向强化复读。
   **这个是 C 方案最大的理论风险**。
   缓解：
   - (a) 用 base model（冻结 Qwen3-8B）做 credit，而非 student EMA；
   - (b) 用 outcome_shifted (2R-1)，让 R=0 时 credit 变负向惩罚（但会破坏 softmax 的非负性）；
   - (c) 叠加 B 方案的 conditional 过滤（只在 teacher 对 GT confidence > 阈值时启用 C）。

4. **开放式任务（无 GT）的 outcome 定义**
   若 R 由 reward model 打分（非 binary），C 方案依然可用（R ∈ [0,1] 即可），但梯度规模会和 R 分布强耦合。
   标注：先只在 MCQ + tooluse（有 binary outcome）验证，其他数据集不纳入 v17。

### 8.2 回滚
C 方案是新的 `reward_type`，和 v5 默认 `teacher_log_prob` 并列；
只要 `algorithm.tasd.reward_type ≠ teacher_credit`，C 方案代码完全不激活。
零回滚风险。

---

## 9. 代码骨架

### 9.1 新增 reward_type

**`verl/trainer/config/tasd_simple.yaml`**：
```yaml
reward_type: "teacher_log_prob"   # teacher_log_prob | teacher_prob | teacher_credit(new)

# C 方案专用参数
credit_weight_fn: "log_prob_softmax"   # log_prob_softmax | nll_softmax | entropy_softmax | uniform
credit_temperature: 1.0                  # softmax τ
credit_topk: 0                           # 0=不截断；>0=只保留 top-K 权重
credit_outcome_mode: "binary"            # binary(R∈{0,1}) | shifted(R∈{-1,1})
credit_scale_by_T: true                  # 是否乘 T 使 mean(r)=R
```

### 9.2 核心实现（`core_algos.py` 新增函数，~60 行）

```python
def compute_teacher_credit_reward(
    teacher_logp: torch.Tensor,          # (B, T) teacher log P(y_t | ctx)
    response_mask: torch.Tensor,         # (B, T) 有效 token mask
    outcome: torch.Tensor,               # (B,) sequence-level reward
    weight_fn: str = "log_prob_softmax",
    temperature: float = 1.0,
    topk: int = 0,
    outcome_mode: str = "binary",
    scale_by_T: bool = True,
) -> torch.Tensor:
    """
    r_t = R × w_t [× T], where w_t = f(teacher_t) / Σ f(teacher_τ)
    """
    B, T_full = teacher_logp.shape
    # 1. outcome 预处理
    if outcome_mode == "shifted":
        R = 2.0 * outcome - 1.0                        # (B,)
    else:
        R = outcome                                    # (B,)

    # 2. 权重函数
    if weight_fn == "log_prob_softmax":
        s = teacher_logp                               # (B, T)
    elif weight_fn == "nll_softmax":
        s = -teacher_logp
    elif weight_fn == "uniform":
        s = torch.zeros_like(teacher_logp)
    else:
        raise ValueError(weight_fn)

    # 3. 只在 mask=1 的位置做 softmax
    s_masked = s.masked_fill(response_mask == 0, float("-inf"))
    # 数值稳定：减 max（已由 softmax 内部处理）
    w = torch.softmax(s_masked / max(temperature, 1e-4), dim=-1)  # (B, T), Σ_t w=1

    # 4. 可选 top-k 截断
    if topk > 0:
        topk_vals, topk_idx = w.topk(k=min(topk, T_full), dim=-1)
        mask_topk = torch.zeros_like(w).scatter_(-1, topk_idx, 1.0)
        w = w * mask_topk
        w = w / (w.sum(dim=-1, keepdim=True) + 1e-8)

    # 5. credit 分配
    r = R.unsqueeze(-1) * w                            # (B, T)

    # 6. scale by T
    if scale_by_T:
        T_eff = response_mask.sum(dim=-1, keepdim=True).clamp(min=1.0)
        r = r * T_eff

    # 7. 清零无效位置
    r = r * response_mask.to(r.dtype)
    return r
```

### 9.3 dp_actor.py / ray_trainer.py dispatch

在现有 `reward_type` 分支处加一个 `elif reward_type == "teacher_credit":`：
```python
elif reward_type == "teacher_credit":
    from verl.trainer.ppo.core_algos import compute_teacher_credit_reward
    token_reward = compute_teacher_credit_reward(
        teacher_logp=teacher_logp,
        response_mask=response_mask,
        outcome=outcome_score,  # (B,) 0/1 from reward_tensor
        weight_fn=tasd_cfg.get("credit_weight_fn", "log_prob_softmax"),
        temperature=tasd_cfg.get("credit_temperature", 1.0),
        topk=tasd_cfg.get("credit_topk", 0),
        outcome_mode=tasd_cfg.get("credit_outcome_mode", "binary"),
        scale_by_T=tasd_cfg.get("credit_scale_by_T", True),
    )
```

### 9.4 总行数预估

| 文件 | +行 | 说明 |
|---|---|---|
| `verl/trainer/config/tasd_simple.yaml` | +10 | 5 个新字段 |
| `verl/workers/config/actor.py` | +5 | SelfDistillationConfig 新字段 |
| `verl/trainer/ppo/core_algos.py` | +60 | `compute_teacher_credit_reward` |
| `verl/workers/actor/dp_actor.py` 或 `ray_trainer.py` | +15 | dispatch 分支 |
| `nebula_scripts/tasd_simple/tasd_simple_parametric.sh` | +10 | env → hydra override |
| `nebula_scripts/submit_tasd_v17_credit_sweep.sh` | +200 | 新建 sweep |
| **总计** | **~300** | |

比 B 方案轻一半（B 需要 GT-conditional forward + 两次 teacher pass）。

---

## 10. 路线图整合

```
v13 (in flight)  v5 baseline 复现
   ↓ if peak ≥ 0.65
v14 = v13 + 最小 error_pool              (group-shared + format-only)
   ↓
┌──── v15 = v14 + A方案 outcome_fusion    (已实现 in simplifed)
│ or
└──── v15' = v14 + C方案 teacher_credit   (本文档)

   ↓ if A 成功 or C 成功
v16 = v15 + B方案 outcome-conditional teacher
   (B 方案验证终极形态)
```

**建议选择 C（而非 A）作 v15'**：
- A 只是"整体 scale"，C 是"credit assignment"，后者信息量更大；
- 算力同为零开销；
- C 的风险（teacher EMA 复读时 credit 分错）可通过 `credit_weight_fn=nll_softmax`（f1）或 `base_model` 作 credit 来源缓解。

---

## 11. 结论

C 方案是在 A（scale only）和 B（telescoping，算力翻倍）之间的**中间甜点**：

| 维度 | A | **C** | B |
|---|---|---|---|
| Σ r 无偏对齐 outcome | ❌ Σ=R·Σlog_p | **✅ Σ=RT** | ✅ Σ=V_T-V_0 |
| credit assignment | ⚠️ 均匀 scale | **✅ 按 teacher 权重** | ✅ 按 V 曲线斜率 |
| 需要 GT | ❌ 不需要 | **❌ 不需要** | ✅ 需要 |
| 额外 teacher forward | 0 | **0** | +100% |
| 实现复杂度 | 低 | **低** | 中 |
| 理论风险 | outcome=0 信号单一 | **teacher 复读时 credit 失准** | teacher 对 GT 不稳定 |

下一步：
1. 等 v13 复现 0.68；
2. v14 最小 error_pool 上线验证；
3. v15 选 A or C（一周内选一个，另一个作对照）；
4. v16 叠 B（若前者有明显 gap 需要更精细的 credit）。

---

_草稿版本：v0.1；待 v13/v14 结果再修订 §7.3 的成功判据阈值。_
