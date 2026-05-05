# TASD B 方案设计文档：Outcome-Conditional Teacher Reward

> 版本：v0.2（草稿，未开工）
> 所在分支：v5-baseline（基于 52149bd）
> 路线图定位：v16（v14 error_pool + v15 A/C 验证通过后启动）
> 配套文档：
> - [v14 最小 error_pool 设计](./TASD%20v14%20%E8%AE%BE%E8%AE%A1%E6%96%87%E6%A1%A3-minimal%20error_pool%20on%20v5.md)
> - [C 方案 outcome-based teacher credit assignment](./TASD%20C%E6%96%B9%E6%A1%88%E8%AE%BE%E8%AE%A1%E6%96%87%E6%A1%A3-outcome-based%20teacher%20credit%20assignment.md)

> **v0.2 更新说明**：B 方案最早在 simplifed 分支（v12=A、v13=B 的路线图下）设计。因 HEAD 代码漂移问题，路线图已切换为 v5-baseline → v13(v5复现) → v14(error_pool) → v15(A 或 C) → v16(B)。本次整理对齐新路线图，并补充 C 方案对比与协同。

## 1. 背景与动机

### 1.1 当前 TASD reward 的结构性缺陷

当前 [compute_tasd_token_rewards](file:///Users/awesome_jimmy/lazada/SDPO/verl/trainer/ppo/core_algos.py#L2324) 的 per-token reward 公式为：

```
r_t = teacher_log_prob(y_t | prompt, y_{<t})         # teacher_log_prob
r_t = teacher_prob(y_t | prompt, y_{<t})             # teacher_prob
```

这种设计有两个根本问题：

1. **真实 outcome reward 从未进入梯度**。`reward_tensor` 里承载的答对/答错信号
   只被用于：
   - 判定 success / failure → 影响 `reference_answer` 与 `error_pool` 的选择
   - 进入 teacher context 作为提示
   
   但 **梯度直接来源只有 teacher_log_prob**。极端情况下，student 答案全错
   (outcome=0) 但 teacher_log_prob 很高 → reward 依旧很高 → 鼓励错误行为。

2. **teacher 作为 student 的 EMA 副本，对复读 token 会给高概率** → 形成
   "复读 → teacher 追随 → reward 自洽 → 鼓励更多复读" 的死循环。
   这是 v7-v10 bio 数据集 rlen 爆炸到 8192 的机制根因。

### 1.2 现有方案对比

| 方案 | 公式 | 额外 forward | Σr_t 无偏 | 需要 GT | 治复读 | 实现成本 |
|---|---|---|---|---|---|---|
| v5 `teacher_log_prob` | `r_t = log P_teacher(y_t)` | 0 | ❌ Σ=Σ log_p | ❌ | ❌ | — |
| **A** `outcome fusion` | `r_t = log P_teacher(y_t) × R` | 0 | ❌ Σ=R·Σ log_p | ❌ | ✅ (R=0 全断) | ★ |
| **C** `teacher credit assign` | `r_t = R × softmax(log_p)_t × T` | 0 | ✅ Σ=R·T | ❌ | ✅ (R=0 全断) | ★ |
| **B** `outcome-conditional` | `r_t = V_t − V_{t-1}`, `V_t=log P_teacher(y*\|y_{≤t})` | ×T→~1.1-2× 原 forward | ✅ Σ=V_T−V_0 | ✅ | ✅ (复读时 V 不变) | ★★★ |

三个方向的本质差别：
- **A 方案**：outcome 作序列级 scale，均匀缩放所有 token reward，无 credit 差异化；
- **C 方案**：outcome 作 token reward 总量（蛋糕），teacher softmax 作分配权重（分蛋糕规则）；保持 Σr = R·T 无偏；零额外算力；
- **B 方案**：token reward 是 teacher 对 GT 信念的边际增量，最严格的 Bellman credit；需要 GT 与额外 teacher forward。

B 与 A/C 不是替代关系，而是精度递进。实际落地顺序见路线图（§7）。

## 2. B 方案核心公式

### 2.1 定义 teacher belief（teacher 对正确答案的信念）

给定原始 prompt `x` 和 GT 答案 `y*`，定义 teacher 在看到 student 部分轨迹
`y_{≤t}` 后对"最终答案正确"的信念：

```
V_t := log P_teacher(y* | x, y_{≤t})
```

这个量的解读：
- `V_0 = log P_teacher(y* | x)` : 不看 student 任何输出时 teacher 对正确的先验信心
- `V_T = log P_teacher(y* | x, y_{1..T})` : 看完 student 全部响应后 teacher 对正确的后验信心

### 2.2 Per-token reward = 边际信念增量

```
r_t := V_t − V_{t-1}
     = log P_teacher(y* | x, y_{≤t}) − log P_teacher(y* | x, y_{<t})
```

**直观含义**：加上这个 token 后，teacher 对"最终答对"的信心增加了多少。

**三种典型场景**：

| 场景 | Δ log P_teacher | r_t | 解读 |
|---|---|---|---|
| y_t 是关键推理步骤 | +3.0 | 正 | teacher 被说服，credit 高 |
| y_t 是无意义填充 | ≈0 | ≈0 | 信念不变，无 credit |
| y_t 把 teacher 带偏 | −2.0 | 负 | 惩罚这个 token |
| 复读已有内容 | ≈0 | ≈0 | teacher 信念不变 → **复读自动归零** |

### 2.3 Telescoping 性质（与 outcome 的严格一致性）

所有 token 的 reward 之和具有 **telescoping 性质**：

```
Σ_{t=1}^{T} r_t = V_T − V_0
                = log P_teacher(y* | x, y_{1..T}) − log P_teacher(y* | x)
```

这意味着 **per-token reward 的总和 = teacher 对 student 整条响应的 "outcome-style"
信心增量**。对比 GRPO：
- GRPO: Σ r_t = outcome (0/1)，离散，粒度粗
- B 方案: Σ r_t = Δ log-belief，连续，且按 token 精细分配

这是严格 Bellman 意义下的 per-token advantage，从第一原理上保证了
"token 级 credit 加总 = 序列级 outcome 信号"的一致性。

## 3. 对不同数据集的具体实现

### 3.1 MCQ (bio/sciknoweval)

GT 答案是 `A/B/C/D` 一个字母。teacher belief 计算：

```python
def compute_mcq_belief(teacher_model, x, y_prefix, gt_letter):
    # 拼接: prompt + student 前 t 个 token + "<answer>"
    # 让 teacher 在 <answer> 后预测下一个 token
    inputs = x + y_prefix + "<answer>"
    with torch.no_grad():
        logits = teacher_model(inputs).logits[:, -1, :]  # (B, vocab)
    # 只看 A/B/C/D 四个 token 的 logit
    letter_ids = [tok("A"), tok("B"), tok("C"), tok("D")]
    letter_logits = logits[:, letter_ids]               # (B, 4)
    letter_log_probs = F.log_softmax(letter_logits, dim=-1)
    return letter_log_probs[:, gt_idx]                   # (B,)
```

对整条 `y_1..y_T`，需要 T+1 个位置上各做一次这样的查询
（V_0, V_1, ..., V_T）。

### 3.2 Tooluse

没有单 token GT，但有 GT action sequence `a*`。两种做法：

**做法 1（完整 GT 续写概率）**：
```
V_t := log P_teacher(a* | x, y_{≤t})
     = Σ_{k} log P_teacher(a*_k | x, y_{≤t}, a*_{<k})
```
一次 forward 对 `(x, y_{≤t}, a*)` 拼接输入 → 取 GT 位置的 log_prob 求和。

**做法 2（GT 首 token 近似）**：
```
V_t := log P_teacher(a*_0 | x, y_{≤t})
```
只看 GT 动作的第一个 token 概率（比如 "Action: search" 中的 "search"）。
便宜但信号稀疏。

### 3.3 Math / 开放式答案

GT 是数学表达式或短字符串（如 `"42"`, `"\\frac{3}{4}"`），按做法 1 处理。

## 4. 算力与工程可行性

### 4.1 额外 forward 开销

朴素实现：每个位置 t 都要一次 teacher forward → `(B × T)` 次。**不可行**。

**批量并行化关键技巧**：

将 T 个不同前缀 `(x, y_{≤0}), (x, y_{≤1}), ..., (x, y_{≤T})` 打包成
**一个** (T+1) × L 的 input_ids，做 **一次** teacher forward：

```python
# 原始一次 forward 只算 y_T 的 log_prob
# 现在一次 forward 算 V_0..V_T (T+1 个值)
teacher_input = x + y_1 + y_2 + ... + y_T + "<answer>"  # 长度 L
logits = teacher(teacher_input).logits                    # (1, L, vocab)

# V_t 对应 logits[:, len(x) + t + len("<answer>") - 1, :]
# 即在 "看完 y_{≤t} + <answer>" 位置的 next-token 分布
```

对 MCQ，这几乎等价于 **原有一次 teacher forward 的 2×**（多拼一个 `<answer>`
模板并读取 T+1 个位置的 logit）。

对 tooluse 做法 1，需要 T 个独立的拼接 `x + y_{≤t} + a*`，可按 micro-batch
打包。最坏开销 ~ T 倍原 forward。

### 4.2 缓存优化（进阶）

MCQ 场景下，teacher 对 `x + y_1..y_T` 做一次 forward 得到的 hidden states
已经包含了所有前缀位置的信息。可以：

1. 一次 forward 拿到 `(x + y + "<answer>")` 的 hidden_states
2. 对每个 t，取 `hidden_states[len(x) + t + <answer-offset>]`  → 4 个字母 logit
3. 无需重复 forward

此优化把额外开销从 2× 压到 ~1.1×，与原 teacher forward 算力相当。

### 4.3 GT 泄漏风险

**核心顾虑**：teacher 看到 `y_{≤t} + "<answer>"` 然后预测 A/B/C/D → 这不就是
直接做 MCQ 任务吗？GT 会不会以某种方式泄漏回 student？

**答案**：不会。teacher 在这里只是 **评估者**，不参与 policy。student 看到的
teacher context 仍然是原始的 `x` 与 `reference_answer`（按 v11 门控已脱敏）。
V_t 只进入 reward 计算，不进入 student 的 prompt。

与 A 方案 `outcome × teacher credit` 相比，B 方案 **暴露给 student 的信息量**
完全一致（都只通过 reward 梯度反向传播），只是 reward 的归因精度更高。

## 5. 数值稳定性

### 5.1 log_prob 量级

`log P_teacher(y*)` 在 MCQ 上量级 ~ `[-4, 0]`（4 选项先验 log(1/4) ≈ -1.4），
差分 `r_t = V_t - V_{t-1}` 量级 ~ `[-2, +2]`，数值健康。

Tooluse 做法 1 的 `log P_teacher(a*)` 是完整 GT 序列的 log-likelihood，量级
`[-100, 0]` 以上；差分量级也相应更大。建议对 tooluse 做 **clip**：

```
r_t = (V_t - V_{t-1}).clamp(min=-C, max=C)   # C 取 5.0 ~ 10.0
```

### 5.2 telescoping 守恒

clamp 会破坏 `Σ r_t = V_T - V_0` 的严格等式。对此可选两种策略：

- **策略 A（保精度）**：不 clamp，靠 advantage 侧的 `clip_adv` 控爆炸
- **策略 B（保稳定）**：clamp 后归一化：`r_t *= (V_T - V_0) / Σ r_t_clamped`

推荐策略 A + `clip_adv=2.0`（TASD 现有默认），已经过 v5-v10 验证稳健。

## 6. 与 A / C 方案的协同

B 方案不排斥 A、C 方案，三者可叠加：

```
r_t^A = log P_teacher(y_t)               × R               # A: outcome 均匀 scale
r_t^C = R × softmax(log P_teacher)_t × T                    # C: outcome 作总量, teacher 作分配器
r_t^B = log P_teacher(y*|y_{≤t}) − log P_teacher(y*|y_{<t}) # B: teacher 对 GT 信念的边际增量

r_t^{B+C} = α · r_t^B + (1-α) · r_t^C
```

**B+C 组合的意义**：
- C 方案在错样本（R=0）上整条序列 reward=0，无任何 token 级学习信号；
- B 方案在错样本上仍有信号（V_t 可能下降），但需要额外 teacher forward；
- B+C 可以在 R=1 样本上用 C 的零开销 credit，在 R=0 样本上激活 B 的边际信念信号，实现"分段采信"。

实践建议（对齐 v5-baseline 路线图）：
- **v14**：最小 error_pool（已设计，未实现）
- **v15**：A 或 C 二选一（零算力方案先行验证）
- **v16（本方案）**：B 独立 reward_type，与 v15 胜者对照
- **v17（可选）**：B+C 融合，在 α ∈ {0.3, 0.5, 0.7} 扫 best

## 7. 实验设计（v16）

假设 v15 的 A 或 C 方案已跑通且相对 v13 baseline 有正向收益，v16 B 方案矩阵：

| Job | 数据集 | reward_type | GT 使用 | 备注 |
|---|---|---|---|---|
| B1 | bio | `conditional_mcq`（V_t-V_{t-1}, 4 字母版） | 显式 | 基准 |
| B2 | bio | `conditional_mcq` + `clip_r=5.0` | 显式 | 稳定性消融 |
| B3 | tooluse | `conditional_tooluse_fast`（做法 2：GT 首 token） | 显式 | 便宜版 |
| B4 | tooluse | `conditional_tooluse_full`（做法 1：完整 GT） | 显式 | 精确版 |

**成功判据**：
- B1 peak ≥ max(v13, v15) peak × 1.02（相对前序最强基线有提升）
- V_t 时序曲线单调性：mean(V_T - V_0) 在 correct rollout 上 > 0，在 wrong rollout 上 < 0（验证 GT 信念信号有效）
- 额外 teacher forward 开销 ≤ 原 teacher forward × 1.5（否则必须用 §4.2 缓存优化）

## 8. 风险与回退

### 8.1 已知风险

1. **teacher 不确定时 V_t 噪声大**：teacher_entropy_gate（已存在）可作为
   二级 gate，当 teacher 自身对 `y*` 信念不足时屏蔽该样本。
2. **GT 缺失情境**（SR=0 的 group）：B 方案要求 GT，对 GT fallback 机制
   依赖度较高；失败时退化为当前 teacher_log_prob。
3. **完整 GT 续写概率受 GT 长度影响**：tooluse 做法 1 的 V_t 量级随 |a*|
   线性变化，需按 `|a*|` 归一化：`V_t := V_t / len(a*)`。

### 8.2 回退策略

B 方案作为独立 `reward_type="teacher_conditional"` 加入 `compute_tasd_token_rewards`，
默认关闭。若实验失败可直接切回 `teacher_log_prob`（v5 baseline）。

## 9. 代码实现骨架（伪代码）

```python
# core_algos.py 新增 reward_type
elif reward_type == "teacher_conditional":
    # student_log_probs / teacher_log_probs 已在外层准备好
    # 需要额外传入 teacher_beliefs: (B, T+1) = [V_0, V_1, ..., V_T]
    assert teacher_beliefs is not None
    V_curr = teacher_beliefs[:, 1:]       # V_1..V_T
    V_prev = teacher_beliefs[:, :-1]      # V_0..V_{T-1}
    reward = (V_curr - V_prev).clamp(min=-clip_r, max=clip_r)
```

```python
# ray_trainer.py 新增 teacher_beliefs 计算
if tasd_reward_type == "teacher_conditional":
    gt_answers = batch.non_tensor_batch["ground_truth"]    # 解析 GT
    teacher_beliefs = compute_teacher_beliefs(              # 新增函数
        teacher_wg=self.actor_rollout_wg,
        prompts=batch.batch["prompts"],
        responses=batch.batch["responses"],
        gt_answers=gt_answers,
        data_source=batch.non_tensor_batch["data_source"],
        belief_mode="mcq_letter" if ds in ("sciknoweval", "mcq")
                     else "tooluse_full",
    )
    teacher_fwd_batch.batch["teacher_beliefs"] = teacher_beliefs
```

## 10. 结论

B 方案在第一原理上最干净：
- `Σ_t r_t = V_T - V_0` 保证 token 级 credit 与 outcome 严格一致
- 复读 token 自动归零（teacher 信念不变）
- 不依赖启发式超参（无需 entropy_gate / fusion_alpha）

唯一缺点是额外 teacher forward 算力（~1.1-2× 原开销，缓存优化后 ≤ 1.5×）。工程上可控。

**路线图位置（v5-baseline）**：
```
v13 (in flight) v5 复现 baseline
  ↓ peak ≥ 0.65
v14           最小 error_pool（见配套文档）
  ↓
v15           A 方案 或 C 方案（零算力先行）
  ↓ 验证 outcome-awareness 有效性
v16 (本方案)  B 方案 outcome-conditional teacher
  ↓ 若 token 级精细归因有额外收益
v17 (可选)    B+C 融合 (§6)
```

**关键前置条件**：
1. v13 peak ≥ 0.65（v5 baseline 跑通，否则 teacher 本身质量不够）
2. v14 error_pool 不引入 advantage 偏倚（否则 B 方案信念差分也会被污染）
3. v15 的 A 或 C 已验证"outcome 进入梯度"路径有效（B 是同一思路的精细化版本）
