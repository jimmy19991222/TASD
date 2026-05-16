# Self-Teacher Advantage with Bidirectional Baselines

## 一、动机：GRPO 的 Credit Assignment 瓶颈

GRPO 在 RLVR 场景下的训练流程：

```
prompt x → 采样 G 条 response → 环境打分 r_i ∈ {0,1} → 归一化得到 A_i → 广播到所有 token
```

核心瓶颈：**序列级 advantage 被 uniform 广播到所有 token**。一条正确的 response 里，每个 token 获得相同的正 advantage——无论它是关键推理步骤还是无意义的填充词。反之，一条错误 response 里的所有 token 也被均匀惩罚，包括那些本身正确的前置步骤。

FIPO 和 SDPO 分别从不同角度尝试解决这个问题。本文提出一种更直接的路径：**用 self-teacher 直接产生 token-level advantage，完全绕过序列级归一化**。

---

## 二、核心思想

### 2.1 Self-Teacher 的角色

给定一条 response y 及其环境反馈（如组内成功 response 作为参考），将当前模型 conditioned on feedback 作为 self-teacher：

```
Student:  π_θ(· | x, y_{<t})          — 只看 prompt 和已生成 token
Teacher:  π_θ(· | x, feedback, y_{<t}) — 额外看到 feedback（正确答案/环境输出）
```

Teacher 拥有比 student 更多的信息，因此能够回溯判断每个 token 的质量。

### 2.2 统一框架：A_t = Q_t - V_t

所有基于 self-teacher 的 token-level advantage 都共享同一个公式，区别仅在于 baseline V_t 的选择：

```
Q_t = log π_teacher(y_t | x, feedback, y_{<t})   ← 所有方法共用

V_t = ???                                         ← 各方法的差异所在

A_t = Q_t - V_t                                   ← token-level advantage
```

**直觉解释**：

- Q_t = teacher 对 student 实际选择 y_t 的评分
- V_t = "合理预期"——在这个位置，teacher 应该给多少分
- A_t = 实际评分 vs 预期评分 = 超出（或低于）预期的程度

### 2.3 关键性质：无需归一化

A_t 的取值自然有界：

```
Q_t ∈ (-∞, 0]        — log probability
V_t ∈ (-∞, 0]        — log probabilities 的函数
A_t = Q_t - V_t       — 有界差值
```

由于 y_t 通常是 student 分布中的高概率 token（student 自己选的），Q_t 通常 ≥ V_t，即 A_t 通常略正。只有当 teacher 明确不认可 student 的选择时（teacher 更偏好其他 token），A_t 才会为负。

**这意味着不需要 z-score 归一化来制造正负信号——正负方向由 teacher 的评估天然决定。**

---

## 三、Baseline 的选择：横向 vs 纵向

A_t = Q_t - V_t 中 V_t 的选择是本方法的核心设计。我们提出两种互补的 baseline，分别回答不同的问题：

### 3.1 横向 Baseline：Cross-Entropy Baseline（V_CE）

```
V_CE_t = Σ_{k=1}^{K} π_student(a_k | x, y_{<t}) · log π_teacher(a_k | x, feedback, y_{<t})
```

其中 {a_1, ..., a_K} 是 student 分布的 top-K 候选 token。

**回答的问题**："在位置 t，student 所有可能的选择中，teacher 平均会给多少分？"

**含义**：student 选的 token 比它的平均选择好吗？

**维度**：横向对比——同一位置，不同 token 之间的比较。

### 3.2 纵向 Baseline：Causal EMA Baseline（V_EMA）

```
V_EMA_t = α · V_EMA_{t-1} + (1-α) · Q_{t-1}    （α 通常取 0.9）
V_EMA_0 = Q_0
```

**回答的问题**："根据前面位置的 teacher 评分趋势，位置 t 应该得多少分？"

**含义**：这个 token 延续了前面的质量轨迹吗？还是突然偏离了？

**维度**：纵向对比——同一 response 内，不同位置之间的比较。

### 3.3 为什么两者需要互补：一个具体例子

```
Prompt: "证明 2+2=5"（错误的命题）
Student response: "Let's see... 2+2=4... wait no... actually 2+2=5... therefore QED."
Teacher context: 看到这个 response + feedback（这是错的）

逐位置的 teacher 评分（Q_t）和两种 baseline：

  token       Q_t     V_CE_t    V_EMA_t    含义
  ──────      ───     ──────    ───────    ────
  "Let's"    -0.5     -0.52      -0.50      正常开场
  "see"      -0.4     -0.42      -0.45      正常
  "2+2=4"    -0.3     -0.35      -0.40      对了！teacher 认可
  "wait"     -0.4     -0.42      -0.38      开始犹豫
  "no"       -0.5     -0.52      -0.42      否定了正确答案
  "actually" -0.6     -0.65      -0.46      teacher 不认可
  "2+2=5"    -3.8     -1.50      -0.65      大错！
  "therefore" -2.1    -1.20      -0.85      被前面污染
  "QED"      -3.0     -1.80      -1.10      强行结论
```

**关键看 "no" 这个 token（学生否定了正确的 "2+2=4"）：**

```
用 V_CE（横向 baseline）：
  A = Q - V_CE = -0.5 - (-0.52) = +0.02  ← 中性
  → teacher 在 "no" 位置对 student 的所有可能选择评分都差不多
  → V_CE 看不出问题 ❌

用 V_EMA（纵向 baseline）：
  A = Q - V_EMA = -0.5 - (-0.42) = -0.08  ← 负！
  → 前一个 token "2+2=4" 是好的（Q=-0.3）
  → 突然到 "no"（Q=-0.5）→ 质量下降了
  → EMA 捕捉到了这个转折 ✅
```

**反过来，V_CE 有一个 V_EMA 做不到的事——全组失败时的安全退化：**

```
当 teacher ≈ student（全组失败，teacher 无额外信息）：

V_CE baseline:
  V_CE_t = E_student[Q_t] ≈ E_student[log π_student(·)]
  Q_t ≈ log π_student(y_t)
  → A_t ≈ log π_student(y_t) - E[log π_student(·)] ≈ 0
  → 安全退化 ✅

V_EMA baseline:
  V_EMA 在积累 Q_t（本身是 teacher ≈ student 时的 noisy log prob）
  → V_EMA 在积累噪声
  → A_t = Q_t - EMA(noisy) ≠ 0（可能有随机正负）
  → 不如 V_CE 稳定 ⚠️
```

**结论：两者互补。** V_CE 保证跨 token 的正确 baseline 和安全退化，V_EMA 捕捉轨迹中的质量转折和 context pollution 的校正。

### 3.4 融合方案

```
V_t = β · V_CE_t + (1-β) · V_EMA_t

A_t = Q_t - V_t
```

β 的选择：
- **β = 1.0（纯 V_CE）**：最安全，variance reduction 理论最优。丢失时序信号。
- **β = 0.7（V_CE 主导）**：推荐默认值。V_CE 提供主要 baseline，V_EMA 辅助捕捉转折。
- **β = 0.5（等权）**：信号最强。如果 teacher 质量高可以尝试。
- **β = 0（纯 V_EMA）**：不推荐单独使用，全组失败时不稳定。

---

## 四、与 GRPO / FIPO / SDPO 的区别

### 4.1 信号流对比

```
GRPO:
  环境 reward r_i → 序列归一化 → A_i（seq-level）→ 广播 → A_{i,t} = A_i

FIPO:
  环境 reward r_i → 序列归一化 → A_i（seq-level）
  Policy shift Δlogp → 前向累积 FutureKL_t → f_t
  → A_{i,t} = A_i × f_t

SDPO:
  Teacher forward → KL(student ‖ teacher) → 直接作为 loss minimize
  隐式 advantage = log π_teacher(y_t) - log π_student(y_t)

本方法:
  Teacher forward → Q_t
  Bidirectional baselines → V_t = β·V_CE + (1-β)·V_EMA
  → A_t = Q_t - V_t → 直接用于 policy gradient
```

### 4.2 核心区别

| 维度 | GRPO | FIPO | SDPO | 本方法 |
|------|------|------|------|--------|
| Reward 来源 | 环境 | 环境 | Teacher (隐式) | Teacher (显式) |
| Advantage 方向 | 由环境 reward 决定 | 由环境 reward 决定 | 由 teacher-student 差异决定 | 由 teacher 评估决定 |
| Advantage 粒度 | 序列级（uniform） | Token 级（前向调制） | Token 级（logit-level） | Token 级（双向 baseline） |
| Baseline 选择 | group mean (z-score) | 无（继承 GRPO） | log π_student(y_t) | β·V_CE + (1-β)·V_EMA |
| 是否需要归一化 | 是（z-score） | 是（继承 GRPO） | 否（KL 自带 scale） | 否（baseline 自带 scale） |
| 环境 reward 的角色 | 直接决定 advantage | 直接决定 advantage | 不需要 | 构建 teacher context |
| 全组失败时的行为 | Advantage collapse | Advantage collapse | A ≈ 0（安全） | A ≈ 0（安全） |
| 处理 context pollution | ❌ | ❌ | ❌ | ✅（V_EMA） |
| 安全退化 | ❌ | ❌ | ✅ | ✅（V_CE） |

### 4.3 Reward 在本方法中的角色

环境 reward 不再直接参与 advantage 计算，而是**控制 teacher context 的构建**：

```
r_i = 1（成功）:
  teacher context = [prompt, "Your answer was correct.", response_i]
  → teacher 确认 response → Q_t 高 → A_t 正

r_i = 0（失败）+ 组内有成功 response y_best:
  teacher context = [prompt, "Correct solution:\n{y_best}\n\nYour failed attempt:", response_i]
  → teacher 在错误位置不认可 → Q_t 低 → A_t 负

r_i = 0（失败）+ 组内全失败:
  teacher context = [prompt, response_i]
  → teacher ≈ student → Q_t ≈ V_t → A_t ≈ 0 → 不更新（安全退化）
```

Reward 的信息通过 context 影响 teacher 的 conditional distribution，再由 teacher 决定在哪些 token 上给出正/负信号。这比直接把 scalar reward 广播给所有 token 的信息利用效率更高。

---

## 五、为什么不需要 Token Filter

### 5.1 回顾：为什么之前需要 filter

在 z-score 归一化的 TASD 中：

```
raw reward 全 ≈ 0（teacher ≈ student，无信号）
         ↓
z-score: A_t = (reward_t - mean) / std
         ↓
mean ≈ 0, std ≈ 0.001
         ↓
A_t = 微小差异 / 极小 std = 被放大数千倍
         ↓
噪声变成"强信号" → 需要 filter 抑制噪声
```

**归一化强制制造信号**——即使原始数据没有信号，输出也一定有正有负。这是 filter 存在的根本原因。

### 5.2 为什么新方案不需要

A_t = Q_t - V_t 有三层内建的自动过滤：

**第一层：无信号时自动静默（V_CE 的贡献）**

当 teacher ≈ student（全组失败，teacher 无额外信息）：
- Q_t ≈ log π_student(y_t)
- V_CE_t ≈ E_student[log π_student(·)]
- A_t ≈ log π_student(y_t) - E[log π_student(·)] ≈ 微小正值
- 所有位置的 A_t 几乎相同 → mean-centering 后 ≈ 0

不需要 filter，信号自动消失。

**第二层：无聊 token 自动忽略（V_CE 的贡献）**

大多数 token（格式符号、连接词）的位置，teacher 和 student 的分布几乎一致，无论 teacher context 里有什么：
- π_teacher("，" | ...) ≈ π_student("，" | ...)
- → 这些位置 Q_t ≈ V_CE_t → A_t ≈ 0

只有在关键决策点（答案选择、推理转折），teacher 和 student 才会产生分歧。
→ A_t 天然是 sparse 的。

**第三层：转折检测与 context pollution 校正（V_EMA 的贡献）**

如果一条 response 开头正确、中间出错、后面被污染：
- V_EMA 在前面的正确部分保持高水平
- 错误 token 出现 → Q 骤降 → A = Q - V_EMA 强负
- 被污染的后续 token → Q 仍然低，但 V_EMA 也已降低 → A ≈ 0
- → 只有关键错误点被惩罚，无辜 token 不被误伤

### 5.3 唯一建议保留的保护：Advantage Clipping

```python
A_t = clamp(Q_t - V_t, -C, C)  # C = 5.0
```

这不是 filter（不会将信号设为 0），只是数值保护——防止极端情况下 teacher 在某个位置给出极低概率导致的数值溢出。类似 FIPO 对 influence weight 的 `[1-ε, 1+ε]` 限制。

---

## 六、Baseline 的理论最优性

### 6.1 RL 中的 Baseline 理论

在 policy gradient 中，advantage 的通用形式是 A_t = Q_t - B_t。B_t 的选择直接影响梯度的 variance。经典结果（Greensmith, Bartlett & Baxter, 2004）表明：

```
使 Var(∇log π · (Q_t - B_t)) 最小的 B_t* 是：

  B_t* = E[Q_t · ||∇log π||²] / E[||∇log π||²]

近似地，当 ||∇log π||² 对同一 state 下不同 action 变化不大时：

  B_t* ≈ E[Q_t | s_t]  ← 给定 state 的 Q 的条件期望
```

### 6.2 V_CE 是最优 baseline 的近似

```
E[Q_t | s_t] = Σ_a π(a | s_t) · Q(a, s_t)

其中 s_t = (x, y_{<t}) 是位置 t 的 state
         π(a | s_t) 是 student 选择 token a 的概率
         Q(a, s_t) 是 teacher 对 token a 的评分

我们的 V_CE_t = Σ_{k=1}^K π_student(a_k | x, y_{<t}) · log π_teacher(a_k | x, f, y_{<t})
```

**对比发现：V_CE_t 和 E[Q_t | s_t] 在数学形式上完全一样！**

唯一的区别是 E[Q_t | s_t] 对整个词表求和，而 V_CE_t 只对 top-K 近似。当 K 足够大（如 K=100，覆盖 student 绝大部分概率质量），V_CE_t 就是最优 baseline 的高精度近似。

### 6.3 V_EMA 的作用

V_EMA 不追求 variance minimization，而是解决一个不同维度的问题：**位置间的质量漂移检测**。

```
V_CE 保证：在每个位置上，advantage 的方差最小（理论最优）
V_EMA 保证：当 response 质量发生转折时，advantage 能捕捉到变化（实践有效）

两者融合 = variance optimal + context aware
```

---

## 七、计算开销分析

### 7.1 已有的计算（TASD/SDPO 标准 pipeline）

| 计算步骤 | 输出 | 已有？ |
|----------|------|--------|
| Student 生成 response | y_{1:T} | ✅ |
| Student forward（计算 log probs） | student_log_probs, student_topk | ✅ |
| Teacher forward（conditioned on feedback） | teacher_log_probs, teacher_topk | ✅ |

### 7.2 新增的计算

| 计算步骤 | 操作 | 开销 |
|----------|------|------|
| Teacher logprobs at student top-K | gather 操作 | 可忽略 |
| V_CE_t = dot(student_probs, teacher_at_student_topk) | (B,T,K) → (B,T) | 可忽略 |
| V_EMA_t = EMA(Q_{<t}) | (B,T) 上的循环 | 可忽略 |
| V_t = β·V_CE + (1-β)·V_EMA | 逐元素加权 | 可忽略 |
| A_t = Q_t - V_t | 逐元素减法 | 可忽略 |
| Mean-centering | per-response mean | 可忽略 |

**额外开销 ≈ 0**。所有信息已在标准 pipeline 中计算完毕。

唯一需要确认的是：teacher forward 是否保存了对 student top-K token 的 log-prob。如果当前只保存了 teacher 自己的 top-K 和对 generated token 的 log-prob，需要额外保存 teacher 对 student top-K 位置的值。这只是一次 `logits.gather(dim=-1, index=student_topk_indices)` 操作，在 forward pass 内完成，不引入额外的 forward。

---

## 八、完整算法

```
Algorithm: Self-Teacher Advantage with Bidirectional Baselines

Input: 模型 π_θ, 数据集 D, 每个 prompt 采样 G 条 response, top-K 大小 K, 融合系数 β

Repeat:
  1. 采样 prompt x, 生成 G 条 response {y_i}_{i=1}^G
  2. 环境评估，获得 reward {r_i}
  3. 构建 teacher context:
     - 成功 response: context_i = [x, "correct", y_i]
     - 失败 response (组内有成功 y_best): context_i = [x, y_best, y_i]
     - 失败 response (组内全失败): context_i = [x, y_i]
  4. Student forward: 获得 student_topk_probs (B, T, K), student_topk_indices (B, T, K)
  5. Teacher forward (with context): 获得 teacher_log_probs (B, T)
     同时 gather teacher 在 student_topk_indices 上的 log_probs → teacher_at_student_topk (B, T, K)
  6. 计算横向 baseline V_CE:
     student_probs_norm = student_topk_probs / sum(student_topk_probs, dim=-1)
     V_CE = sum(student_probs_norm * teacher_at_student_topk, dim=-1)
  7. 计算纵向 baseline V_EMA:
     V_EMA[:, 0] = teacher_log_probs[:, 0]
     for t = 1 to T-1:
       V_EMA[:, t] = α * V_EMA[:, t-1] + (1-α) * teacher_log_probs[:, t-1]
  8. 融合 baseline:
     V = β * V_CE + (1-β) * V_EMA
  9. 计算 raw advantage:
     A = teacher_log_probs - V
  10. Per-response mean-centering:
      for each response i:
        A_mean_i = mean(A_{i,t} for valid t)
        A_{i,t} = A_{i,t} - A_mean_i
  11. Clipping:
      A = clamp(A, -C, C)
  12. Policy gradient update:
      L = -mean(A * log π_θ(y_t | x, y_{<t}) * mask)
      θ ← θ - lr * ∇L
Until converged
```

---

## 九、安全性分析

### 9.1 Death Spiral 免疫

TASD 的死亡螺旋：

```
success_rate ↓ → teacher 无信号 → z-score 放大噪声 → 模型退化 → success_rate ↓↓
```

本方法的行为：

```
success_rate ↓ → teacher 无信号 → Q_t ≈ V_CE_t → A_t ≈ 0 → 无梯度 → 模型不变
→ 等待下一个 batch 出现成功 response → 恢复学习
```

**失败模式是"停滞"，不是"崩溃"。** 停滞是安全的——模型保持当前水平直到新信号出现。

### 9.2 Entropy Collapse 缓解

TASD 的 entropy collapse 机制：

```
所有 token 都有强 advantage → 所有位置的分布都被 push → entropy 急剧下降
```

本方法的行为：

```
A_t 天然 sparse（大多数位置 ≈ 0）→ 只有关键位置有梯度 → 非关键位置的分布不变
→ entropy 下降更慢，因为只有少数位置被更新
```

### 9.3 Length Explosion 缓解

TASD 的 length explosion 机制：

```
长 response 的 token 数多 → 归一化时"投票权"多 → advantage 分布被偏移 → 奖励长度
```

本方法中 A_t = Q_t - V_t 不涉及跨 response 的归一化：

```
A_t 只取决于当前位置的 teacher 和 student 分布
→ 与 response 总长度无关
→ 不存在长度偏差
```

Per-response mean-centering 进一步确保：长 response 和短 response 的 A_t 平均值都为 0。

---

## 十、与 SDPO 的理论关系

### 10.1 SDPO 的隐式 advantage

SDPO 的 loss:
```
L_SDPO = KL(π_student(·|x, y<t) ‖ sg(π_teacher(·|x, f, y<t)))
```

展开后等价的 policy gradient advantage（论文 Proposition 2.1）:
```
A_SDPO(ŷ_t) = log π_teacher(ŷ_t | x, f, y<t) / π_student(ŷ_t | x, y<t)
```

对于生成的 token y_t:
```
A_SDPO(y_t) = log π_teacher(y_t) - log π_student(y_t)   ← teacher/student log ratio
```

### 10.2 统一 Baseline 视角

```
A_t = Q_t - B_t

不同方法的 baseline 选择：

B_t = 0                               → A_t = Q_t（raw teacher score，无 baseline）
B_t = log π_student(y_t)              → A_t = SDPO advantage
B_t = mean(Q_t for all t in group)    → A_t ≈ TASD（flat mean baseline）
B_t = EMA(Q_{<t})                     → A_t = Causal Bidirectional Credit
B_t = E_student[Q_t]                  → A_t = Cross-Entropy Bidirectional Credit
B_t = β·E_student[Q_t] + (1-β)·EMA   → A_t = Bidirectional Credit（本方法）
```

### 10.3 为什么 B_t = E_student[Q_t] 比 B_t = log π_student(y_t) 更好

SDPO 用 log π_student(y_t) 做 baseline：
```
A_SDPO = log π_teacher(y_t) - log π_student(y_t)
       = "teacher 比 student 更认可这个 token 吗？"
```

问题在于：student 选了 y_t，student 对 y_t 的概率天然不低（自己选的）。这意味着分母偏大，log ratio 被稀释。特别是对于失败的 response，teacher 和 student 对 y_t 的 log prob 差距不大（teacher 被迫给 student 已选的 token 分配概率），导致失败信号消失。

我们的方法用 E_student[Q_t] 做 baseline：
```
A_ours = log π_teacher(y_t) - E_{a~student}[log π_teacher(a)]
       = "teacher 对这个 token 的评价 vs teacher 对 student 平均选择的评价"
```

区别在于：SDPO 的 baseline 是 student 自己的判断，我们的 baseline 是 teacher 对 student 整个分布的期望评价。即使 teacher 对 y_t 给了中等概率，只要 teacher 更偏好其他选择（baseline 中某些 a_k 得分更高），A_t 就会是负的。失败信号不会消失。

---

## 十一、实现伪代码

```python
def compute_self_teacher_advantage(
    teacher_log_probs,            # (B, T) teacher 对 generated token 的 log prob
    student_topk_probs,           # (B, T, K) student 的 top-K 概率
    student_topk_indices,         # (B, T, K) student 的 top-K token ids
    teacher_logits,               # (B, T, V) teacher 完整 logits
    response_mask,                # (B, T) valid token mask
    beta=0.7,                     # V_CE vs V_EMA 融合系数
    ema_alpha=0.9,                # EMA 衰减系数
    clip_value=5.0,               # advantage 上下界
):
    """
    计算 self-teacher token-level advantage with bidirectional baselines
    无 z-score 归一化，无 token filter
    """
    B, T, K = student_topk_probs.shape
    
    # ===== Q_t: teacher 对实际选择的评分 =====
    Q = teacher_log_probs  # (B, T)
    
    # ===== V_CE: 横向 baseline（cross-entropy） =====
    # Teacher 对 student top-K tokens 的评分
    teacher_at_student_topk = teacher_logits.gather(2, student_topk_indices)  # (B, T, K)
    # Student top-K 概率归一化
    student_probs_norm = student_topk_probs / student_topk_probs.sum(-1, keepdim=True)
    # V_CE = E_student[teacher_score]
    V_CE = (student_probs_norm * teacher_at_student_topk).sum(-1)  # (B, T)
    
    # ===== V_EMA: 纵向 baseline（causal EMA） =====
    V_EMA = torch.zeros_like(Q)  # (B, T)
    V_EMA[:, 0] = Q[:, 0]
    for t in range(1, T):
        V_EMA[:, t] = ema_alpha * V_EMA[:, t-1] + (1 - ema_alpha) * Q[:, t-1]
    
    # ===== 融合 baseline =====
    V = beta * V_CE + (1 - beta) * V_EMA  # (B, T)
    
    # ===== Raw advantage =====
    A = Q - V  # (B, T)
    
    # ===== Per-response mean-centering（消除 prompt 难度差异） =====
    A_masked = A * response_mask
    seq_means = A_masked.sum(-1, keepdim=True) / response_mask.sum(-1, keepdim=True).clamp(min=1)
    A = (A - seq_means) * response_mask
    
    # ===== Clipping（数值保护） =====
    A = torch.clamp(A, -clip_value, clip_value)
    
    return A
```

---

## 十二、预期效果与实验设计

### 12.1 预期对比

| 指标 | GRPO | TASD (z-score) | 本方法 |
|------|------|----------------|--------|
| Best Acc | 基准 | +5~10% | +5~10%（预期与 TASD 持平） |
| Final Acc | 基准 | 不稳定（大 drop） | 更稳定（预期 drop 更小） |
| Entropy Drop | 中等 | 严重（95%+） | 轻（预期 <80%） |
| Length Explosion | 无 | 常见（73%） | 无（预期） |
| 需要 Filter/Gate | 否 | 是（必须） | 否 |
| 超参数数量 | 少 | 多（filter type, gate, AEW...） | 少（clip_value, K, β, α） |

### 12.2 消融实验设计

| 实验组 | β | V_EMA | V_CE | 目的 |
|--------|---|-------|------|------|
| β=1.0 | 1.0 | ✗ | ✅ | 纯 V_CE 的效果 |
| β=0.7 | 0.7 | ✅ | ✅ | 推荐融合 |
| β=0.5 | 0.5 | ✅ | ✅ | 等权融合 |
| β=0.0 | 0.0 | ✅ | ✗ | 纯 V_EMA（预期不稳定） |
| 无 baseline | - | ✗ | ✗ | A=Q（验证 baseline 必要性） |

### 12.3 实验方案

**第一步：可行性验证（1-2 天）**
- 在 Biology 上对比 GRPO baseline vs 本方法（β=0.7）
- 观察 accuracy 曲线、entropy 曲线、length 曲线
- 确认 death spiral 不发生

**第二步：Baseline 消融（3-5 天）**
- 对比 β=1.0 / 0.7 / 0.5 / 0.0
- 观察 V_CE 和 V_EMA 各自的贡献
- 确定最优 β

**第三步：和 TASD best config 对比（3-5 天）**
- 对比 TASD-v5 best（gate_hard_keep_reward + aew=none + gmSeq）
- 本方法如果 accuracy 持平但 stability 更好 → 成功

**第四步：Scale study（1 周）**
- 在 Qwen3-1.7B / 4B / 8B 上对比
- 预期：模型越大 → teacher 越准 → 本方法增益越大（和 SDPO 的 scaling 发现一致）
