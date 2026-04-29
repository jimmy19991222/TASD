# Teacher Context 增强改动说明

## 背景

TASD (Teacher-Student Adaptive Distillation) 中，teacher 模型在计算 token-level reward 时，其输入 context 的质量直接决定 reward 信号的引导能力。本次改动统一增强了所有数据集（tooluse / sciknoweval）的 teacher context，核心目标：

1. **错误答案可视化**：teacher 看到"学生答了什么"，而非仅知道"答错了"
2. **细粒度反馈**：精确指出错误类别（格式/Action/答案），而非笼统的 mismatch
3. **成功 rollout 自参考**：成功的 rollout 固定参考自己的答案，避免被错误示范干扰

---

## 一、修改概览

| 文件 | 修改内容 | 影响范围 |
|------|---------|---------|
| `verl/utils/reward_score/feedback/tooluse.py` | feedback 从简单 mismatch → 三级细粒度（格式/Action/Input） + 5 种格式错误分类 + 监控指标 | tooluse |
| `verl/utils/reward_score/feedback/mcq.py` | feedback 从空字符串 → 两级细粒度（格式/答案） | sciknoweval (bio/chemistry/physics/material) |
| `verl/workers/config/actor.py` | `feedback_template` 新增 `failed_attempt` 段落 | 所有数据集 |
| `verl/trainer/ppo/ray_trainer.py` | 1. `_get_solution` 简化：成功 rollout 永远返回自己<br>2. Ground truth fallback：无成功 rollout 时用标准答案构造示范<br>3. 格式错误类型分布监控指标 | 所有数据集 |

---

## 二、Reward Feedback 生成逻辑对比

### 2.1 Tooluse

**文件**: `verl/utils/reward_score/feedback/tooluse.py`

#### 修改前
```python
feedback_parts = []
if not actions_correct:
    feedback_parts.append(f"Actions mismatch: predicted {pred_actions}, expected {gt_actions}")
if not action_inputs_correct:
    feedback_parts.append(f"Action inputs mismatch: predicted {pred_action_inputs}, expected {gt_action_inputs}")
```

**特点**: 只有简单的 mismatch 描述，不区分错误类型。

#### 修改后

| 错误类型 | 触发条件 | 示例反馈 |
|---------|---------|---------|
| **格式错误-缺失两者** | 无 `Action:` 和 `Action Input:` | `Format error: missing both 'Action:' and 'Action Input:' fields` |
| **格式错误-缺失Action** | 有 `Action Input:` 但无 `Action:` | `Format error: missing 'Action:' field` |
| **格式错误-缺失Action Input** | 有 `Action:` 但无 `Action Input:` | `Format error: missing 'Action Input:' field` |
| **格式错误-空JSON** | 有 `Action Input:` 但无 JSON 内容 | `Format error: 'Action Input:' has no JSON content` |
| **格式错误-JSON解析失败** | JSON 语法错误 | `Format error: JSON parse error in Action Input (...)` |
| **Action 错误** | 工具类型/数量/顺序不匹配 | `Action error: should call [search], but called [calculator] (wrong action: used [calculator] instead of [search])` |
| **Input 错误** | 参数 key/value 不匹配 | `Input error: expected {'query': 'test'}, but got {'query': 'example'} (query should be 'test', got 'example')` |

**特点**: 精确指出错误类别和具体差异（missing/extra/wrong value/json_parse）。

**监控指标**: 每种格式错误类型自动上报到 metrics：
- `self_distillation/format_error_missing_both`
- `self_distillation/format_error_missing_action`
- `self_distillation/format_error_missing_action_input`
- `self_distillation/format_error_empty_json`
- `self_distillation/format_error_json_parse_error`

---

### 2.2 SciknowEval (MCQ)

**文件**: `verl/utils/reward_score/feedback/mcq.py`

#### 修改前
```python
return {
    "score": reward,
    "acc": reward,
    "pred": multiple_choice_answer,
    "incorrect_format": 1 if incorrect_format else 0,
    "feedback": "",  # 永远是空的
}
```

**特点**: 不返回任何 feedback，teacher 不知道"为什么错了"。

#### 修改后

| 错误类型 | 触发条件 | 示例反馈 |
|---------|---------|---------|
| **格式错误** | 没有 `<answer>...</answer>` 标签 | `Format error: response does not contain valid <answer>...</answer> tags` |
| **答案错误** | 格式正确但选项不对 | `Answer error: predicted A, expected B` |

**特点**: 区分"格式问题"和"知识问题"，teacher 能针对性引导。

---

## 三、Teacher Context 结构对比

### 3.1 失败 Rollout（有成功示范）

#### 修改前

```
system: You are a helpful assistant.

user: {原始问题}

The following is feedback from your unsuccessful earlier attempt:

Actions mismatch: predicted [calculator], expected [search]

Correctly solve the original question.

assistant: {当前模型生成的 response}
```

**问题**:
- Teacher 看不到"错误的答案长什么样"
- Feedback 只有文字描述，没有错误答案对照
- 如果 group 中无成功 rollout，teacher 看不到任何正确答案

#### 修改后

```
system: You are a helpful assistant.

user: {原始问题}

Correct solution:

Action: search
Action Input: {"query": "quantum computing"}

Your previous attempt:

Action: calculator
Action Input: {"expression": "1+1"}

The following is feedback from your unsuccessful earlier attempt:

Action error: should call [search], but called [calculator] (wrong action: used [calculator] instead of [search])

Correctly solve the original question.

assistant: {当前模型生成的 response}
```

**改进**:
1. **新增 failed_attempt**: Teacher 看到完整的错误答案
2. **细粒度 feedback**: 精确指出"调用了错误的工具"、"参数值不对"
3. **成功示范**: 同 group 的成功 rollout 作为参考答案

---

### 3.2 失败 Rollout（无成功示范）

#### 修改前

```
system: You are a helpful assistant.

user: {原始问题}

Correctly solve the original question.

assistant: {当前模型生成的 response}
```

**问题**: Teacher 没有任何参考信息，只能凭自身知识评分。

#### 修改后

```
system: You are a helpful assistant.

user: {原始问题}

Correct solution:

Action: search
Action Input: {"query": "quantum computing"}

Your previous attempt:

Action: calculator
Action Input: {"expression": "1+1"}

The following is feedback from your unsuccessful earlier attempt:

Action error: should call [search], but called [calculator] (wrong action: used [calculator] instead of [search])

Correctly solve the original question.

assistant: {当前模型生成的 response}
```

**改进**:
1. **Ground truth fallback**: 当 group 内无成功 rollout 时，用标准答案构造正确示范（`Action: search...`）
2. **错误答案 + 详细反馈**: Teacher 看到"哪里错了"和"应该怎么答"
3. **防止死亡螺旋**: 即使 success rate 归零，teacher 仍有正确答案作为参考

---

### 3.3 成功 Rollout（自参考）

#### 修改前

成功 rollout 的 `solution` 从同 group 的**随机成功样本**中选择，可能选到别人：

```
# Rollout A (成功，acc=1.0)
solution = random.choice(success_by_uid[uid])  # 可能选到 B 的答案

# Rollout B (成功，acc=1.0)  
solution = random.choice(success_by_uid[uid])  # 可能选到 A 的答案
```

**问题**: 成功 rollout 学习别人的答案，而非巩固自己的正确行为。

#### 修改后

成功 rollout 固定返回**自己的答案**：

```
# Rollout A (成功，acc=1.0)
solution = response_texts[A]  # 固定看自己

# Rollout B (成功，acc=1.0)
solution = response_texts[B]  # 固定看自己
```

**改进**: 成功 rollout 参考自己，强化自身正确行为；失败 rollout 才看别人的示范。

---

## 四、核心差异总结

| 维度 | 修改前 | 修改后 |
|------|-------|-------|
| **Feedback 内容** | 空字符串 / 简单 mismatch | 细粒度分类（格式/Action/Input/答案） |
| **错误答案展示** | 无 | `Your previous attempt` 段落 |
| **成功示范选择** | 随机选同 group 成功样本 | **成功 rollout 固定看自己** |
| **失败示范选择** | 随机选同 group 成功样本 | 随机选同 group 成功样本 + **ground truth fallback** |
| **无成功 rollout 时** | Teacher 无正确答案参考 | **Ground truth 构造示范**，防止死亡螺旋 |
| **适用数据集** | 所有 | tooluse + sciknoweval |

---

## 五、实验脚本

**脚本**: `nebula_scripts/submit_tasd_tooluse_feedback_enhanced_sweep.sh`

**核心标签**: `-fbEnhanced`

**可扫描超参**:
- `REMOVE_THINKING_FROM_DEMONSTRATION`: True（默认）/ False
  - True: 正确答案示范去掉 `<think>...</think>`
  - False: 正确答案示范保留完整 thinking

---

## 六、局限性

即使做了上述增强，核心问题仍存在：

1. **Teacher 无领域知识**: Teacher 只是语言模型，tooluse 上不知道"应该调用什么工具"
2. **依赖成功 rollout**: 如果 group 内所有 rollout 都失败，teacher 仍看不到正确答案
3. **无法完全解决死亡螺旋**: 细粒度反馈只能延缓崩溃，不能阻止 success rate 归零

**建议后续方向**:
- 混合 reward: `reward = α * teacher_log_prob + β * success_reward`
- 或在数据预处理阶段把标准答案注入 teacher 的 system prompt
