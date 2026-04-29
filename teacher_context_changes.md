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
| `verl/trainer/ppo/ray_trainer.py` | 1. `_get_solution` 简化：成功 rollout 永远返回自己<br>2. Ground truth fallback：无成功 rollout 时用标准答案构造示范<br>3. 格式错误类型分布监控指标<br>4. 新增 `feedback_available_fraction` / `feedback_used_fraction` 观测指标 | 所有数据集 |
| `verl/trainer/config/actor/actor.yaml`<br>`verl/workers/config/actor.py` | `self_distillation.include_environment_feedback`（默认 False）<br>`self_distillation.environment_feedback_only_without_solution`（yaml 默认 True） | 所有数据集 |
| `nebula_scripts/tasd_simple/tasd_simple_parametric.sh` | 两个开关均环境变量化并通过 hydra 注入 | 所有数据集 |
| `nebula_scripts/submit_tasd_feedback_enhanced_sweep.sh` | 固定 `INCLUDE_ENVIRONMENT_FEEDBACK=True` + `ENVIRONMENT_FEEDBACK_ONLY_WITHOUT_SOLUTION=False` | fbEnhanced sweep |

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
3. **防止死亡螺旋**: 即使 success rate 归零，teacher 仍能通过 GT fallback 看到正确答案

---

### 3.2.1 教师获取正确答案的双路径

Teacher 获取正确答案有两个独立来源（代码见 `ray_trainer.py` L874-900）：

1. **成功 rollout 路径**（L874-886）：`_get_solution` 从同 group 中采样成功的 response
2. **Ground truth fallback 路径**（L888-900）：当 `solution_strs[i] is None`（group 内无成功 rollout）时，从 `reward_model.ground_truth` 构造标准答案示范

因此即使 group 内所有 rollout 都失败，teacher 的正确示范来自 GT fallback（数据集标准答案）；只有当 GT fallback 也未提供时，teacher 才真正失去正确答案参考。

**完整的 teacher context 组成**（`environment_feedback_only_without_solution=False`）：

```
Correct solution: {同 group 成功样本 或 GT fallback 构造的正确答案}

Your previous attempt: {学生当前生成的答案}

The following is feedback from your unsuccessful earlier attempt:
{细粒度错误反馈：格式错误/Action错误/答案错误}
```

---

### 3.3 Feedback 注入的两道闸门（v6 诊断新增）

fbEnhanced 是否真正生效，取决于两个独立开关的组合：

```python
# verl/trainer/ppo/ray_trainer.py :: _build_teacher_message
feedback_only_without_solution = self_distillation_cfg.get(
    "environment_feedback_only_without_solution", False
)
use_feedback = has_feedback and (not feedback_only_without_solution or not has_solution)
```

| `include_environment_feedback` | `environment_feedback_only_without_solution` | GT fallback 状态 | 实际效果 |
|---|---|---|---|
| False | *（任意）* | *（任意）* | 完全关闭 feedback，等价于普通 TASD |
| True | True | 开启（默认，覆盖 ~100%） | **feedback 被 solution 挡住，等价于关闭** ⚠️ |
| True | True | 无 solution 的样本 | feedback 仅在兜底场景注入 |
| **True** | **False** | *（任意）* | **真正的 fbEnhanced：feedback 与 solution 并存** ✅ |

**关键踩坑**（v6 实验 `fbEnhanced-v2-...20260429_162342`）：

- 日志显示 `feedback_available_fraction = 0.58 ~ 0.71`（MCQ feedback 链路已通）
- 但 `feedback_used_fraction = 0.0`（最终没有一个样本用上 feedback）
- 根因：`actor.yaml` 默认 `environment_feedback_only_without_solution: True`，叠加 GT fallback 让所有样本都有 solution → feedback 永远被挡住
- 结果：v6 的 `teacher_message` 与 v5 完全相同（system + prompt + solution_section），**训练轨迹等价于 v5，仅 SwanLab 多出观测指标**
- 修复：sweep 脚本固定 `ENVIRONMENT_FEEDBACK_ONLY_WITHOUT_SOLUTION="False"`，让 feedback 与 solution 并存

**调试信号**：
- `self_distillation/feedback_available_fraction`：feedback 字段非空的样本比例（上游 reward 链路健康度）
- `self_distillation/feedback_used_fraction`：真正进入 teacher_message 的比例（最终生效率）
- 两者差值 > 0 即意味着 feedback 被闸门挡掉，需要检查 `environment_feedback_only_without_solution`

---

### 3.4 成功 Rollout（自参考）

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

**脚本**: `nebula_scripts/submit_tasd_feedback_enhanced_sweep.sh`（`PROJECT_NAME=TASD-v6`）

**核心标签**: `-fbEnhanced`

**关键环境变量**（两者必须同时设置才能真正启用）：
```bash
INCLUDE_ENVIRONMENT_FEEDBACK="True"                     # 闸门 1：是否在 teacher context 注入 feedback
ENVIRONMENT_FEEDBACK_ONLY_WITHOUT_SOLUTION="False"      # 闸门 2：False = feedback 与 solution 并存
```

**可扫描超参**:
- `REMOVE_THINKING_FROM_DEMONSTRATION`: True（默认）/ False
  - True: 正确答案示范去掉 `<think>...</think>`
  - False: 正确答案示范保留完整 thinking

**Commit 参考**:
- `a7eb014`：`include_environment_feedback` 参数化
- `2055fc1`：`environment_feedback_only_without_solution` 参数化 + sweep 启用 feedback+solution 并存

---

## 六、局限性

即使做了上述增强，核心问题仍存在：

1. **Teacher 无领域知识**: Teacher 只是语言模型，tooluse 上不知道"应该调用什么工具"
2. **依赖 GT fallback**: 当 group 内无成功 rollout 时，teacher 的正确示范完全依赖 GT fallback；如果数据集未提供 ground_truth，teacher 才真正失去正确答案参考
3. **无法完全解决死亡螺旋**: 细粒度反馈 + GT fallback 能延缓崩溃，但不能阻止 success rate 归零（当 GT 也不存在时 teacher 才真正失去参考）

**建议后续方向**:
- 或在数据预处理阶段把标准答案注入 teacher 的 system prompt
- 配置健康度：在启动日志中硬检查 `include_environment_feedback` 与 `environment_feedback_only_without_solution` 组合，避免再次出现 v6 这种"开关开了但被另一侧默认值静默抵消"的失效实验
