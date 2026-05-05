# TASD v14 设计文档：最小版 error_pool on v5-baseline

> 版本：v14 草稿（未开工）
> 基础：`v5-baseline` 分支，commit `52149bd`（+ v13 9298637 sweep 脚本）
> 目标：在 v5 干净基础上，以最小代码改动引入 group-shared error_pool，让 teacher 能看到同一题下同组其他 rollout 的"格式错误"作为反例，缓解复读 / 错误模式自洽。

---

## 1. 背景与动机

### 1.1 为什么需要 error_pool

v5 的 teacher context 是 **per-rollout + 空 feedback**，即每条 rollout 只看到：

```
[problem] + [该 rollout 自己的 solution（若成功）] + [空 feedback]
```

问题：
- **teacher 看不到反例**：它无法区分"我这个答案为什么错"，只能朝"当前答案"方向打分；
- **复读正反馈**：当 student 答 `A A A A A...` 这种退化模式时，teacher 仍然给出合理的 log_prob（因为 teacher 也是 student 的 EMA），形成 reward 自洽；
- **缺乏群体信号**：同一题 8 个 rollout 各自为战，失败经验没有传递。

### 1.2 为什么选 **format-only** 错例池

HEAD (simplifed) 上的 v11 已经实证了 format-only gating 的价值：
- **语义错例会泄漏选项**：MCQ 题中若把"A / B / C"全部作为错例展示，teacher 会看到"B、C、D 都是错的" → 正确答案 A 被变相泄漏；
- **tooluse action 空间泄漏**：若把所有错误 action 当反例，实质上在帮 teacher 做"剩余动作推理"；
- **格式错例（无 `<answer>` 标签 / 无 `<tool_call>` 结构）才是真正的反例**：它只告诉 teacher "这种输出结构是错的"，不泄漏标签信息。

### 1.3 为什么选 **group-shared**

参考 memory `v5/v6 teacher prompt一致性对advantage无偏性的影响`：
- v5 同组 teacher prompt 一致 → `group_mean` 消除共性偏差，advantage 计算无偏；
- v6 per-rollout feedback → teacher prompt 漂移 → 破坏 group-relative advantage 理论基础。

v14 的 error_pool 必须 **group-shared**：同一 uid 的所有 rollout 看到完全相同的错例池（去重后 + 相同 reference answer），保障 advantage 的无偏性。

---

## 2. 核心设计

### 2.1 数据流

```
rollout batch (batch_size=32*8=256 条 response, 32 个 uid 每 uid 8 条)
    │
    ▼
┌──────────────────────────────────────────┐
│ v14 新增: _build_group_assets(batch, ...) │
│  for each uid:                             │
│    1. 挑 reference: 首个 success 的 final  │
│       answer span；若无 success 则用 GT    │
│    2. 挑 failed_indices: score<1 且       │
│       format_error_type != "none" (v11)    │
│    3. 每条失败 rollout → _build_display()  │
│       (tail 截断 + tag)                   │
│    4. dedup + sort by count + top-K       │
│    5. render 为 error_pool_text           │
│  返回 assets[uid] = {                     │
│    reference_answer,                       │
│    error_pool_text,                        │
│    n_errors, n_unique, format_counts       │
│  }                                         │
└──────────────────────────────────────────┘
    │
    ▼
for each rollout i (uid=uid_of_i):
  if teacher_context_mode == "group_shared":
      ├─ 用 assets[uid]["error_pool_text"] 注入
      ├─ 用 assets[uid]["reference_answer"] 注入
      └─ teacher prompt 在同 uid 内完全一致 ✅
  else (per_rollout, v5 默认):
      └─ legacy v5 路径，不变
```

### 2.2 teacher prompt 模板（group_shared 分支）

```
{problem}

Below are error patterns other students made for this problem; avoid them:
{error_pool_text}    # 若 pool 非空

Here is a correct reference answer:
{reference_answer}   # 若存在 success 或 GT
```

若 error pool 为空（无失败 rollout 或全部 answer-extractable），则退回仅 reference answer 的 prompt（等价 v5 的"空 feedback + solution" 结构）。

---

## 3. 最小代码变更清单

### 3.1 配置层（3 字段）

**`verl/trainer/config/tasd_simple.yaml`**（+8 行）：
```yaml
# v14: group-shared error_pool
teacher_context_mode: "per_rollout"   # per_rollout(默认, v5) | group_shared(v14)
max_errors_in_pool: 8                  # 每 group 去重后错例上限
error_answer_max_chars: 1024           # 每条错例答案字符上限
error_pool_format_only: true           # v11: 仅格式错例入池（默认 true，避免语义泄漏）
```

**`nebula_scripts/tasd_simple/tasd_simple_parametric.sh`**（+7 行）：
```bash
TEACHER_CONTEXT_MODE="${TEACHER_CONTEXT_MODE:-per_rollout}"
MAX_ERRORS_IN_POOL="${MAX_ERRORS_IN_POOL:-8}"
ERROR_ANSWER_MAX_CHARS="${ERROR_ANSWER_MAX_CHARS:-1024}"
ERROR_POOL_FORMAT_ONLY="${ERROR_POOL_FORMAT_ONLY:-True}"
# 追加到 verl main 的 --override 参数：
#   algorithm.tasd.teacher_context_mode=${TEACHER_CONTEXT_MODE}
#   algorithm.tasd.max_errors_in_pool=${MAX_ERRORS_IN_POOL}
#   algorithm.tasd.error_answer_max_chars=${ERROR_ANSWER_MAX_CHARS}
#   actor_rollout_ref.actor.self_distillation.error_pool_format_only=${ERROR_POOL_FORMAT_ONLY}
```

**`verl/workers/config/actor.py`** 的 `SelfDistillationConfig`（+1 字段）：
```python
error_pool_format_only: bool = True
```

### 3.2 核心逻辑（ray_trainer.py，新增 ~150 行）

**位置**：在 `_make_distillation_batch`（v5 原函数）之前插入以下 5 个辅助方法 + 1 个 dispatch 分支。

#### 3.2.1 辅助方法（静态，~80 行）

```python
@staticmethod
def _tail_truncate(text: str, max_chars: int) -> str:
    """取文本末尾 max_chars 字符，保留最后的答案部分。"""
    if not text:
        return ""
    return text if len(text) <= max_chars else text[-max_chars:]

@staticmethod
def _extract_format_tag(reward_info: dict) -> str:
    """从 reward_extra_info 中提取 format_error_type tag，用于 dedup key。"""
    return reward_info.get("format_error_type") or "none"

@classmethod
def _build_error_display(cls, response: str, reward_info: dict, tail_max_chars: int) -> dict:
    """单条失败 rollout → 错例显示项。"""
    tag = cls._extract_format_tag(reward_info)
    answer = cls._tail_truncate(response, tail_max_chars)
    # dedup_key: 用答案末 200 字符 + tag，避免过长字符串哈希开销
    dedup_key = (tag, answer[-200:])
    return {"answer": answer, "tag": tag, "dedup_key": dedup_key}

@staticmethod
def _dedup_and_aggregate_errors(displays: list, max_unique: int) -> list:
    """去重、按出现次数降序、取 top-K。"""
    grouped = {}
    order = []
    for d in displays:
        k = d["dedup_key"]
        if k not in grouped:
            grouped[k] = {"answer": d["answer"], "tag": d["tag"], "count": 0}
            order.append(k)
        grouped[k]["count"] += 1
    items = [grouped[k] for k in order]
    items.sort(key=lambda x: -x["count"])
    return items[:max_unique]

@staticmethod
def _render_error_pool_text(aggregated: list, n_total: int) -> str:
    """拼接错例池为 teacher-readable 文本。"""
    if not aggregated:
        return ""
    header = f"{n_total} previous attempt(s) failed, showing {len(aggregated)} unique error pattern(s):"
    blocks = []
    for i, item in enumerate(aggregated, start=1):
        blocks.append(
            f"[Error pattern {i}, observed in {item['count']} attempt(s), tag={item['tag']}]:\n{item['answer']}"
        )
    return header + "\n\n" + "\n\n".join(blocks)
```

#### 3.2.2 主构建方法（~50 行）

```python
def _build_group_assets(self, batch, reward_tensor, reward_extra_infos_dict,
                        success_by_uid, response_texts,
                        success_threshold: float,
                        max_errors_in_pool: int,
                        error_answer_max_chars: int,
                        error_pool_format_only: bool) -> dict:
    """per-uid 聚合资源：reference_answer + error_pool_text。"""
    from collections import defaultdict
    uids = batch.non_tensor_batch["uid"]
    batch_size = batch.batch.batch_size[0]
    data_sources = batch.non_tensor_batch.get("data_source", ["unknown"] * batch_size)
    ground_truths = [
        item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None)
        for item in batch
    ]
    seq_scores = reward_tensor.sum(dim=-1).detach().cpu().numpy()

    uid_to_indices = defaultdict(list)
    for i in range(batch_size):
        uid_to_indices[uids[i]].append(i)

    fmt_types_all = (reward_extra_infos_dict.get("format_error_type", None)
                     if reward_extra_infos_dict else None)
    def _has_fmt_error(idx):
        if fmt_types_all is None or idx >= len(fmt_types_all):
            return False
        return fmt_types_all[idx] not in (None, "none")

    assets = {}
    for uid, indices in uid_to_indices.items():
        if not indices:
            continue
        # 1. reference: 首个 success 的 final answer；无 success 则 GT
        reference_answer = None
        success_idxs = success_by_uid.get(uid, [])
        if success_idxs:
            ref_text = response_texts[success_idxs[0]]
            ref_text = self._remove_thinking_trace(ref_text)  # v5 语义
            reference_answer = self._extract_final_answer(
                ref_text, data_sources[indices[0]], error_answer_max_chars
            )
        elif ground_truths[indices[0]]:
            reference_answer = self._extract_final_answer(
                str(ground_truths[indices[0]]),
                data_sources[indices[0]], error_answer_max_chars
            )
        # 2. failed indices（format-only gating）
        if error_pool_format_only:
            failed_indices = [i for i in indices
                              if seq_scores[i] < success_threshold and _has_fmt_error(i)]
        else:
            failed_indices = [i for i in indices if seq_scores[i] < success_threshold]
        # 3. displays
        displays = []
        for i in failed_indices:
            info = {}
            if reward_extra_infos_dict is not None:
                v = reward_extra_infos_dict.get("format_error_type")
                if v is not None and i < len(v):
                    info["format_error_type"] = v[i]
            displays.append(self._build_error_display(
                response_texts[i], info, error_answer_max_chars
            ))
        aggregated = self._dedup_and_aggregate_errors(displays, max_errors_in_pool)
        error_pool_text = self._render_error_pool_text(aggregated, len(failed_indices))
        assets[uid] = {
            "reference_answer": reference_answer,
            "error_pool_text": error_pool_text,
            "n_errors": len(failed_indices),
            "n_unique": len(aggregated),
        }
    return assets
```

#### 3.2.3 主循环 dispatch（~20 行，替换现有 teacher prompt 构建处）

```python
teacher_context_mode = tasd_cfg.get("teacher_context_mode", "per_rollout")
is_group_shared = (teacher_context_mode == "group_shared")

group_assets = None
if is_group_shared:
    group_assets = self._build_group_assets(
        batch=batch,
        reward_tensor=reward_tensor,
        reward_extra_infos_dict=reward_extra_infos_dict,
        success_by_uid=success_by_uid,
        response_texts=response_texts,
        success_threshold=self_distillation_cfg.get("success_reward_threshold", 1.0),
        max_errors_in_pool=tasd_cfg.get("max_errors_in_pool", 8),
        error_answer_max_chars=tasd_cfg.get("error_answer_max_chars", 1024),
        error_pool_format_only=self_distillation_cfg.get("error_pool_format_only", True),
    )

# v5 原循环保持不变；在构建 teacher prompt 处分叉：
for i in range(batch_size):
    uid = uids[i]
    if is_group_shared and group_assets and uid in group_assets:
        a = group_assets[uid]
        err_block = (f"\n\nBelow are error patterns other students made; avoid them:\n{a['error_pool_text']}"
                     if a["error_pool_text"] else "")
        ref_block = (f"\n\nHere is a correct reference answer:\n{a['reference_answer']}"
                     if a["reference_answer"] else "")
        teacher_prompt = f"{problem_text}{err_block}{ref_block}"
    else:
        # legacy v5 per_rollout: 保留原代码
        teacher_prompt = <v5 原构建逻辑>
```

### 3.3 metrics（+10 行）

```python
if is_group_shared and group_assets:
    metrics["tasd/group_n_errors_mean"] = np.mean([a["n_errors"] for a in group_assets.values()])
    metrics["tasd/group_n_unique_mean"] = np.mean([a["n_unique"] for a in group_assets.values()])
    metrics["tasd/group_with_ref_frac"] = np.mean([a["reference_answer"] is not None for a in group_assets.values()])
    metrics["tasd/group_with_pool_frac"] = np.mean([bool(a["error_pool_text"]) for a in group_assets.values()])
```

### 3.4 总变更量预估

| 文件 | +行 | -行 |
|---|---|---|
| `verl/trainer/config/tasd_simple.yaml` | +8 | 0 |
| `verl/workers/config/actor.py` | +1 | 0 |
| `nebula_scripts/tasd_simple/tasd_simple_parametric.sh` | +12 | 0 |
| `verl/trainer/ppo/ray_trainer.py` | **+150** | ~5 |
| `nebula_scripts/submit_tasd_v14_error_pool_sweep.sh` | +180 (新建) | 0 |
| **总计** | **~350** | ~5 |

比 HEAD 的 +800 行轻 5 倍；且所有新代码都可通过 `teacher_context_mode=per_rollout` 完全绕过（等价 v5 行为）。

---

## 4. 实验矩阵（v14 sweep）

`nebula_scripts/submit_tasd_v14_error_pool_sweep.sh`（草案）：

| JOB | context_mode | format_only | max_errors | 说明 |
|---|---|---|---|---|
| v14-ctrl | per_rollout | — | — | 对照：等价 v13（应复现 0.68） |
| v14-fmt8 | group_shared | true | 8 | 主推：format-only pool, top 8 |
| v14-fmt4 | group_shared | true | 4 | 消融：更紧凑的 pool |
| v14-all8 | group_shared | false | 8 | 消融：不做 format gating（测 v11 的判据） |

所有 JOB 固定：bio + v5 其他全部超参（见 v13 sweep）。

---

## 5. 成功判据 / 失败诊断

### 5.1 成功
- `v14-ctrl` peak ≥ 0.65（确认 dispatch 代码没破坏 v5 路径）；
- `v14-fmt8` peak ≥ v14-ctrl + 0.02；rlen 稳定在 80-500；
- `tasd/group_with_pool_frac` 在 step 0-50 保持 0.3-0.8（证明 pool 在起作用，不是空的）。

### 5.2 失败模式
| 现象 | 诊断 |
|---|---|
| v14-ctrl 就崩 | dispatch 代码污染了 v5 路径 → 回滚重查 |
| v14-fmt8 和 ctrl 没差别 | pool 没真正注入 teacher prompt，查 teacher_prompt 日志 |
| v14-fmt8 比 ctrl 差 | format_only gating 在此数据集下不够，错例仍泄漏；尝试 format_only=false |
| response length 爆炸 | error_pool 文本过长，缩减 max_errors_in_pool 或 error_answer_max_chars |

---

## 6. 风险与回滚

### 6.1 主要风险

1. **teacher prompt 长度增加** → teacher log_prob 计算 token 数增加 → reward 尺度漂移；
   缓解：把 error_pool 文本长度纳入指标监控，设置硬上限 `max_errors_in_pool * error_answer_max_chars = 8192`。

2. **GT 泄漏**（c77ca8d 修过的历史问题）：reference_answer 从 success rollout 抽取时，若 `_extract_final_answer` 不小心透出原 GT → 实质给 teacher 送了标签；
   缓解：首个成功 rollout 的答案**本身就是正确答案**，这不算泄漏——因为 group 内已经有 1 个 student 做到了。危险的是 GT fallback 路径（无 success）直接把标签当 reference，这种情况下"reference"的效果接近监督学习；记作 feature 而非 bug，但需要监控 `reference_source_gt_frac`（若 GT 占比过高，说明 student 答对率太低，可能已经进入崩塌边缘）。

3. **per_rollout 分支污染**：v14 改动了 teacher prompt 主循环，即便 `context_mode=per_rollout` 也走同一份代码。
   缓解：v14-ctrl 作为对照，必须复现 v13 的 peak 才能释放后续 JOB。

### 6.2 回滚策略
- 所有新代码默认不激活（`teacher_context_mode=per_rollout` 为 default）；
- git revert 单个 commit 即可回到 v13。

---

## 7. 后续路线

| 版本 | 增量 | 依赖 |
|---|---|---|
| v13 | v5 baseline 复现 | （当前已提交 nebula，待验证） |
| **v14** | v13 + 最小 error_pool（本文档） | v13 成功 |
| v15 | v14 + A 方案 outcome fusion | v14 成功（复用 v12 commit 9f12132 的 core_algos.py 局部改动） |
| v16 | v15 + B 方案 outcome-conditional teacher reward | v15 成功（按 "TASD B方案设计文档" 实现） |

---

## 8. 开工前 checklist

- [ ] v13 job 复现 peak ≥ 0.65
- [ ] v13 代码路径无 assertion/NaN 异常
- [ ] 在 `v5-baseline` 分支切出 `v14-error-pool` feature branch
- [ ] 按第 3 节改动逐个文件 patch
- [ ] 本地 dry-run 验证 parametric 正确拼出新 hydra override
- [ ] ray_trainer.py 语法 lint 通过
- [ ] 提交前 `git diff v5-baseline` 总行数 ≤ 400
- [ ] submit v14 sweep（4 JOB）

---

_草稿版本：v0.1；下次更新待 v13 结果出来后修订成功判据阈值。_
