# Teacher-Guided Branching Rollout (TG-Branching)

**分支**: `tg-branching-rollout`,base=`baseline`
**日期**: 2026-06-01
**状态**: Phase 0 进行中

---

## §0 一句话

每个 prompt 由 student 单链生成,在 token 熵突变处用 teacher 在 student top-K 里挑 argmax/argmin 两个 token,分裂成两条续写链。3 次分裂 → 8 条共前缀的轨迹 → GRPO 训练。

目的:制造**方差大、prefix 强相关**的 group 样本,把 GRPO 的 group baseline 喂到不平凡的方差结构里。

---

## §1 与已有 research 的衔接

| 文档 | 关系 |
|---|---|
| `teacher_guided_grpo.md` (TG-GRPO) | 同源直觉(teacher 当 selector,不当 critic),但 TG-GRPO 是 **loss-side 加权**;TG-Branching 是 **rollout-side 制造样本**。两者正交,可叠加。 |
| `next_gen_sd_proposals.md` §4 CT-SD | 用 entropy 选 fork point + sibling rollout,但 sibling 是 student 自采。TG-Branching 替换为 **teacher 主导的 argmax/argmin**,信号更强。 |
| `teacher_primed_value_grpo.md` (TPV-GRPO) | 完全不同 axis(value head),正交。 |
| `bellman_three_q_analysis.md` | 仍 relevant:branching 让 group baseline 接近 advantage estimator 的同 prefix 假设,$V_\pi$ bias 减小。 |

---

## §2 关键架构事实(来自 baseline 上的 Phase-1 调研)

1. vLLM rollout 走 async 路:`AgentLoopManager` → `SingleTurnAgentLoop.run` → `vLLMReplica.engine.generate(TokensPrompt, SamplingParams, request_id)`。
2. vLLM 的 `n` 从未被设过,trainer 端用 `gen_batch.repeat(repeat_times=rollout.n, interleave=True)` 平铺,8 个 rollout = 8 次独立 `engine.generate` 调用,prefix cache 复用 prompt KV。
3. teacher 模型只在 FSDP actor worker 上,**不在 vLLM 上** — 但 SDPO 的 teacher = student EMA + 不同 prompt,可以让同一个 vLLM engine 用 privileged-context prompt 拿 next-token logprobs,等价于 teacher 查询。
4. baseline 上 SDPO 的 "ref" 来自 **peer rollout**(同 uid 组里 reward 高的兄弟)— branching rollout 期间 peer 还没生成完,**ref-peer 不可用**。
5. GRPO `compute_grpo_outcome_advantage` 除以 `id_std + eps`,8 共 prefix 的 sibling 若 reward 全相同 → 梯度爆炸。

---

## §3 Teacher 模式(V1 支持)

| 模式 | 何时可用 | 内容 |
|---|---|---|
| `marker` | 任何时刻 | 在 assistant 段前 prepend 静态 `VERDICT_RIGHT_MARKER` 字符串,不需要任何 ground truth |
| `gt_marker` | 数据集有 `ground_truth` 字段时 | prepend `"This answer is verified correct: <gt>"`(模板可配),from `batch.non_tensor_batch["reward_model"][i]["ground_truth"]` |
| `ref-gt` (V1.5) | 同上 | 把 gt 作为完整 ref,模拟 SDPO 的 reprompt 但用 parquet gt 而非 peer rollout |
| `ref-peer` | 训练时全 batch rollout 完后 | **不在 V1 范围**,因为 branching 期间 peer 没完成。延后 V2。 |

V1 默认 `gt_marker`,因为它在 sciknoweval / lcb 上都有 `ground_truth` 列。

---

## §4 决策 token 检测(rolling z-score)

- 维护过去 W=20 个 token 的熵(熵从 vLLM 返回的 top-K logprobs 截断估计)。
- 第 t 个 token(t ≥ W)若 `H_t > running_mean + k * running_std` → 决策 token。
- W 个保护期 token 内不触发(信号不稳定)。
- 默认 k=2.0;若一个 prompt 整条链扫完不足 3 个决策 token,**降 k**(0.5 步长,floor 0.5),重扫,直到拿到 3 个。
- 实现细节:用 deque(maxlen=W) 增量维护 `sum`/`sum_sq` 计算 mean/std。

---

## §5 Teacher 分裂选择

- 在决策位 t 上,student 的 top-K(K=50)候选 `{c_1..c_K}` 已经从 vLLM 返回(我们要把现有 sampling_params["logprobs"]=K 通过)。
- 同时发一个 vLLM 请求:prompt = `privileged_context + student_response[:t]`, max_tokens=1, logprobs=K(K 取 50,确保覆盖)。
- 读取 teacher 返回的 top-K → 在与 student top-K 的交集 `S` 上找 `argmax_{c∈S} log p_T(c)` 和 `argmin_{c∈S} log p_T(c)`。
- 若交集 < 2 个,把 K 拉到 100 重试一次;还不行,该位 fallback 为「student top-1 + student top-2」继续(等价于不分裂的 epsilon-greedy)。

---

## §6 树管理 & Batch 对齐

- 每个 prompt 生成一棵深度 3 的 binary tree,8 个 leaf。
- Trainer 仍消费 `(B*n, T)` flat batch(n=8)— rollout 模块需把 8 个 leaf flatten 进同一个 prompt 的 tile group,uid 不变。
- 新增 DataProto field `branch_token_mask: (B*n, T)`,1 表示该位是 teacher 强制注入,0 表示 student 自采。
- prefix cache:每条 leaf 续写时 prompt = `original_prompt + 共享前缀 + 分裂 token`,前两部分 KV 命中。最坏情况是分裂深 3 → 4 次 generate 调用 / leaf。

---

## §7 三种 branch token 损失模式(ablation)

- `branch_token_loss_mode = "all"`:所有 token 一起训(包括 teacher 注入位)。最 naive,但因 token 在 student top-K 内,off-policy 偏差有限。
- `branch_token_loss_mode = "mask"`:屏蔽掉 branch token 的 PG 信号,等价于 `response_mask[branch_token_mask] = 0`。最干净。
- `branch_token_loss_mode = "only"`:**只**在 branch token 上算 PG,其他位置 mask 掉。研究 hypothesis "decision token 才是真正的信号"。

实现:在 `compute_policy_loss_vanilla` 调用前,根据模式调整 `response_mask`。无需改 core_algos。

---

## §8 GRPO 群组 std floor

- 加 `algorithm.adv_std_floor`(默认 0.05)。
- `compute_grpo_outcome_advantage` 计算 `id_std` 后,`id_std = max(id_std, adv_std_floor)`。
- 防止 8 leaf reward 全 0 或全 1 时 advantage 爆炸。

---

## §9 Phase 计划

| Phase | 内容 | DoD |
|---|---|---|
| 0 | backport `teacher_context_mode ∈ {marker, gt_marker, ref-gt}` 框架到 baseline。新建 `verl/utils/verdict_markers.py`,扩 SelfDistillationConfig,改 `_maybe_build_self_distillation_batch`。 | smoke import + 一次 `--dry-run` Hydra 解析通过 |
| 1 | branching rollout 模块。新增 `BranchingAgentLoop` 子类,实现 entropy-spike + teacher branch-query + 树管理。`branch_token_mask` 字段透传。 | 1 prompt 单卡 smoke 拿到 8 条共 prefix 的 leaf |
| 2 | GRPO 端:`branch_token_loss_mode` 三模式 + `adv_std_floor`。 | 8-prompt 1-step 跑通,metric 全部展示 |
| 3 | sciknoweval/biology 6h GPU pilot:branching vs vanilla GRPO | acc 不掉 + std 显著高于 vanilla |
| 4 | SDPO 接入(branching + reprompt 同存) | 留作 V2,先冻结 |

---

## §10 风险

| 风险 | 缓解 |
|---|---|
| vLLM logprobs=50 在长 response 下显存压力 | response_length=4096 × 50 × float32 ≈ 800KB / leaf,可承受 |
| 8 leaf 串行 generate → wall clock 慢 2-3× | 第一阶段接受;V2 看是否能用 vLLM 的 `n` 内置参数加速分裂前段 |
| gt_marker 把答案泄露进 student rollout | 只在 teacher branch-query 用,不在 student 主链 prompt 里。**严禁**让 student 看到 gt。 |
| id_std collapse | adv_std_floor 已防 |
| 决策 token 全部集中在序列开头 | 熵检测前 W 个保护期 + sigma 检验,后段才容易触发。可能仍需调 W |

---

## §11 命名

- 分支: `tg-branching-rollout`
- rollout class 名: `BranchingAgentLoop`(继承 `SingleTurnAgentLoop`)
- 配置 namespace: `actor_rollout_ref.rollout.branching.*`
- loss mode: 不新增,沿用 `vanilla` + `branch_token_loss_mode`(因为 advantage 仍是 GRPO 的)。SDPO V2 时再考虑。

---

## §12 N-trees topology(2026-06 升级)

### 动机

Phase-3 / SDPO branching pilot 观察：`n_splits=3` 单棵深树在 sciknoweval 上 acc≈0.485，显著低于 vanilla GRPO ceiling（0.5875）。诊断显示 8 个 leaf 共享前 ~70-80% prefix 时，reward variance 主要来自尾段决策点，group baseline 信号集中，advantage 趋同 → 模型快速过拟合到一条主路径。

**N-trees 拓扑**：把「1 prompt → 1 棵 8-leaf 深树」换成「1 prompt → 多棵浅树」，每棵树独立 vLLM seed + 独立 prompt rollout，树间无前缀共享。这样 8 个 leaf 拆成 4 组共前缀的 (sibling, sibling) pair，每个 pair 内部仍由 teacher 在 disagreement 位上分裂，但 pair 之间的多样性来自不同初始 rollout，大幅降低过拟合风险。

### 配置（`BranchingConfig`）

- `n_trees: int = 1` — 每个 prompt 派生的独立树数量，默认 1 保持向后兼容。
- `n_splits: int = N` — 每棵树的分裂深度。每棵树叶子数 `2**n_splits`。
- 强制约束：`rollout.n == n_trees * 2**n_splits`（违反则在 `__post_init__` 抛错）。
- `split_trigger: str = "entropy"` — 候选 `{entropy, entropy_disagreement}`。
  - `entropy`：沿用旧逻辑，在 candidate 位中选第一个高熵位分裂。
  - `entropy_disagreement`：在所有候选高熵位中遍历，只在 teacher 在 student top-K 的 argmax 与 student 实际采样 token 不一致的位置分裂；若全部位置一致则该树不分裂（记 `branching/no_disagreement_count`）。要求 `teacher_guided_rollout=True`。

### 推荐首发

```
N_TREES=4 N_SPLITS=1 ROLLOUT_N=8 \
BRANCH_TOKEN_LOSS_MODE=suffix \
SPLIT_TRIGGER=entropy_disagreement \
bash nebula_scripts/submit_tg_branching_pilot.sh --variant grpo_tg
# --variant sdpo_tg 同理
```

4 棵 1-split 树 = 4 个独立 (sibling, sibling) pair / prompt，每个 pair 仅在最关键的 disagreement 位差一个 token。SwanLab run name 自动带 `tT4-bsplit1-trigDis` 标签，便于和 vanilla GRPO / 旧 1-tree branching 同图对比。

### 实现要点

- `BranchingAgentLoop` 全局 slot 计数器仍按 base prompt key，从 slot 解构 `(tree_idx, leaf_idx_in_tree)`。
- `cache_key = base_cache_key + "#t" + tree_idx`，每棵树独立 sibling owner / asyncio.Future。
- vLLM seed：`seed = (base_seed + tree_idx * 1_000_003) & 0x7FFFFFFF`，保证不同树初始 rollout 真正发散。
- counter rollback / forfeit 用 base_cache_key 而非 tree-specific key。
- `branch_indices` 在树内编号，leaf 写回时仍是全局唯一 traj_id（uuid4）。
