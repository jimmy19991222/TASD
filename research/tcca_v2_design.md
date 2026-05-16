# TCCA V2 — Iterative Teacher-Guided Chain Rollout

> **状态**：设计文档（待用户审）；审完再写代码
> **目的**：完整重写 rollout 路径，把 TCCA 从 "Mode B append" 升级为 "iterative chain"
> **作者**：与你 2026-05-16 19:50 对话整理
> **替代**：当前 `intervention_rollout.py:_do_real_intervention`（已知 ΔR≡0 bug）

---

## 1. Motivation

### 1.1 当前 V1 的 fatal flaw（commit `21f5da9`）

```
当前 composite y' = y_<t + teacher_k_tokens + y_>=t+k  (保留原 student tail)
                                                ↑
                              answer 在尾部，没换 → R(y') == R(y) → ΔR ≡ 0
```

实测 smoke `intervention/delta_reward_{mean,std,min,max,pos_rate}` 全 0。**ΔR 因果通道根本没工作**。

### 1.2 V2 设计原则（用户决策）

1. **OPSD-style teacher**：teacher 上下文加入 reference answer（"The correct answer is X"）
2. **chain rollout**：n=8 个 samples 按链式逐步产生（不是独立采样）
3. **单点 intervention**：每 iteration teacher 写 **k=1** 个 token
4. **student 真续写**：teacher 写完后 student 重新 rollout 到 EOS
5. **shared prefix 不算 loss**：`response_mask[:t_i] = 0` 对 y 和 y' 都生效
6. **完全替换标准 rollout**：`enable_intervention=True` → chain rollout, `False` → 退化为标准 GRPO
7. **chain_length=8 默认**，可配置（旧的 `top_k_positions` 重命名）

### 1.3 V2 vs V1 对比表

| 维度 | V1 (current) | V2 (this design) |
|---|---|---|
| rollout 结构 | 标准 GRPO n + Mode B append | 链式 n 个 derived sequentially |
| teacher context | EMA teacher, no ref | **OPSD: prompt + ref_answer + y_{i-1}** |
| intervention 时机 | rollout 后, 失败 sample 上 | **每 iteration, 每 prompt 都做** |
| teacher 写多少 token | k=2 | **k=1（最小干预）** |
| student tail | ❌ 用原 tail（hack）| ✅ **真 rollout 到 EOS** |
| 每 prompt sample 数 | n + n_failed (变长) | **固定 chain_length（默认 8）** |
| response_mask shared prefix | 不动 | **置 0** |
| 预计 compute | 1.5× GRPO | **~8× GRPO**（chain 串行） |

---

## 2. Pipeline overview

```
INPUT: prompt batch (B prompts), ref_answer per prompt
OUTPUT: augmented batch (B × chain_length samples, 每 prompt 的 chain_length 个 derived rollouts 同 uid)

Step 0: standard student rollout y_0 (每 prompt 1 个)
        → y_0_batch (B samples)

For i in 1..chain_length-1:
   Step A: 构造 OPSD teacher context (含 ref_answer)
   Step B: teacher forward → logp_T_opsd on y_{i-1}'s tokens
   Step C: student logp logp_S (from rollout_log_probs 或 _compute_old_log_prob)
   Step D: divergence_t = |logp_T_opsd - logp_S| · response_mask, 排尾 → t_i (per prompt)
   Step E: teacher 在 t_i 写 1 个 token z_t_i (OPSD ctx 下 argmax)
   Step F: 构造每个 prompt 的 student rollout prefix:
           prefix_i_j = original_prompt_j + y_{i-1}[j, :t_i_j] + z_t_i_j
   Step G: async student rollout (per-sample, variable prefix length):
           continuation_i_j = student.generate(prefix_i_j, max_new_tokens=remaining_budget)
   Step H: 构造 y_i_batch:
           y_i_j = y_{i-1}[j, :t_i_j] + z_t_i_j + continuation_i_j (pad to T)
           response_mask[j, :t_i_j] = 0  (Layer 2: shared prefix off)
           response_mask[j, t_i_j:t_i_j+1+len(continuation_i_j)] = 1
   Step I: reward_fn(y_i_batch) → R(y_i), 写入 token_level_scores

Step Final: augmented_batch = DataProto.concat([y_0_batch, y_1_batch, ..., y_{n-1}_batch])
            B × chain_length samples, 每 prompt 的 n 个 derived 同 uid
            交给标准 GRPO advantage 计算
```

---

## 3. 详细 pseudocode（per iteration）

### 3.1 数据结构约定

```python
P_orig = max_prompt_length                 # 原 prompt 左 pad 后的长度（固定 per batch）
T = max_response_length                    # response 右 pad 后的长度（固定 per batch）
P_opsd = max_prompt_length + ref_ans_len   # OPSD context 含 ref answer 的 prompt 长度
B = train_batch_size                       # 32

# 每个 chain iteration 产生 1 个 DataProto
y_i_batch.batch has:
  prompts          (B, P_orig)
  responses        (B, T)
  input_ids        (B, P_orig + T)
  attention_mask   (B, P_orig + T)
  position_ids     (B, P_orig + T)
  response_mask    (B, T)               # 关键: shared prefix 已置 0
  rollout_log_probs (B, T)              # 来自 vLLM (用于下一轮 divergence)
  old_log_probs    (B, T)               # 来自 _compute_old_log_prob
  token_level_scores (B, T)             # 来自 reward_fn
  token_level_rewards (B, T)            # 同 above (无 KL penalty 时)

y_i_batch.non_tensor_batch has:
  uid              (B,)  # ← 与原 prompt 同 uid (GRPO group 关键)
  data_source, reward_model, extra_info, raw_prompt
```

### 3.2 Step A — OPSD teacher context 构造

```python
def build_opsd_teacher_context(prompt_batch, tokenizer, config, ref_template):
    """
    在 system prompt 前置 'The correct answer is {r*}.\n', 再 apply chat template.

    Args:
        prompt_batch: DataProto, 含 non_tensor_batch['raw_prompt']
                      (raw_prompt[i] = list of {role, content} chat messages)
                      和 non_tensor_batch['reward_model'][i]['ground_truth'] (str)
        ref_template: e.g. "The correct answer is {r}.\n"

    Returns:
        opsd_input_ids:      (B, P_opsd)  # 左 pad
        opsd_attention_mask: (B, P_opsd)
        opsd_position_ids:   (B, P_opsd)
        P_opsd: int (the new prompt length after adding ref_answer)
    """
    B = len(prompt_batch)
    messages_with_ref = []
    for i in range(B):
        ref = prompt_batch.non_tensor_batch['reward_model'][i].get('ground_truth', '')
        raw_msgs = prompt_batch.non_tensor_batch['raw_prompt'][i]  # list of dict
        ref_text = ref_template.format(r=ref)
        # 把 ref_text 加到 user prompt 前缀
        new_msgs = [dict(m) for m in raw_msgs]
        if new_msgs[-1]['role'] == 'user':
            new_msgs[-1]['content'] = ref_text + new_msgs[-1]['content']
        else:
            new_msgs.append({'role': 'user', 'content': ref_text})
        messages_with_ref.append(new_msgs)

    opsd = tokenizer.apply_chat_template(
        messages_with_ref,
        tokenize=True, return_tensors='pt', return_dict=True,
        add_generation_prompt=True, padding=True, truncation=True,
        max_length=config.actor_rollout_ref.actor.self_distillation.max_reprompt_len,
    )
    P_opsd = opsd['input_ids'].shape[1]
    opsd_position_ids = compute_position_id_with_mask(opsd['attention_mask'])
    return opsd['input_ids'], opsd['attention_mask'], opsd_position_ids, P_opsd
```

### 3.3 Step B+C — Teacher / student logp on y_{i-1}

```python
def compute_divergence_with_opsd_teacher(y_prev_batch, opsd_ctx, actor_rollout_wg, config, tokenizer):
    """
    返回 (B, T) divergence = |logp_T_opsd - logp_S| · response_mask
    """
    # Step B: teacher forward 用 OPSD ctx
    # 拼接 teacher_input_ids = opsd_input_ids (P_opsd) + y_prev_batch.batch['responses'] (T)
    teacher_input_ids = torch.cat([opsd_ctx['input_ids'], y_prev_batch.batch['responses']], dim=1)
    teacher_attn = torch.cat([opsd_ctx['attention_mask'], y_prev_batch.batch['response_mask']], dim=1)
    teacher_pos = compute_position_id_with_mask(teacher_attn)

    teacher_fwd_batch = DataProto.from_dict(tensors={
        'teacher_input_ids': teacher_input_ids,
        'teacher_attention_mask': teacher_attn,
        'teacher_position_ids': teacher_pos,
        'responses': y_prev_batch.batch['responses'],
        'input_ids': y_prev_batch.batch['input_ids'],
        'attention_mask': y_prev_batch.batch['attention_mask'],
        'position_ids': y_prev_batch.batch['position_ids'],
    })
    teacher_fwd_batch.meta_info = {
        'temperature': config.actor_rollout_ref.rollout.temperature,
        'micro_batch_size': config.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu,
        'pad_token_id': tokenizer.pad_token_id,
        'distill_topk': None,
        'compute_prior_shift_surprise': False,
    }
    teacher_result = actor_rollout_wg.compute_teacher_log_probs(teacher_fwd_batch)
    logp_T_opsd = teacher_result.batch['teacher_log_probs_on_response']  # (B, T)

    # Step C: student logp
    # 优先用 rollout_log_probs (vLLM 算的); 否则需要 _compute_old_log_prob
    if 'rollout_log_probs' in y_prev_batch.batch:
        logp_S = y_prev_batch.batch['rollout_log_probs']
    elif 'old_log_probs' in y_prev_batch.batch:
        logp_S = y_prev_batch.batch['old_log_probs']
    else:
        raise ValueError("y_prev_batch 缺 student logp; 需要先调 _compute_old_log_prob")

    divergence = (logp_T_opsd.float() - logp_S.float()).abs()
    divergence = divergence * y_prev_batch.batch['response_mask'].float()
    return divergence, logp_T_opsd  # logp_T_opsd 复用于下一步 teacher_generate
```

### 3.4 Step D — 选 t_i

```python
def select_t_i(divergence, response_mask, exclude_tail=8):
    """
    argmax | logp_T - logp_S |, 排除尾部 exclude_tail token (避免选 EOS).
    Returns: (B,) long tensor of t_i in response coordinates.
    """
    B, T = divergence.shape
    L = response_mask.sum(dim=-1).long()
    idx = torch.arange(T, device=divergence.device).unsqueeze(0).expand(B, T)
    keep = idx < (L - exclude_tail).unsqueeze(-1).clamp(min=1)
    divergence = divergence * keep.float()
    return divergence.argmax(dim=-1)  # (B,)
```

### 3.5 Step E — teacher 写 1 token (OPSD ctx 下)

```python
# 复用现有 actor_rollout_wg.teacher_generate_at_positions(sub_proto)
# 关键: t_star_abs 在 OPSD ctx 坐标 (P_opsd + t_i), 不是 P_orig + t_i
def teacher_write_one_token(y_prev_batch, opsd_ctx, t_i, actor_rollout_wg, config, tokenizer):
    """
    Returns: teacher_tokens (B,) — teacher's argmax token at each t_i
    """
    P_opsd = opsd_ctx['input_ids'].shape[1]
    t_star_abs = (P_opsd + t_i).long()

    # 同样需要 FSDP chunk divisibility padding (复用 21f5da9 fix 逻辑)
    B = len(t_i)
    world_size = config.trainer.n_gpus_per_node
    pad_n = (world_size - B % world_size) % world_size
    if pad_n > 0:
        t_star_abs_padded = torch.cat([t_star_abs, t_star_abs[:pad_n]])
        teacher_input_ids = torch.cat([opsd_ctx['input_ids'], y_prev_batch.batch['responses']], dim=1)
        teacher_input_ids_padded = torch.cat([teacher_input_ids, teacher_input_ids[:pad_n]], dim=0)
        # ... 同样 pad attention_mask, position_ids
    else:
        ...

    sub_proto = DataProto.from_dict(tensors={
        'teacher_input_ids': teacher_input_ids_padded,
        'teacher_attention_mask': teacher_attn_padded,
        'teacher_position_ids': teacher_pos_padded,
        't_star_abs': t_star_abs_padded,
    }, meta_info={'k': 1, 'temperature': 0.0})  # greedy

    out = actor_rollout_wg.teacher_generate_at_positions(sub_proto)
    teacher_tokens = out.batch['intervention_tokens'][:B, 0]  # (B,) drop padding & k=1
    return teacher_tokens
```

### 3.6 Step F+G — student 真续写

```python
async def _student_continue_one(prefix_ids, max_tokens, server_manager, sampling):
    import uuid
    out = await server_manager.generate(
        request_id=uuid.uuid4().hex,
        prompt_ids=prefix_ids,
        sampling_params={**sampling, 'max_tokens': max(1, max_tokens)},
    )
    return out.token_ids  # list[int]

def student_continue_after_intervention(
    y_prev_batch, t_i, teacher_tokens, async_rollout_manager, config, tokenizer
):
    """
    每个 prompt:
      prefix = original_prompt (left-pad stripped) + y_prev[:t_i] + teacher_token
      调 server_manager.generate per-sample, max_tokens = T - t_i - 1
    Returns:
      list of (B,) continuation token id lists (variable length per sample)
    """
    import asyncio
    from verl.utils.ray_utils import get_event_loop  # or fallback

    B = len(t_i)
    T = y_prev_batch.batch['responses'].shape[1]
    P_orig = y_prev_batch.batch['input_ids'].shape[1] - T
    rollout_cfg = config.actor_rollout_ref.rollout
    sampling = dict(temperature=rollout_cfg.temperature, top_p=rollout_cfg.top_p)

    prefixes, budgets = [], []
    for j in range(B):
        # strip left padding from original prompt
        prompt_full = y_prev_batch.batch['input_ids'][j, :P_orig]
        prompt_mask = y_prev_batch.batch['attention_mask'][j, :P_orig].bool()
        prompt_ids = prompt_full[prompt_mask].tolist()
        # append y_prev[:t_j] and teacher token
        t_j = t_i[j].item()
        y_prefix_ids = y_prev_batch.batch['responses'][j, :t_j].tolist()
        prefix = prompt_ids + y_prefix_ids + [teacher_tokens[j].item()]
        prefixes.append(prefix)
        budgets.append(T - t_j - 1)  # 剩余 response 长度预算

    loop = get_event_loop()
    continuations = loop.run_until_complete(asyncio.gather(*[
        _student_continue_one(p, b, async_rollout_manager.server_manager, sampling)
        for p, b in zip(prefixes, budgets)
    ]))
    return continuations  # list[list[int]] length B, variable inner length
```

### 3.7 Step H — 构造 y_i_batch

```python
def build_y_i_batch(y_prev_batch, t_i, teacher_tokens, continuations, P_orig, T, tokenizer, ref_uids):
    """
    y_i[j] = y_prev[j, :t_i_j] + teacher_tokens[j] + continuations[j], pad 到 T
    response_mask[j, :t_i_j] = 0  (shared prefix)
    response_mask[j, t_i_j:t_i_j+1+len(continuations[j])] = 1
    """
    B = len(t_i)
    device = y_prev_batch.batch['responses'].device
    pad_id = tokenizer.pad_token_id

    new_responses = torch.full((B, T), pad_id, dtype=torch.long, device=device)
    new_response_mask = torch.zeros((B, T), dtype=torch.long, device=device)

    for j in range(B):
        t_j = t_i[j].item()
        # copy y_prev[j, :t_j]
        new_responses[j, :t_j] = y_prev_batch.batch['responses'][j, :t_j]
        # teacher token at t_j
        if t_j < T:
            new_responses[j, t_j] = teacher_tokens[j]
        # student continuation from t_j+1
        cont_ids = continuations[j]
        cont_start = t_j + 1
        cont_end = min(T, cont_start + len(cont_ids))
        if cont_end > cont_start:
            cont_take = cont_end - cont_start
            new_responses[j, cont_start:cont_end] = torch.tensor(
                cont_ids[:cont_take], dtype=torch.long, device=device
            )
        # response_mask: prefix [0, t_j) = 0; teacher+continuation = 1; pad = 0
        new_response_mask[j, t_j:cont_end] = 1

    # rebuild input_ids, attention_mask, position_ids
    prompts = y_prev_batch.batch['prompts']  # (B, P_orig), 同 prompt 复用
    prompt_mask = y_prev_batch.batch['attention_mask'][:, :P_orig]
    new_input_ids = torch.cat([prompts, new_responses], dim=1)
    new_attention = torch.cat([prompt_mask, new_response_mask], dim=1)
    new_position = compute_position_id_with_mask(new_attention)

    # copy non_tensor (重要: uid 保持与 prompt 一致, GRPO group 才能聚合)
    new_non_tensor = {k: v.copy() for k, v in y_prev_batch.non_tensor_batch.items()}

    return DataProto.from_dict(
        tensors={
            'prompts': prompts,
            'responses': new_responses,
            'response_mask': new_response_mask,
            'input_ids': new_input_ids,
            'attention_mask': new_attention,
            'position_ids': new_position,
        },
        non_tensor_batch=new_non_tensor,
        meta_info=dict(y_prev_batch.meta_info or {}),
    )
```

### 3.8 Step I — reward 复打分

```python
def rescore_reward_on_y_i(y_i_batch, reward_fn, device):
    # 删掉会让 reward 短路的旧字段
    for k in ['rm_scores', 'token_level_scores', 'token_level_rewards', 'advantages', 'returns']:
        if k in y_i_batch.batch.keys():
            y_i_batch.batch.pop(k)
    try:
        r = reward_fn(y_i_batch, return_dict=True)
        reward_tensor = r['reward_tensor'].to(device)
    except TypeError:
        reward_tensor = reward_fn(y_i_batch).to(device)
    y_i_batch.batch['token_level_scores'] = reward_tensor
    y_i_batch.batch['token_level_rewards'] = reward_tensor
    return y_i_batch
```

### 3.9 主函数

```python
def tcca_v2_chain_rollout(
    prompt_batch,        # 含 raw_prompt, reward_model, uid 的 DataProto, B prompts
    actor_rollout_wg,    # for teacher FSDP forward + generate
    async_rollout_manager,  # for student vLLM async generation
    reward_fn,           # for reward rescoring
    config,
    tokenizer,
) -> DataProto:
    chain_length = config.algorithm.intervention_credit.chain_length  # 默认 8
    enable_intervention = config.algorithm.intervention_credit.enable_intervention

    if not enable_intervention:
        # 退化: 标准 GRPO rollout n=chain_length
        return standard_rollout(prompt_batch, n=chain_length)

    # Step 0: y_0 = 标准 rollout n=1
    y_0_batch = standard_rollout(prompt_batch, n=1)
    chain_batches = [y_0_batch]

    # 构造 OPSD ctx 一次（同一个 prompt batch, 所有 iteration 共用）
    opsd_ctx = build_opsd_teacher_context(prompt_batch, tokenizer, config,
                                          ref_template="The correct answer is {r}.\n")

    for i in range(1, chain_length):
        y_prev = chain_batches[i - 1]
        # 3.3 divergence
        div, _ = compute_divergence_with_opsd_teacher(y_prev, opsd_ctx, actor_rollout_wg, config, tokenizer)
        # 3.4 t_i
        t_i = select_t_i(div, y_prev.batch['response_mask'], exclude_tail=8)
        # 3.5 teacher 写 1 token
        teacher_tokens = teacher_write_one_token(y_prev, opsd_ctx, t_i, actor_rollout_wg, config, tokenizer)
        # 3.6 student 续写
        continuations = student_continue_after_intervention(
            y_prev, t_i, teacher_tokens, async_rollout_manager, config, tokenizer
        )
        # 3.7 构造 y_i
        y_i = build_y_i_batch(y_prev, t_i, teacher_tokens, continuations,
                              P_orig=prompt_batch.batch['input_ids'].shape[1] - chain_batches[0].batch['responses'].shape[1],
                              T=chain_batches[0].batch['responses'].shape[1],
                              tokenizer=tokenizer, ref_uids=...)
        # 3.8 reward
        y_i = rescore_reward_on_y_i(y_i, reward_fn, device=y_prev.batch['responses'].device)
        chain_batches.append(y_i)

    augmented = DataProto.concat(chain_batches)
    # B * chain_length samples, 同 prompt 的 chain_length 个同 uid → GRPO group 自然处理
    return augmented
```

---

## 4. Integration with ray_trainer.py

### 4.1 调用位置

当前 ray_trainer.py:1989 是 `generate_sequences` 主调用。把它替换为：

```python
if self.config.algorithm.adv_estimator == 'intervention_credit' and \
   self.config.algorithm.intervention_credit.get('enable_intervention', False):
    # TCCA V2 chain rollout
    from verl.trainer.ppo.bayesian_credit.tcca_chain import tcca_v2_chain_rollout
    gen_batch_output = tcca_v2_chain_rollout(
        prompt_batch=gen_batch,
        actor_rollout_wg=self.actor_rollout_wg,
        async_rollout_manager=self.async_rollout_manager,
        reward_fn=self.reward_fn,
        config=self.config,
        tokenizer=self.tokenizer,
    )
    # gen_batch_output 已含 chain_length × B samples, 同 uid 已 set
    # 跳过 `batch = batch.repeat(rollout.n)` (已经有 n=chain_length 个 sample)
else:
    # 标准 GRPO rollout
    gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch_output)
```

### 4.2 跳过 Mode B append (intervention_rollout 不再调)

ray_trainer.py:2230 现有 `if adv_estimator == 'intervention_credit': run_teacher_intervention_rollout(...)` 块**整个删掉**（chain rollout 已经在 step 0 之前替代了它）。

### 4.3 删 c_t / token_causal_credit / intervention_delta_reward

intervention_credit.py 公式回退到 GRPO + 可选 base_reweight：

```python
A_seq = R_i - mean_group(R)       # GRPO group-relative
A_t   = A_seq · base_reweight · response_mask · length_scale
        ↑                          ↑
        Layer 3 暂用最简 GRPO    response_mask 在 chain rollout 时已设好 prefix=0
```

c_t / token_causal_credit / lambda_token_credit 这些字段**全部移除**（Layer 3 等以后重谈再加回）。

---

## 5. Cost analysis (修正版 2026-05-16 20:00)

### 5.1 之前估算错在哪里

旧版假设 "每 iter 都是完整 rollout 时间"，但忽略：
1. **t_i 渐进后移**：teacher 早期 token 被前几轮"矫正"后，下一轮 divergence 自然落在更后位置
   → iter i 的 continuation 长度 ≈ T × (1 - i/n), 不是 T
2. **vLLM batch 并行**：32 prompts 同 batch async_gather，wall-clock ≈ 单 prompt 时间，不是 32× 串行

### 5.2 修正后的计算（chain_length=8 为例）

continuation 长度等比衰减：
```
y_0:   1.00 T_rollout  (标准 rollout)
y_1:   0.90 T_rollout  (teacher 修早期 ~10% 位置 → 续写大部分)
y_2:   0.80 T_rollout
y_3:   0.70 T_rollout
y_4:   0.60 T_rollout
y_5:   0.50 T_rollout
y_6:   0.40 T_rollout
y_7:   0.25 T_rollout
─────────────────
total ≈ 5.15 T_rollout
```

加上 teacher forward (每 iter ~3s, 8 iter = 24s, 但其实跟 student rollout 串行，所以已包含在上面 wall-clock 内的小份额) → **总 rollout overhead ≈ 5.2× baseline**

### 5.3 单 step + 250-step job 时长

| chain_length | rollout overhead | 单 step (rollout + train) | 250-step job 时长 | vs baseline |
|---|---|---|---|---|
| baseline GRPO n=8 | 1.0× | 30s + 70s = 100s | **~7h** | 1.0× |
| TCCA n=2 | 1.9× | 57s + 70s = 127s | ~8.8h | 1.26× |
| TCCA n=4 | 3.4× | 102s + 70s = 172s | ~12h | 1.7× |
| TCCA n=6 | 4.5× | 135s + 70s = 205s | ~14h | 2.0× |
| **TCCA n=8** | **5.15×** | **155s + 70s = 225s** | **~15-16h** | **2.2×** |

**结论**：chain_length=8 大约 **15-16h/job**, ~2.2× baseline, **完全可接受**。
之前 "8× / 104h" 严重高估了, 用户直觉是对的。

### 5.4 资源规划建议

- **chain_length=8 默认即可**（之前担心是冤枉）
- 单 job ~15h, 配额内能跑 1-2 job 并行
- 论文 ablation: 跑 chain_length ∈ {2, 4, 8} 看 marginal return
  - 2: 9h × 1 job (快速验证)
  - 4: 12h × 1 job (中等)
  - 8: 15h × 1 job (论文 main)

---

## 6. 实施风险

| 风险 | 缓解 |
|---|---|
| 🚨 `async_rollout_manager.server_manager.generate` API 没用过, 接口可能与文档不符 | smoke 先验证 server_manager.generate 能正常返回 |
| 🚨 chain rollout 替换标准 rollout 影响 ray_trainer 主循环, 改动大 | 加 enable_intervention 开关，False 时完全退化 |
| 🟡 OPSD ctx 长度 P_opsd 可能超过 max_reprompt_len | clip/truncate ref_answer 或加 length check |
| 🟡 student continuation 可能比预算长 (vLLM ignore max_tokens?) | sampling_params 含 max_tokens, vLLM 会 respect |
| 🟡 P_orig / T 在 chain iteration 间是否一致? (是, 因为 prompt_batch 不变) | OK |
| 🟡 FSDP chunk divisibility (复用 21f5da9 fix) | 已有方案 |
| 🟢 reward function 是否能正确处理 composite | naive_reward_manager 只 decode response, OK |

---

## 7. Open questions（待 user 决定）

1. ~~**chain_length 默认值**：8（你的初衷） vs 4-6（折中 compute）？~~ → **修正后 ~15h/job, 直接用 8**
2. **Step 0 的 y_0**：用标准 generate_sequences 还是直接 async_rollout_manager？
3. **Iteration i 的 t_i 选择**：是否允许 t_i 和 t_{i-1} 接近？（已在 t_i 在 y_{i-1} 上选, 自动是新位置, 不需要 min_gap）
4. **OPSD ref_template 模板**：`"The correct answer is {r}.\n"` 还是更详细的？
5. **失败 fallback**：如果某个 prompt 在 iteration 5 续写空了（continuation = []）, 接下来怎么办？跳过剩余 iter 用 y_5？
6. ~~**Compute budget**：你的开发机配额能支持 chain_length=8 的 250-step job (~100h)？~~ → **修正后 ~15h/job, 配额内 OK**

---

## 8. 实施分阶段（合 user "先文档再代码"原则）

| Phase | 内容 | 时长 |
|---|---|---|
| **P0** (now) | 本文档 + pseudocode 你审 | 已完成 |
| **P1** | 写 `tcca_chain.py` 新文件（包含本文档 §3 所有函数） | 0.5 天 |
| **P2** | smoke 验证 server_manager.generate API 能用 | 0.3 天 |
| **P3** | 集成到 ray_trainer.py + 删掉旧 intervention_rollout 调用 | 0.5 天 |
| **P4** | 端到端 smoke (chain_length=2 极简版) | 0.3 天 |
| **P5** | chain_length=8 full smoke + debug | 0.5 天 |
| **总计** | | **2-3 天** |

---

## 9. 论文 final framing

> **TCCA: Iterative Teacher-Guided Chain Rollout with OPSD-Conditioned Counterfactuals**
>
> We propose TCCA, an iterative rollout strategy that produces N derived samples per prompt through teacher-guided single-token interventions. Each iteration: (1) teacher with privileged reference-answer context identifies the maximum-divergence position in the previous rollout, (2) replaces 1 token with its argmax under the OPSD context, (3) student re-rolls from the corrected prefix. The resulting N-sample chain forms a contrastive group naturally compatible with GRPO group-relative advantage, with shared prefixes masked to prevent gradient duplication. TCCA's iterative structure enables progressive refinement under teacher's privileged supervision while keeping student updates anchored to environment reward.

---

## 等你 review

请 check 以下几点是否对齐：

- [ ] §1.2 设计原则 7 条都对吗？
- [ ] §2 pipeline 流程对吗？
- [ ] §3 每一步的 pseudocode 数据流对吗？特别是 OPSD ctx 坐标（P_opsd vs P_orig）、response_mask 处理
- [ ] §4 ray_trainer integration: 用 enable_intervention 开关 + 删 c_t 等机制 OK 吗？
- [ ] §5 compute cost 你能接受吗？要不要 chain_length 默认改小？
- [ ] §7 open questions 6 条，哪些先决定？

任何不对的告诉我，我修文档后再写代码。
