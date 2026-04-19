# TASD: From Reward to Loss

本文档描述 TASD 算法中 token-level reward 从计算到最终 policy loss 的完整链路，
以及各实验参数对链路的影响。

---

## 1. 整体流程

```mermaid
flowchart TB
    subgraph Teacher["Teacher Forward"]
        T_FWD["teacher & student forward<br/>获取 log_probs + topk_log_probs"]
    end

    subgraph Reward["Token Reward 计算<br/>compute_tasd_token_rewards"]
        RT["reward_type"] --> R_BASE["基础 reward"]
        EG["entropy_gate"] --> GATE["entropy gate 加权<br/>(乘到 reward 上)"]
        R_BASE --> GATE --> R_OUT["token_level_rewards"]
    end

    subgraph Advantage["Advantage 计算<br/>compute_tasd_advantage"]
        RM["response_mask"] --> EM["effective_mask"]
        SDM["self_distillation_mask"] --> EM
        GM["gate_mask"] --> EM
        EM --> GROUP["group_mean / group_std"]
        GROUP --> ADV_RAW["adv = (r - mean) / std"]
        CA["clip_adv"] --> ADV_CLIP["advantage clipping"]
        AEW["adv_entropy_weight"] --> ADV_W["advantage 熵加权<br/>(乘到 advantage 上)"]
        ADV_RAW --> ADV_CLIP --> ADV_W --> ADV_OUT["advantages"]
    end

    subgraph Loss["Policy Loss 计算<br/>compute_policy_loss_vanilla"]
        CR["clip_ratio / clip_ratio_low / clip_ratio_high"] --> PPO["PPO clipped loss"]
        PPO --> LOSS["pg_loss"]
    end

    T_FWD --> Reward
    R_OUT --> Advantage
    ADV_OUT --> Loss
```

---

## 2. Stage 1: Token Reward 计算

**函数**: `compute_tasd_token_rewards` in `core_algos.py`

```mermaid
flowchart TB
    SL["student_log_probs (B,T)"] --- TL["teacher_log_probs (B,T)"]
    STK["student_topk_log_probs (B,T,K)"] --- TTK["teacher_topk_log_probs (B,T,K)"]

    subgraph RewardType["reward_type 选择"]
        direction TB
        TP["teacher_prob<br/>r = exp(teacher_log_prob)<br/>值域 (0, 1]"]
        TLP["teacher_log_prob<br/>r = teacher_log_prob<br/>值域 (-∞, 0]"]
    end

    SL & TL --> RewardType
    R_BASE["reward (B,T)"]

    TP --> R_BASE
    TLP --> R_BASE

    subgraph EntropyGate["entropy_gate (需要 topk)"]
        direction TB
        NONE_G["none → 不修改 reward"]
        HARD_G["hard → gate_mask = (H_t < H_s)<br/>reward *= gate_mask"]
        SOFT_G["soft → w = (H_s - H_t)⁺ / max<br/>reward *= w"]
        SOFTV2_G["soft_v2 → mask = (H_t < H_s)<br/>w = mask × (1 - H_t_norm)<br/>reward *= w"]
    end

    R_BASE --> EntropyGate

    subgraph EntropyCalc["熵计算 (entropy_gate≠none 或 adv_entropy_weight≠none)"]
        direction TB
        EC1["H_t = -Σ p_t × log(p_t)"]
        EC2["H_s = -Σ p_s × log(p_s)"]
        EC3["H_t_norm = H_t / log(K)"]
        EC4["H_s_norm = H_s / log(K)"]
    end

    STK & TTK --> EntropyCalc
    EntropyCalc --> EntropyGate

    EntropyGate --> R_OUT["token_level_rewards (B,T)"]
    EntropyCalc --> TEN["teacher_entropy_norm (B,T)"]
    EntropyCalc --> SEN["student_entropy_norm (B,T)"]
```

### 参数说明

| 参数 | 值域 | 默认 | 说明 |
|------|------|------|------|
| `reward_type` | `teacher_prob` / `teacher_log_prob` | `teacher_prob` | reward 计算方式 |
| `entropy_gate` | `none` / `hard` / `soft` / `soft_v2` | `none` | 熵门控模式 |
| `distill_topk` | int | 100 | top-k 规模（熵计算需要） |

**关键**：`entropy_gate` 在 **reward 阶段** 乘权重，会影响后续 `group_mean` 的计算。

---

## 3. Stage 2: Advantage 计算

**函数**: `compute_tasd_advantage` in `core_algos.py`

```mermaid
flowchart TB
    TLR["token_level_rewards (B,T)"]

    subgraph EffectiveMask["effective_mask 构建"]
        direction TB
        R_M["response_mask<br/>(排除 padding)"]
        SD_M["self_distillation_mask<br/>(排除无 teacher context 的样本)"]
        G_M["gate_mask<br/>(排除 entropy gate 掉的 token)"]
        EM["effective_mask = R_M × SD_M × G_M"]
    end

    TLR --> GROUP
    EM --> GROUP

    subgraph GroupNorm["Group 归一化"]
        direction TB
        GM["group_mean = mean(r[effective])"]
        STD_OPT{"norm_adv_by_std?"}
        STD_YES["group_std = std(r[effective])<br/>std_floor 保护"]
        STD_NO["不除以 std"]
        SF["adv_std_floor<br/>none / auto / float"]
        STD_OPT -->|true| STD_YES
        STD_OPT -->|false| STD_NO
        STD_YES --> SF
        SF --> FLOOR_NONE["none/0.0 → floor=1e-8"]
        SF --> FLOOR_AUTO["auto → floor=1/√N"]
        SF --> FLOOR_FLOAT["float → floor=value"]
    end

    GROUP["adv_i = r_i - group_mean"]
    GROUP --> DIV["adv_i /= group_std (可选)"]
    DIV --> ADV_RAW["advantages (B,T)"]

    subgraph AdvClip["Advantage Clipping"]
        CA_OPT{"clip_adv?"}
        CA_YES["clamp(adv, -clip_adv_value, +clip_adv_value)"]
        CA_NO["不 clip"]
        CA_OPT -->|true| CA_YES
        CA_OPT -->|false| CA_NO
    end

    ADV_RAW --> AdvClip

    subgraph AdvEntropyWeight["Advantage 熵加权 (需要 topk)"]
        AEW_OPT{"adv_entropy_weight?"}
        AEW_NONE["不加权"]
        AEW_TE["teacher_entropy<br/>w = 1 - H_t_norm"]
        AEW_ED["entropy_diff<br/>w = (H_s - H_t)⁺ / max"]
        AEW_HTT["hard_then_weight<br/>w = (H_t < H_s) × (1 - H_t_norm)"]
        AEW_OPT -->|none| AEW_NONE
        AEW_OPT -->|teacher_entropy| AEW_TE
        AEW_OPT -->|entropy_diff| AEW_ED
        AEW_OPT -->|hard_then_weight| AEW_HTT
    end

    AdvClip --> AdvEntropyWeight
    TEN["teacher_entropy_norm"] --> AdvEntropyWeight
    SEN["student_entropy_norm"] --> AdvEntropyWeight

    AdvEntropyWeight --> ADV_OUT["final advantages (B,T)"]
```

### 参数说明

| 参数 | 值域 | 默认 | 说明 |
|------|------|------|------|
| `norm_adv_by_std` | `true` / `false` | `false` | 是否除以 group std |
| `adv_std_floor` | `none` / `auto` / float | `0.0` | std 下界保护 |
| `clip_adv` | `true` / `false` | `true` | 是否 clip advantage |
| `clip_adv_value` | float | `2.0` | advantage clip 范围 [-v, +v] |
| `adv_entropy_weight` | `none` / `hard_filter` / `teacher_conf` / `teacher_conf_filtered` / `certainty_diff_filtered` | `none` | advantage 阶段熵加权 |

**关键**：`adv_entropy_weight` 在 **advantage 阶段** 乘权重，只影响梯度幅度，**不影响 group_mean**。所有模式统一范式：先过滤（teacher 比 student 更确定的位置才保留），再加权。

---

## 4. Stage 3: Policy Loss 计算

**函数**: `compute_policy_loss_vanilla` (TASD 注册为 `loss_mode="tasd"`)

```mermaid
flowchart TB
    OLP["old_log_prob (B,T)"]
    LP["log_prob (B,T)"]
    ADV["advantages (B,T)"]
    RM["response_mask (B,T)"]

    OLP & LP --> RATIO["ratio = exp(log_prob - old_log_prob)"]

    ADV --> LOSS1["L1 = -adv × ratio"]
    RATIO --> LOSS2["L2 = -adv × clamp(ratio, 1-ε_low, 1+ε_high)"]

    LOSS1 & LOSS2 --> CLIP1["clip_loss = max(L1, L2)"]

    LOSS3["L3 = -adv × clip_ratio_c"]
    CLIP1 & LOSS3 --> DUAL_CLIP["dual-clip: adv<0 → min(L3, clip_loss)"]

    DUAL_CLIP --> FINAL_LOSS["pg_losses (B,T)"]
    RM --> AGG["agg_loss(pg_losses, response_mask)"]
    FINAL_LOSS --> AGG --> PG_LOSS["pg_loss (scalar)"]
```

### 参数说明

| 参数 | 值域 | 默认 | 说明 |
|------|------|------|------|
| `clip_ratio` | float | `0.2` | 标准 PPO clip ε |
| `clip_ratio_low` | float | `0.2` | 下界 clip |
| `clip_ratio_high` | float / 10000 | `10000` | 上界 clip（10000=Clip-Higher） |
| `entropy_coeff` | float | `0.0` | 熵奖励系数 |
| `loss_agg_mode` | `token-mean` | `token-mean` | loss 聚合方式 |

---

## 5. 参数组合与效果速查

### 5.1 entropy_gate vs adv_entropy_weight

```mermaid
flowchart LR
    subgraph EG_Reward["entropy_gate: reward 阶段"]
        EG_NONE["none: reward 不变"]
        EG_HARD["hard: reward × {0,1}"]
        EG_SOFT["soft: reward × [0,1]"]
        EG_SOFTV2["soft_v2: reward × mask×(1-H_t)"]
    end

    subgraph AEW_Adv["adv_entropy_weight: advantage 阶段"]
        AEW_NONE2["none: adv 不变"]
        AEW_TE["teacher_entropy: adv × (1-H_t)"]
        AEW_ED["entropy_diff: adv × (H_s-H_t)⁺"]
        AEW_HTT["hard_then_weight: adv × mask×(1-H_t)"]
    end

    EG_Reward -.->|影响 group_mean| AEW_Adv
    AEW_Adv -.->|只影响梯度幅度| LOSS["Loss"]
```

**核心区别**：

| 维度 | entropy_gate (reward 阶段) | adv_entropy_weight (adv 阶段) |
|------|---------------------------|-------------------------------|
| 作用位置 | reward 计算 | advantage 计算 |
| 是否影响 group_mean | **是**（reward 被缩放后 mean 也变） | **否**（adv 减去的 mean 不变） |
| 语义 | "不重要的 token 给小 reward" | "不重要的 token 给小梯度" |
| 统一范式 | 各模式逻辑不同 | 统一：先 hard filter，再加权 |
| 可组合 | 可与 adv_entropy_weight 同时开启 | 可与 entropy_gate 同时开启 |

### 5.2 常见实验配置

| 配置名 | reward_type | entropy_gate | norm_adv_by_std | clip_adv | adv_entropy_weight | 效果描述 |
|--------|-------------|-------------|-----------------|----------|-------------------|---------|
| 基础 TASD | teacher_log_prob | none | true | false | none | 标准 TASD，无熵门控，std 归一化 |
| Hard Gate | teacher_log_prob | hard | true | true | none | 只保留 teacher 更确定的位置 |
| Soft Gate | teacher_log_prob | soft | true | true | none | 按熵差连续加权 reward |
| Adv 硬过滤 | teacher_log_prob | none | true | false | hard_filter | 只保留 teacher 更确定位置的 adv，不加额外权 |
| Adv teacher 确定性 | teacher_log_prob | none | true | false | teacher_conf | 不过滤，全部 token 按 teacher 确定性加权 |
| Adv teacher 确定性(过滤) | teacher_log_prob | none | true | false | teacher_conf_filtered | 先过滤，teacher 越确定 adv 梯度越大 |
| Adv 确定性差值 | teacher_log_prob | none | true | false | certainty_diff_filtered | 先过滤，teacher 比 student 多确定多少，梯度越大 |
| Gate + Adv 加权 | teacher_log_prob | hard | true | true | certainty_diff_filtered | reward 阶段 hard 过滤 + adv 阶段熵差加权 |

### 5.3 adv_std_floor 三态

```mermaid
flowchart TB
    FLOOR_OPT{"adv_std_floor?"}
    F_NONE["none / 0.0<br/>floor = 1e-8<br/>几乎无保护"]
    F_AUTO["auto<br/>floor = 1/√N<br/>N = group 内有效 token 数"]
    F_FLOAT["float (如 0.1)<br/>floor = 该值<br/>固定保护"]
    FLOOR_OPT --> F_NONE
    FLOOR_OPT --> F_AUTO
    FLOOR_OPT --> F_FLOAT

    F_NONE & F_AUTO & F_FLOAT --> STD["group_std = std.clamp(min=floor)"]
    STD --> ADV["adv = (r - mean) / group_std"]
```

---

## 6. 完整数据流

```mermaid
flowchart TB
    subgraph Inputs["模型输入"]
        PROMPT["prompt (same for T & S)"]
        RESP["response (generated)"]
    end

    subgraph Forward["双路 Forward"]
        S_FWD["Student Forward<br/>→ student_log_probs (B,T)<br/>→ student_topk_log_probs (B,T,K)"]
        T_FWD["Teacher Forward<br/>(EMA / self-as-teacher)<br/>→ teacher_log_probs (B,T)<br/>→ teacher_topk_log_probs (B,T,K)"]
    end

    PROMPT & RESP --> S_FWD
    PROMPT & RESP & SOLUTION["successful solution / feedback"] --> T_FWD

    subgraph Reward_Stage["Stage 1: Token Reward"]
        R_TYPE{"reward_type?"}
        R_TP["r = exp(t_log_prob) ∈ (0,1]"]
        R_TLP["r = t_log_prob ∈ (-∞,0]"]
        R_TYPE -->|teacher_prob| R_TP
        R_TYPE -->|teacher_log_prob| R_TLP

        E_GATE{"entropy_gate?"}
        E_NONE["r 不变"]
        E_HARD["r × (H_t < H_s)"]
        E_SOFT["r × (H_s-H_t)⁺/max"]
        E_SV2["r × (H_t<H_s) × (1-H_t_norm)"]
        E_GATE -->|none| E_NONE
        E_GATE -->|hard| E_HARD
        E_GATE -->|soft| E_SOFT
        E_GATE -->|soft_v2| E_SV2
    end

    S_FWD & T_FWD --> Reward_Stage

    subgraph Adv_Stage["Stage 2: Advantage"]
        E_MASK["effective_mask =<br/>response_mask × sdist_mask × gate_mask"]
        G_MEAN["group_mean = mean(r[eff])"]
        G_STD["group_std (optional)<br/>with adv_std_floor"]
        ADV_C["adv = (r - mean) / std<br/>× effective_mask"]
        CLIP_ADV["clip_adv →<br/>clamp(adv, -v, +v)"]
        AEW{"adv_entropy_weight?"}
        AEW_N["adv 不变"]
        AEW_TE["adv × (1-H_t_norm)"]
        AEW_ED["adv × (H_s-H_t)⁺/max"]
        AEW_HTT["adv × (H_t<H_s)×(1-H_t)"]
        AEW -->|none| AEW_N
        AEW -->|teacher_entropy| AEW_TE
        AEW -->|entropy_diff| AEW_ED
        AEW -->|hard_then_weight| AEW_HTT
    end

    Reward_Stage --> Adv_Stage

    subgraph Loss_Stage["Stage 3: Policy Loss"]
        RATIO["ratio = exp(π_θ - π_old)"]
        PPO_CLIP["PPO clipped loss<br/>clip(ratio, 1-ε_low, 1+ε_high)"]
        ENT["entropy_bonus<br/>entropy_coeff × H(π_θ)"]
        TOTAL["pg_loss - entropy_bonus"]
    end

    Adv_Stage --> Loss_Stage
```

---

## 7. 代码索引

| 阶段 | 文件 | 函数 |
|------|------|------|
| Token Reward | `verl/trainer/ppo/core_algos.py` | `compute_tasd_token_rewards` |
| Advantage | `verl/trainer/ppo/core_algos.py` | `compute_tasd_advantage` |
| Policy Loss | `verl/trainer/ppo/core_algos.py` | `compute_policy_loss_vanilla` (注册为 `tasd`) |
| Teacher Forward | `verl/trainer/ppo/ray_trainer.py` | `_maybe_build_self_distillation_batch` + `compute_teacher_log_probs` |
| 配置 | `verl/trainer/config/tasd_simple.yaml` | `algorithm.tasd.*` |
| Sweep 脚本 | `nebula_scripts/submit_tasd_simple_sweep.sh` | 所有 `*_LIST` 数组 |
| 参数化脚本 | `nebula_scripts/tasd_simple/tasd_simple_parametric.sh` | Hydra override |
