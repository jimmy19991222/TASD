"""
实验结果分析脚本
分析 实验结果.json 中的所有指标，输出多维度对比报告（Markdown 格式）
"""

import json
import sys
from collections import defaultdict

# ============ 加载数据 ============
with open("实验结果.json") as f:
    data = json.load(f)

# 输出目标：标准输出重定向到 md 文件
OUTPUT_FILE = "实验结果分析.md"
_out = open(OUTPUT_FILE, "w", encoding="utf-8")

def md(*args, **kwargs):
    """打印到 md 文件，同时也打印到终端"""
    print(*args, **kwargs)
    print(*args, **kwargs, file=_out)


def get_metric(exp_metrics, key):
    """获取某个实验的某个指标的最终值（step最大时）"""
    for item in exp_metrics:
        if item["key"] == key:
            try:
                return float(item["value"])
            except (ValueError, TypeError):
                return None
    return None


def get_metric_peak(exp_metrics, key, mode="max"):
    """获取某个指标的历史最优值（来自 min/max 字段）"""
    for item in exp_metrics:
        if item["key"] == key:
            try:
                if mode == "max" and "max" in item:
                    return float(item["max"]["data"])
                elif mode == "min" and "min" in item:
                    return float(item["min"]["data"])
            except (ValueError, TypeError, KeyError):
                return None
    return None


def get_metric_best(exp_metrics, key, mode="max"):
    """获取某个指标的历史最优值（扫描所有 step）"""
    best_val = None
    best_step = None
    for item in exp_metrics:
        if item["key"] == key:
            try:
                val = float(item["value"])
                if best_val is None:
                    best_val = val
                    best_step = item.get("step", 0)
                elif mode == "max" and val > best_val:
                    best_val = val
                    best_step = item.get("step", 0)
                elif mode == "min" and val < best_val:
                    best_val = val
                    best_step = item.get("step", 0)
            except (ValueError, TypeError):
                continue
    return best_val, best_step


# ============ 实验名清洗（去时间戳和模型名后缀） ============
def shorten_name(name):
    # 去掉时间戳（-20260xxx 及之后）
    import re
    name = re.sub(r'-202\d{5}_\d{6}$', '', name)
    # 去掉模型名后缀
    name = name.replace('-Qwen3-8B', '')
    return name


experiments = {shorten_name(k): v for k, v in data.items()}
exp_names = list(experiments.keys())

md("# 实验结果分析报告\n")

md("## 实验列表\n")
for i, name in enumerate(exp_names):
    md(f"- [{i+1}] `{name}`")

# ============ 关键指标列表 ============
METRICS_OF_INTEREST = {
    # 准确率类（统一用 @16）
    "acc/mean@16":      "val-core/sciknoweval/acc/mean@16",
    "acc/best@16":      "val-core/sciknoweval/acc/best@16/mean",
    "acc/maj@16":       "val-core/sciknoweval/acc/maj@16/mean",
    "acc/worst@16":     "val-core/sciknoweval/acc/worst@16/mean",
    "acc/std@16":       "val-core/sciknoweval/acc/std@16",
    # 训练稳定性
    "entropy":          "actor/entropy",
    "grad_norm":        "actor/grad_norm",
    "pg_clipfrac":      "actor/pg_clipfrac",
    "pg_clipfrac_lower": "actor/pg_clipfrac_lower",
    "ppo_kl":           "actor/ppo_kl",
    "pg_loss":          "actor/pg_loss",
    # Advantage 分布
    "adv/mean":         "critic/advantages/mean",
    "adv/max":          "critic/advantages/max",
    "adv/min":          "critic/advantages/min",
    # Returns 分布
    "returns/mean":     "critic/returns/mean",
    "returns/max":      "critic/returns/max",
    "returns/min":      "critic/returns/min",
    # Reward 分布
    "reward/mean":      "critic/rewards/mean",
    "reward/max":       "critic/rewards/max",
    "reward/min":       "critic/rewards/min",
    # Score 分布（训练）
    "score/mean":       "critic/score/mean",
    "score/max":        "critic/score/max",
    "score/min":        "critic/score/min",
    # Score 分布（val，统一用 @16）
    "score/mean@16":    "val-core/sciknoweval/score/mean@16",
    "score/best@16":    "val-core/sciknoweval/score/best@16/mean",
    "score/worst@16":   "val-core/sciknoweval/score/worst@16/mean",
    # 格式错误率
    "fmt_err/mean@16":  "val-aux/sciknoweval/incorrect_format/mean@16",
    # 序列长度
    "seq_len/mean":     "global_seqlen/mean",
    "seq_len/max":      "global_seqlen/max",
    "seq_len/min":      "global_seqlen/min",
    "seq_len/balanced_max": "global_seqlen/balanced_max",
    "seq_len/balanced_min": "global_seqlen/balanced_min",
    "seq_len/minmax_diff": "global_seqlen/minmax_diff",
    # rollout 相关
    "rollout_ppl":      "rollout_corr/rollout_ppl",
    "rollout_probs_diff_mean": "training/rollout_probs_diff_mean",
    # reward/acc 分布的 diversity（统一用 @16）
    "reward/std@16":    "val-core/sciknoweval/reward/std@16",
    # TASD 特有指标
    "tasd/adv_neg_ver_neg_rate": "tasd/adv_neg_ver_neg_rate",
    "tasd/adv_neg_ver_pos_rate": "tasd/adv_neg_ver_pos_rate",
    "tasd/adv_pos_ver_neg_rate": "tasd/adv_pos_ver_neg_rate",
    "tasd/adv_pos_ver_pos_rate": "tasd/adv_pos_ver_pos_rate",
    "tasd/adv_ver_agree_rate":   "tasd/adv_ver_agree_rate",
    "tasd/advantage_mean":       "tasd/advantage_mean",
    "tasd/advantage_std":        "tasd/advantage_std",
    "tasd/advantage_pos_rate":   "tasd/advantage_pos_rate",
    "tasd/success_rate":         "tasd/success_rate",
    "tasd/token_reward_mean":    "tasd/token_reward_mean",
}

# ============ 构建数据表 ============
table = defaultdict(dict)
for exp_name, exp_metrics in experiments.items():
    for metric_alias, metric_key in METRICS_OF_INTEREST.items():
        val = get_metric(exp_metrics, metric_key)
        table[exp_name][metric_alias] = val

def fmt(v, decimals=4):
    return f"{v:.{decimals}f}" if v is not None else "N/A"

def md_table(headers, rows):
    """输出 Markdown 表格"""
    md("| " + " | ".join(headers) + " |")
    md("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        md("| " + " | ".join(str(c) for c in row) + " |")

# ============ 1. 准确率历史最佳 ============
md("\n## 1. 准确率历史最佳（max.data）\n")
headers = ["实验", "mean@16 best", "mean@16 final", "best@16 best", "best@16 final"]
rows = []
for name, exp_metrics in experiments.items():
    mean_best = get_metric_peak(exp_metrics, "val-core/sciknoweval/acc/mean@16", mode="max")
    mean_final = get_metric(exp_metrics, "val-core/sciknoweval/acc/mean@16")
    best_best = get_metric_peak(exp_metrics, "val-core/sciknoweval/acc/best@16/mean", mode="max")
    best_final = get_metric(exp_metrics, "val-core/sciknoweval/acc/best@16/mean")
    rows.append([name, fmt(mean_best), fmt(mean_final), fmt(best_best), fmt(best_final)])
md_table(headers, rows)

# ============ 2. mean@16 vs best@16 gap ============
md("\n## 2. mean@16 vs best@16 gap（gap越小=模型越稳定）\n")
headers = ["实验", "mean@16 best", "best@16 best", "gap", "consistency%"]
rows = []
for name, exp_metrics in experiments.items():
    m = get_metric_peak(exp_metrics, "val-core/sciknoweval/acc/mean@16", mode="max")
    b = get_metric_peak(exp_metrics, "val-core/sciknoweval/acc/best@16/mean", mode="max")
    if m is not None and b is not None:
        gap = b - m
        cons = m / b * 100 if b > 0 else 0
        rows.append([name, fmt(m), fmt(b), fmt(gap), f"{cons:.1f}%"])
    else:
        rows.append([name, "N/A", "N/A", "N/A", "N/A"])
md_table(headers, rows)

# ============ 3. 训练稳定性 ============
md("\n## 3. 训练稳定性（最终step）\n")
headers = ["实验", "entropy", "grad_norm", "pg_clipfrac", "ppo_kl"]
rows = []
for name in exp_names:
    rows.append([name] + [fmt(table[name].get(m)) for m in ["entropy", "grad_norm", "pg_clipfrac", "ppo_kl"]])
md_table(headers, rows)

# ============ 4. Entropy 历史 ============
md("\n## 4. Entropy 历史（越低越接近 mode collapse）\n")
headers = ["实验", "init (max)", "final", "min", "drop"]
rows = []
for name, exp_metrics in experiments.items():
    ent_min = get_metric_peak(exp_metrics, "actor/entropy", mode="min")
    ent_max = get_metric_peak(exp_metrics, "actor/entropy", mode="max")
    ent_final = get_metric(exp_metrics, "actor/entropy")
    if ent_min is not None and ent_max is not None and ent_final is not None:
        rows.append([name, fmt(ent_max), fmt(ent_final), fmt(ent_min), fmt(ent_max - ent_min)])
    else:
        rows.append([name, "N/A", "N/A", "N/A", "N/A"])
md_table(headers, rows)

# ============ 5. Advantage 分布 ============
md("\n## 5. Advantage 分布（最终step）\n")
headers = ["实验", "adv/mean", "adv/max", "adv/min"]
rows = [[name] + [fmt(table[name].get(m)) for m in ["adv/mean", "adv/max", "adv/min"]] for name in exp_names]
md_table(headers, rows)

# ============ 6. Reward 分布 ============
md("\n## 6. 训练 Reward 分布（最终step）\n")
headers = ["实验", "reward/mean", "reward/max", "reward/min"]
rows = [[name] + [fmt(table[name].get(m)) for m in ["reward/mean", "reward/max", "reward/min"]] for name in exp_names]
md_table(headers, rows)

# ============ 7. 响应长度 ============
md("\n## 7. 响应长度（最终step）\n")
headers = ["实验", "final", "hist_min", "hist_max"]
rows = []
for name, exp_metrics in experiments.items():
    sl_min = get_metric_peak(exp_metrics, "global_seqlen/mean", mode="min")
    sl_max = get_metric_peak(exp_metrics, "global_seqlen/mean", mode="max")
    sl_final = get_metric(exp_metrics, "global_seqlen/mean")
    rows.append([name, fmt(sl_final, 0), fmt(sl_min, 0), fmt(sl_max, 0)])
md_table(headers, rows)

# ============ 8. 格式错误率 ============
md("\n## 8. 格式错误率（最终step，mean@16）\n")
headers = ["实验", "fmt_err/mean@16"]
rows = [[name, fmt(table[name].get("fmt_err/mean@16"))] for name in exp_names]
md_table(headers, rows)

# ============ 9. Rollout 分布漂移 ============
md("\n## 9. Rollout 分布漂移\n")
headers = ["实验", "rollout_ppl (final)", "rollout_ppl (max)", "probs_diff (final)", "probs_diff (max)"]
rows = []
for name, exp_metrics in experiments.items():
    ppl = get_metric(exp_metrics, "rollout_corr/rollout_ppl")
    pdiff = get_metric(exp_metrics, "training/rollout_probs_diff_mean")
    ppl_max = get_metric_peak(exp_metrics, "rollout_corr/rollout_ppl", mode="max")
    pdiff_max = get_metric_peak(exp_metrics, "training/rollout_probs_diff_mean", mode="max")
    rows.append([name, fmt(ppl), fmt(ppl_max), fmt(pdiff), fmt(pdiff_max)])
md_table(headers, rows)

# ============ 10. Returns 分布 ============
md("\n## 10. Returns 分布（最终step）\n")
headers = ["实验", "returns/mean", "returns/max", "returns/min"]
rows = [[name] + [fmt(table[name].get(m)) for m in ["returns/mean", "returns/max", "returns/min"]] for name in exp_names]
md_table(headers, rows)

# ============ 11. Score 分布 ============
md("\n## 11. Score 分布（训练，最终step）\n")
headers = ["实验", "score/mean", "score/max", "score/min"]
rows = [[name] + [fmt(table[name].get(m)) for m in ["score/mean", "score/max", "score/min"]] for name in exp_names]
md_table(headers, rows)

# ============ 12. TASD 特有指标 ============
md("\n## 12. TASD 特有指标（advantage 与 verifier 一致性）\n")
headers = ["实验", "adv-ver-agree", "adv-ver-neg-neg", "adv-neg-ver-pos", "adv-pos-ver-neg", "adv-pos-ver-pos"]
rows = [[name] + [fmt(table[name].get(m)) for m in [
    "tasd/adv_ver_agree_rate", "tasd/adv_neg_ver_neg_rate",
    "tasd/adv_neg_ver_pos_rate", "tasd/adv_pos_ver_neg_rate", "tasd/adv_pos_ver_pos_rate"
]] for name in exp_names]
md_table(headers, rows)

# ============ 13. TASD Advantage 统计 ============
md("\n## 13. TASD Advantage 统计\n")
headers = ["实验", "advantage_mean", "advantage_std", "advantage_pos_rate"]
rows = [[name] + [fmt(table[name].get(m)) for m in [
    "tasd/advantage_mean", "tasd/advantage_std", "tasd/advantage_pos_rate"
]] for name in exp_names]
md_table(headers, rows)

# ============ 14. TASD Success Rate & Token Reward ============
md("\n## 14. TASD Success Rate & Token Reward\n")
headers = ["实验", "success_rate", "token_reward_mean"]
rows = [[name] + [fmt(table[name].get(m)) for m in ["tasd/success_rate", "tasd/token_reward_mean"]] for name in exp_names]
md_table(headers, rows)

# ============ 15. 序列长度详细 ============
md("\n## 15. 序列长度详细分析（最终step）\n")
seq_metrics = ["seq_len/mean", "seq_len/max", "seq_len/min", "seq_len/balanced_max", "seq_len/balanced_min", "seq_len/minmax_diff"]
headers = ["实验"] + seq_metrics
rows = [[name] + [fmt(table[name].get(m), 0) for m in seq_metrics] for name in exp_names]
md_table(headers, rows)

# ============ 16. 综合排名 ============
md("\n## 16. 综合排名（按历史最佳 mean@16 降序）\n")
scores = []
for name, exp_metrics in experiments.items():
    m16 = get_metric_peak(exp_metrics, "val-core/sciknoweval/acc/mean@16", mode="max") or 0
    b16 = get_metric_peak(exp_metrics, "val-core/sciknoweval/acc/best@16/mean", mode="max") or 0
    scores.append((name, m16, b16))
scores.sort(key=lambda x: x[1], reverse=True)

headers = ["排名", "实验", "mean@16 best", "best@16 best"]
rows = []
for i, (name, m16, b16) in enumerate(scores):
    rows.append([i+1, name, fmt(m16), fmt(b16)])
md_table(headers, rows)

md("\n---\n*分析完成*")
_out.close()
print(f"\n已输出到 {OUTPUT_FILE}")
