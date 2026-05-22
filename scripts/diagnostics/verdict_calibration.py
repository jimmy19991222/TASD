#!/usr/bin/env python3
"""Verdict-calibration diagnostic for the prepend-right/prepend-wrong contrastive teacher.

Measures whether the base model can act as a calibrated judge: does the per-sequence
log-Bayes factor

    delta_seq = sum_t [ log p_T(s_t | ctx_right, s_<t) - log p_T(s_t | ctx_wrong, s_<t) ]

distinguish verifier-correct from verifier-incorrect rollouts?

If AUC(delta_seq, R) < 0.7 -> teacher cannot judge -> verdict_contrast credit
assignment is infeasible. If AUC > 0.8 -> proceed to per-token implementation.

Single-GPU HF script (no vllm); meant for offline calibration only.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Make the project importable so we can reuse the verifier.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from verl.utils.reward_score.feedback import compute_score  # noqa: E402
from verl.utils.verdict_markers import VERDICT_RIGHT_MARKER, VERDICT_WRONG_MARKER  # noqa: E402


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--val_parquet", required=True)
    ap.add_argument("--data_source", default="sciknoweval")
    ap.add_argument("--n_examples", type=int, default=50)
    ap.add_argument("--n_rollouts_per_example", type=int, default=4)
    ap.add_argument("--max_new_tokens", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--output_json", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--swanlab_project", default=None)
    ap.add_argument("--swanlab_experiment", default=None)
    return ap.parse_args()


def auc_binary(scores: np.ndarray, labels: np.ndarray) -> float:
    """ROC AUC for binary labels without sklearn (rank-based, handles ties)."""
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    n_pos = int((labels == 1).sum())
    n_neg = int((labels == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    i = 0
    while i < len(scores):
        j = i
        while j + 1 < len(scores) and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank
        i = j + 1
    sum_pos_ranks = ranks[labels == 1].sum()
    return float((sum_pos_ranks - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def build_messages(prompt_messages: list[dict], verdict_marker: str | None) -> list[dict]:
    """Prepend a system-level verdict marker before the existing prompt messages."""
    if verdict_marker is None:
        return list(prompt_messages)
    return [{"role": "system", "content": verdict_marker}, *prompt_messages]


def encode_prompt_and_completion(
    tokenizer,
    prompt_messages: list[dict],
    completion_text: str,
    verdict_marker: str | None,
    device: torch.device,
) -> tuple[torch.Tensor, int, int]:
    """Returns (input_ids, prompt_len, completion_len) for one (ctx, completion)."""
    msgs = build_messages(prompt_messages, verdict_marker)
    prompt_ids = tokenizer.apply_chat_template(
        msgs, add_generation_prompt=True, tokenize=True, return_tensors="pt"
    ).to(device)
    completion_ids = tokenizer(
        completion_text, add_special_tokens=False, return_tensors="pt"
    ).input_ids.to(device)
    full = torch.cat([prompt_ids, completion_ids], dim=-1)
    return full, prompt_ids.shape[-1], completion_ids.shape[-1]


@torch.no_grad()
def completion_logprob(
    model, input_ids: torch.Tensor, prompt_len: int
) -> tuple[float, int]:
    """Sum of log p(token | prefix) over completion positions."""
    logits = model(input_ids).logits  # (1, T, V)
    log_probs = torch.log_softmax(logits[:, :-1, :].float(), dim=-1)
    target = input_ids[:, 1:]
    target_lp = log_probs.gather(-1, target.unsqueeze(-1)).squeeze(-1)  # (1, T-1)
    completion_lp = target_lp[:, prompt_len - 1 :]  # positions predicting completion tokens
    return float(completion_lp.sum().item()), int(completion_lp.numel())


def generate_rollouts(
    model,
    tokenizer,
    prompt_messages: list[dict],
    n: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    device: torch.device,
) -> list[str]:
    prompt_ids = tokenizer.apply_chat_template(
        prompt_messages, add_generation_prompt=True, tokenize=True, return_tensors="pt"
    ).to(device)
    out = model.generate(
        prompt_ids,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        max_new_tokens=max_new_tokens,
        num_return_sequences=n,
        pad_token_id=tokenizer.eos_token_id,
    )
    decoded = []
    for i in range(out.shape[0]):
        gen = out[i, prompt_ids.shape[-1] :]
        decoded.append(tokenizer.decode(gen, skip_special_tokens=True))
    return decoded


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print(f"[verdict_cal] loading tokenizer + model from {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[
        args.dtype
    ]
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=dtype, trust_remote_code=True
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    print(f"[verdict_cal] loading {args.val_parquet}")
    df = pd.read_parquet(args.val_parquet)
    df = df.head(args.n_examples).reset_index(drop=True)
    print(f"[verdict_cal] {len(df)} examples × {args.n_rollouts_per_example} rollouts")

    records: list[dict] = []
    t0 = time.time()

    for idx, row in df.iterrows():
        prompt_messages = list(row["prompt"])
        if isinstance(prompt_messages[0], np.ndarray):
            prompt_messages = [dict(m) for m in prompt_messages]
        else:
            prompt_messages = [dict(m) for m in prompt_messages]

        rm = row["reward_model"]
        ground_truth = rm["ground_truth"] if isinstance(rm, dict) else dict(rm)["ground_truth"]
        data_source = row.get("data_source", args.data_source)

        rollouts = generate_rollouts(
            model,
            tokenizer,
            prompt_messages,
            n=args.n_rollouts_per_example,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            device=device,
        )

        for r_idx, rollout in enumerate(rollouts):
            try:
                scored = compute_score(
                    data_source=data_source,
                    solution_str=rollout,
                    ground_truth=ground_truth,
                    extra_info=None,
                )
                R = float(scored["score"])
            except Exception as e:  # noqa: BLE001
                print(f"[verdict_cal] WARN scoring failed idx={idx} r={r_idx}: {e}")
                continue

            ids_r, plen_r, clen_r = encode_prompt_and_completion(
                tokenizer, prompt_messages, rollout, VERDICT_RIGHT_MARKER, device
            )
            ids_w, plen_w, clen_w = encode_prompt_and_completion(
                tokenizer, prompt_messages, rollout, VERDICT_WRONG_MARKER, device
            )
            ids_n, plen_n, clen_n = encode_prompt_and_completion(
                tokenizer, prompt_messages, rollout, None, device
            )

            lp_right, _ = completion_logprob(model, ids_r, plen_r)
            lp_wrong, _ = completion_logprob(model, ids_w, plen_w)
            lp_none, _ = completion_logprob(model, ids_n, plen_n)

            # Normalize by completion length for length-robust ranking.
            n_tok = max(clen_r, 1)
            delta_seq = lp_right - lp_wrong
            delta_per_tok = delta_seq / n_tok

            records.append(
                {
                    "ex_idx": int(idx),
                    "roll_idx": int(r_idx),
                    "R": R,
                    "n_completion_tokens": int(n_tok),
                    "lp_right": lp_right,
                    "lp_wrong": lp_wrong,
                    "lp_none": lp_none,
                    "delta_seq": delta_seq,
                    "delta_per_tok": delta_per_tok,
                }
            )

        if (idx + 1) % 5 == 0 or idx == len(df) - 1:
            elapsed = time.time() - t0
            print(
                f"[verdict_cal] {idx + 1}/{len(df)}  "
                f"records={len(records)}  elapsed={elapsed:.0f}s"
            )

    if not records:
        raise RuntimeError("No records collected — calibration check failed before any scoring.")

    rec_df = pd.DataFrame(records)
    R_arr = rec_df["R"].to_numpy()
    delta_arr = rec_df["delta_seq"].to_numpy()
    delta_pt_arr = rec_df["delta_per_tok"].to_numpy()

    auc_seq = auc_binary(delta_arr, (R_arr > 0.5).astype(np.int64))
    auc_per_tok = auc_binary(delta_pt_arr, (R_arr > 0.5).astype(np.int64))

    delta_pos_mean = float(delta_arr[R_arr > 0.5].mean()) if (R_arr > 0.5).any() else float("nan")
    delta_neg_mean = float(delta_arr[R_arr <= 0.5].mean()) if (R_arr <= 0.5).any() else float("nan")
    sep = delta_pos_mean - delta_neg_mean
    pooled_std = float(delta_arr.std()) or 1e-8
    cohens_d = sep / pooled_std

    pos_rate = float((R_arr > 0.5).mean())

    summary = {
        "model_path": args.model_path,
        "val_parquet": args.val_parquet,
        "n_examples": int(args.n_examples),
        "n_rollouts_per_example": int(args.n_rollouts_per_example),
        "n_records": int(len(rec_df)),
        "pos_rate": pos_rate,
        "auc_delta_seq": auc_seq,
        "auc_delta_per_tok": auc_per_tok,
        "delta_pos_mean": delta_pos_mean,
        "delta_neg_mean": delta_neg_mean,
        "delta_separation": sep,
        "cohens_d": cohens_d,
        "verdict": (
            "PROCEED" if auc_seq >= 0.8 else ("MARGINAL" if auc_seq >= 0.7 else "STOP")
        ),
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.output_json)) or ".", exist_ok=True)
    out_payload = {"summary": summary, "records": records}
    with open(args.output_json, "w") as f:
        json.dump(out_payload, f, indent=2)
    print("\n=== Verdict Calibration Summary ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print(f"[verdict_cal] wrote {args.output_json}")

    if args.swanlab_project:
        try:
            import swanlab

            swanlab.init(
                project=args.swanlab_project,
                experiment_name=args.swanlab_experiment or "verdict_calibration",
                config=vars(args),
            )
            swanlab.log({f"calib/{k}": v for k, v in summary.items() if isinstance(v, (int, float))})
            swanlab.finish()
        except Exception as e:  # noqa: BLE001
            print(f"[verdict_cal] swanlab log failed: {e}")


if __name__ == "__main__":
    main()
