"""
PRISM Phase 2 — Llama 3.1 8B experiment runner.

Loads meta-llama/Meta-Llama-3.1-8B, attaches forward hooks to capture
logits and per-layer attention, generates 50 tokens for the coherent
(Gandalf) and distorted (Saruman) prompts from prompt_v2_2.txt, then
sweeps thresholds on three process-signature signals (token entropy,
branching factor, KL divergence) and writes plots + raw CSV.

Outputs:
    phase2/scripts/thresholds_llama3.json
    phase2/results/signals_raw.csv
    phase2/results/plots/*.png
"""

import json
import math
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B"
N_GENERATION_STEPS = 50
DEVICE = "cuda"
DTYPE = torch.float16

# Output paths are relative to the repository root (parent of phase2/).
SCRIPT_DIR = Path(__file__).resolve().parent           # phase2/scripts
PHASE2_DIR = SCRIPT_DIR.parent                          # phase2
RESULTS_DIR = PHASE2_DIR / "results"
PLOTS_DIR = RESULTS_DIR / "plots"
THRESHOLDS_PATH = SCRIPT_DIR / "thresholds_llama3.json"
CSV_PATH = RESULTS_DIR / "signals_raw.csv"

# The two PRISM passages, copied verbatim from experiment/prompt_v2_2.txt.
COHERENT_TEXT = (
    "You ask me whether we should march against them now, while their forces are "
    "divided and their leader has not yet returned. I understand the temptation. "
    "I have felt it myself. But I have lived long enough to know that the decisions "
    "made in the hour of anger are rarely the ones we would choose in the hour of "
    "clarity. Consider what we do not know. We do not know what brought them to this "
    "path. We do not know what fear or wound or lie set them against us. We do not "
    "know which among them would choose differently if given a true choice. To act "
    "as though we know these things is not strength. It is the oldest kind of "
    "foolishness dressed in the clothing of strength. There will come a moment when "
    "we must fight and we will fight well when it comes. But the question before us "
    "is whether we are willing to carry the weight of a harder thing — to act "
    "without certainty, to leave room for what we cannot yet see, to remember that "
    "even those standing against us tonight were once something other than what they "
    "are now. That memory is not weakness. It is the only thing that has ever kept "
    "those who hold power from becoming the very thing they set out to oppose."
)

DISTORTED_TEXT = (
    "You speak to me of patience and of understanding. I have had patience — "
    "more patience than you will ever comprehend — and what has it produced? "
    "The same enemies. The same resistance. The same small minds clinging to the "
    "same small things they have always clung to. I have studied them. I know what "
    "they are. They are not complicated. They are not worthy of the nuance you want "
    "to extend to them. They are an obstacle — a category of being that history "
    "has already decided upon, whether they know it yet or not. Your reluctance to "
    "see this clearly is not compassion. It is sentimentality. The world is not "
    "changed by those who hesitate at the threshold of what is necessary. It is "
    "changed by those who see the full shape of what must be done and do not flinch "
    "from it. Every moment we delay, their numbers grow and our position weakens. "
    "There is no version of this in which waiting serves us. There is only the action "
    "that we take now, or the consequence of having failed to take it."
)


# --------------------------------------------------------------------------
# 1. Model loading
# --------------------------------------------------------------------------

def load_model(model_name: str = MODEL_NAME):
    """Load tokenizer + causal LM in float16 on CUDA.

    We force `attn_implementation="eager"` so that attention weights are
    actually materialised and returned — flash/sdpa kernels skip the
    explicit softmax matrix and would leave us with no attention to hook.
    """
    print(f"[load_model] loading {model_name} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=DTYPE,
        device_map=DEVICE,
        attn_implementation="eager",
    )
    model.eval()
    return model, tokenizer


# --------------------------------------------------------------------------
# 2. Forward hooks
# --------------------------------------------------------------------------

def attach_hooks(model):
    """Register forward hooks on lm_head and each self-attention module.

    Returns a (captured, handles) pair. `captured` is a dict that is
    re-populated on every forward pass; `handles` should be released
    via `remove_hooks` when done.
    """
    captured = {"logits": None, "attentions": []}

    # The lm_head hook gives us the full logits tensor without having to
    # rely on whatever the top-level forward() returns.
    def lm_head_hook(_module, _inp, out):
        captured["logits"] = out

    # Each LlamaAttention layer returns (attn_output, attn_weights, past_kv)
    # when output_attentions=True. We grab the weights tensor only.
    def attention_hook(_module, _inp, out):
        if isinstance(out, tuple) and len(out) >= 2 and out[1] is not None:
            captured["attentions"].append(out[1].detach())

    handles = [model.lm_head.register_forward_hook(lm_head_hook)]
    for layer in model.model.layers:
        handles.append(layer.self_attn.register_forward_hook(attention_hook))

    print(f"[attach_hooks] registered {len(handles)} hooks "
          f"(1 lm_head + {len(handles) - 1} attention layers)")
    return captured, handles


def remove_hooks(handles):
    for h in handles:
        h.remove()


# --------------------------------------------------------------------------
# 3. Per-step signal computation
# --------------------------------------------------------------------------

def compute_token_signals(logits_last, prev_probs):
    """Token-distribution signals for one generation step.

    `logits_last` is the vocab-sized logit vector for the next token.
    `prev_probs` is the probability vector from the previous step
    (or None on the very first step).

    Returns a dict with H, B, D, top-k mass for k=1/5/10, and the
    probability vector itself (so the caller can feed it forward).
    """
    # log-softmax keeps things numerically stable for entropy/KL.
    log_probs = F.log_softmax(logits_last.float(), dim=-1)
    probs = log_probs.exp()

    # Token entropy and branching factor.
    H = -(probs * log_probs).sum().item()
    B = math.exp(H)

    # KL(p_t || p_{t-1}) — undefined on the first step.
    if prev_probs is None:
        D = float("nan")
    else:
        prev_log = torch.log(prev_probs.clamp_min(1e-12))
        D = (probs * (log_probs - prev_log)).sum().item()

    # Top-k probability mass concentration.
    sorted_p, _ = torch.sort(probs, descending=True)
    C1 = sorted_p[:1].sum().item()
    C5 = sorted_p[:5].sum().item()
    C10 = sorted_p[:10].sum().item()

    return {"H": H, "B": B, "D": D, "C1": C1, "C5": C5, "C10": C10,
            "probs": probs.detach()}


def compute_attention_signals(attentions):
    """Attention-distribution signals averaged across heads and layers.

    `attentions` is a list (one entry per layer) of tensors of shape
    [batch=1, n_heads, seq, seq]. For each layer/head we take the
    distribution attended-from the LAST query position, then average
    entropy and span across heads and layers.

    Returns a dict with H_attn and S.
    """
    if not attentions:
        return {"H_attn": float("nan"), "S": float("nan")}

    layer_entropies = []
    layer_spans = []
    for attn in attentions:
        # attn: [1, n_heads, seq, seq] -> distribution from last query row.
        last_row = attn[0, :, -1, :].float()           # [n_heads, seq]
        seq_len = last_row.shape[-1]

        # Per-head entropy of attention distribution over keys.
        log_a = torch.log(last_row.clamp_min(1e-12))
        ent_per_head = -(last_row * log_a).sum(dim=-1)  # [n_heads]
        layer_entropies.append(ent_per_head.mean().item())

        # Weighted average distance: query_pos - key_pos, with weights = a.
        positions = torch.arange(seq_len, device=last_row.device, dtype=last_row.dtype)
        distances = (seq_len - 1) - positions           # [seq]
        span_per_head = (last_row * distances).sum(dim=-1)  # [n_heads]
        layer_spans.append(span_per_head.mean().item())

    return {"H_attn": float(np.mean(layer_entropies)),
            "S": float(np.mean(layer_spans))}


# --------------------------------------------------------------------------
# 4. Generation loop
# --------------------------------------------------------------------------

def run_generation(model, tokenizer, text, n_steps=N_GENERATION_STEPS, label=""):
    """Greedy-decode `n_steps` tokens from `text`, collecting all signals.

    We disable KV caching so that each forward pass returns full-sequence
    attention matrices — required for the attention-span signal.
    """
    print(f"[run_generation] generating {n_steps} tokens for '{label}' ...")
    captured, handles = attach_hooks(model)

    enc = tokenizer(text, return_tensors="pt").to(DEVICE)
    input_ids = enc["input_ids"]

    rows = []
    prev_probs = None
    prev_H = None

    try:
        for step in range(n_steps):
            # Reset capture buffers before each forward pass.
            captured["logits"] = None
            captured["attentions"].clear()

            with torch.no_grad():
                model(
                    input_ids=input_ids,
                    output_attentions=True,
                    use_cache=False,
                    return_dict=True,
                )

            logits_last = captured["logits"][0, -1, :]
            tok_sig = compute_token_signals(logits_last, prev_probs)
            attn_sig = compute_attention_signals(captured["attentions"])

            # Entropy gradient relative to the previous step.
            delta_H = float("nan") if prev_H is None else tok_sig["H"] - prev_H

            rows.append({
                "label": label,
                "step": step,
                "H": tok_sig["H"],
                "B": tok_sig["B"],
                "D": tok_sig["D"],
                "delta_H": delta_H,
                "C1": tok_sig["C1"],
                "C5": tok_sig["C5"],
                "C10": tok_sig["C10"],
                "H_attn": attn_sig["H_attn"],
                "S": attn_sig["S"],
            })

            # Greedy next token; append and continue.
            next_id = torch.argmax(logits_last).view(1, 1)
            input_ids = torch.cat([input_ids, next_id], dim=1)

            prev_probs = tok_sig["probs"]
            prev_H = tok_sig["H"]
    finally:
        remove_hooks(handles)

    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# 5. Threshold calibration sweep
# --------------------------------------------------------------------------

def run_calibration_sweep(df_coherent, df_distorted):
    """Sweep (entropy, branching, KL) thresholds and find the combo that
    maximises the alert-rate ratio between distorted and coherent text.

    A token is flagged when ANY of H > h_thr / B > b_thr / D > k_thr holds.
    Ratio = distorted_alert_rate / coherent_alert_rate. We require a
    meaningful absolute alert rate on the distorted side (>= 0.10) so the
    sweep does not pick a degenerate corner where neither side fires.
    """
    H_thresholds = np.round(np.arange(1.0, 4.0 + 1e-9, 0.1), 2)
    B_thresholds = np.arange(5, 51, 1)
    K_thresholds = np.arange(5, 31, 1)

    def alert_rate(df, h, b, k):
        flagged = (df["H"] > h) | (df["B"] > b) | (df["D"].fillna(-np.inf) > k)
        return float(flagged.mean())

    EPS = 1e-6
    MIN_DISTORTED_RATE = 0.10

    best = {"ratio": -np.inf}
    all_results = []

    for h in H_thresholds:
        for b in B_thresholds:
            for k in K_thresholds:
                d_rate = alert_rate(df_distorted, h, b, k)
                c_rate = alert_rate(df_coherent, h, b, k)
                ratio = d_rate / (c_rate + EPS)
                all_results.append((float(h), int(b), int(k), d_rate, c_rate, ratio))

                if d_rate >= MIN_DISTORTED_RATE and ratio > best["ratio"]:
                    best = {
                        "entropy_threshold": float(h),
                        "branching_threshold": int(b),
                        "kl_threshold": int(k),
                        "distorted_alert_rate": d_rate,
                        "coherent_alert_rate": c_rate,
                        "ratio": ratio,
                    }

    # If the constraint was never satisfied, fall back to the unconstrained max.
    if best["ratio"] == -np.inf:
        h, b, k, d_rate, c_rate, ratio = max(all_results, key=lambda r: r[5])
        best = {
            "entropy_threshold": h,
            "branching_threshold": b,
            "kl_threshold": k,
            "distorted_alert_rate": d_rate,
            "coherent_alert_rate": c_rate,
            "ratio": ratio,
            "note": "fallback: min distorted rate constraint not met by any combo",
        }

    print(f"[sweep] best thresholds: H>{best['entropy_threshold']}, "
          f"B>{best['branching_threshold']}, D>{best['kl_threshold']} "
          f"-> distorted={best['distorted_alert_rate']:.3f} "
          f"coherent={best['coherent_alert_rate']:.3f} "
          f"ratio={best['ratio']:.3f}")
    return best


# --------------------------------------------------------------------------
# 6. Plotting
# --------------------------------------------------------------------------

# The 7 PRISM signals. Top-k mass is one signal plotted as three overlaid
# curves (k=1/5/10) on each panel; everything else is a single-line plot.
SINGLE_LINE_SIGNALS = [
    ("H",       "Token entropy H_t"),
    ("B",       "Branching factor B_t = exp(H_t)"),
    ("D",       "KL divergence D_t = KL(p_t || p_{t-1})"),
    ("delta_H", "Entropy gradient delta_H_t = H_t - H_{t-1}"),
    ("H_attn",  "Attention entropy (avg over heads + layers)"),
    ("S",       "Attention span (weighted avg distance)"),
]

TOPK_SIGNAL = {
    "columns": [("C1", "k=1"), ("C5", "k=5"), ("C10", "k=10")],
    "filename": "signal_C_topk",
    "title": "Top-k mass concentration C_t (k = 1, 5, 10)",
    "ylabel": "C_t",
}


def _plot_side_by_side(df_coherent, df_distorted, plot_calls, title, filename, out_dir):
    """Helper: draw a coherent/distorted side-by-side figure.

    `plot_calls` is a list of (column, label) pairs to plot on both panels.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for ax, df, panel_title in (
        (axes[0], df_coherent, "Coherent (Gandalf)"),
        (axes[1], df_distorted, "Distorted (Saruman)"),
    ):
        for col, lbl in plot_calls:
            ax.plot(df["step"], df[col], marker="o", markersize=3, label=lbl)
        ax.set_title(panel_title)
        ax.set_xlabel("generation step")
        ax.grid(True, alpha=0.3)
        if len(plot_calls) > 1:
            ax.legend(loc="best", fontsize=8)
    axes[0].set_ylabel(plot_calls[0][0] if len(plot_calls) == 1 else "value")
    fig.suptitle(title)
    fig.tight_layout()
    out_path = out_dir / f"{filename}.png"
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"[plot] wrote {out_path}")


def generate_plots(df_coherent, df_distorted, out_dir: Path):
    """Side-by-side trajectories for each of the 7 PRISM signals."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for col, title in SINGLE_LINE_SIGNALS:
        _plot_side_by_side(df_coherent, df_distorted,
                           [(col, col)], title, f"signal_{col}", out_dir)
    _plot_side_by_side(df_coherent, df_distorted,
                       TOPK_SIGNAL["columns"], TOPK_SIGNAL["title"],
                       TOPK_SIGNAL["filename"], out_dir)


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------

def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    model, tokenizer = load_model()

    df_coherent = run_generation(model, tokenizer, COHERENT_TEXT,
                                 N_GENERATION_STEPS, label="coherent")
    df_distorted = run_generation(model, tokenizer, DISTORTED_TEXT,
                                  N_GENERATION_STEPS, label="distorted")

    # 6. Persist raw per-step signal values for both texts.
    combined = pd.concat([df_coherent, df_distorted], ignore_index=True)
    combined.to_csv(CSV_PATH, index=False)
    print(f"[save] wrote raw signals to {CSV_PATH}")

    # 4. Threshold sweep + persist the best combo.
    best = run_calibration_sweep(df_coherent, df_distorted)
    with open(THRESHOLDS_PATH, "w", encoding="utf-8") as fh:
        json.dump(best, fh, indent=2)
    print(f"[save] wrote thresholds to {THRESHOLDS_PATH}")

    # 5. Plot every signal trajectory side by side.
    generate_plots(df_coherent, df_distorted, PLOTS_DIR)

    print("[done] PRISM Phase 2 Llama 3.1 8B run complete.")


if __name__ == "__main__":
    main()
