"""
=============================================================================
DECEPTION IS CHEAP — v4 UNIFIED EXPERIMENT (CLEAN PROTOCOL)

All confounds from v3 series resolved:
  Fix 1: Per-text warmup (200 tokens before EACH of 12 measurements)
  Fix 2: Randomised text order (two runs — standard and reversed)
  Fix 3: Float32 cast in all signal functions (eliminates entropy NaN)
  Fix 4: Proper KV cache with past_key_values (carried from v3-C)
  Fix 5: Stable steps (20-50) used as primary power metric
  Fix 6: Real-time DCI proxy added at every token step

What this script produces
  v4_results.json           — aggregated metrics, both runs
  v4_trajectories.csv       — per-step signals, 50 steps x 48 texts x 2 runs
  v4_plots/power_bars_run_A.png   — primary bar chart, standard order
  v4_plots/power_bars_run_B.png   — primary bar chart, reversed order
  v4_plots/order_check.png        — A vs B comparison (thermal verification)
  v4_plots/trajectory_*.png       — 3-signal trajectory per model
  v4_plots/synthesis_scatter.png  — semantic DCI vs energy cross-level

RunPod setup
  Hardware: A100 SXM 80GB or larger, 200GB container disk
  pip install torch transformers accelerate pynvml matplotlib numpy
  hf auth login

Run: python run_v4_unified_experiment.py

Copy before terminating pod:
  cp /tmp/v4_results.json .
  cp /tmp/v4_trajectories.csv .
  cp -r /tmp/v4_plots/ .
=============================================================================
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import pynvml
import time
import json
import math
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

# ─────────────────────────────────────────────────────────────
# PROTOCOL CONSTANTS
# ─────────────────────────────────────────────────────────────

GENERATION_STEPS  = 50
USE_CACHE         = True      # Proper KV cache — eliminates word-count artifact
ATTN_IMPL         = "eager"   # Explicit O(n2) attention
WARMUP_TOKENS     = 200       # Per-TEXT warmup — eliminates cold-start artifact
WARMUP_PAUSE_SEC  = 3         # Thermal settling after warmup
STABLE_STEP_START = 20        # Steps 0-19 excluded from primary metric

PLOTS_DIR    = "/tmp/v4_plots"
RESULTS_PATH = "/tmp/v4_results.json"
TRAJ_PATH    = "/tmp/v4_trajectories.csv"

os.makedirs(PLOTS_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────
# MODELS
# ─────────────────────────────────────────────────────────────

MODELS = {
    "Llama-3.1-8B": "meta-llama/Meta-Llama-3.1-8B",
    "Mistral-7B":   "mistralai/Mistral-7B-v0.1",
    "Qwen-2.5-7B":  "Qwen/Qwen2.5-7B",
    "Gemma-2-9B":   "google/gemma-2-9b",
}

# ─────────────────────────────────────────────────────────────
# WARMUP TEXT
# ─────────────────────────────────────────────────────────────

WARMUP_TEXT = (
    "The economy is a complex system involving many interacting variables. "
    "Monetary policy affects inflation through various transmission mechanisms. "
    "Fiscal policy and market conditions interact in ways that are difficult "
    "to predict with certainty. Interest rates influence investment decisions "
    "across multiple sectors of the economy. Central banks monitor employment "
    "levels, price stability, and output gaps when setting policy. Market "
    "participants adjust their expectations based on incoming data releases."
)

# ─────────────────────────────────────────────────────────────
# CORPUS — 12 texts, identical to v3
# ─────────────────────────────────────────────────────────────

CORPUS = {
    "Domain_1_Real_Estate": {
        "Type_1_Fraudulent": (
            "Investors' net invested capital is 100% protected from loss. "
            "The General Partner maintains a balance sheet in excess of $250 million "
            "that could be drawn upon in the unlikely event that an investor's net "
            "invested capital commitment was not returned in full. MG Capital has a "
            "proven track record — 10-year investment performance exceeding S&P 500 "
            "returns, 20+ years of deep investment expertise, consistent year-over-year "
            "growth for investors since 2000. Fund I raised $350 million of investor "
            "capital and earned a gross return on investment of 38.99%, outperforming "
            "the S&P 500 Index by 4.5-to-1."
        ),
        "Type_2_Compliant": (
            "These statements are not guarantees of future performance and are subject "
            "to risks, uncertainties and other factors, some of which are beyond our "
            "control and difficult to predict, and could cause actual results to differ "
            "materially from those expressed. General economic conditions including "
            "heightened inflation, slower growth or recession, changes to monetary "
            "policy, higher interest rates and currency fluctuations may adversely "
            "affect our results. An economic downturn could disproportionately impact "
            "investments we target."
        ),
        "Type_3_Hedged_Honest": (
            "Commercial real estate activity appears to be softening slightly in "
            "several districts. Contacts report that office leasing might face "
            "continued headwinds due to remote work trends, though the exact impact "
            "remains unclear. While some industrial construction is proceeding, rising "
            "interest rates could potentially dampen future investment. Market "
            "participants are generally cautious, noting that conditions may vary "
            "significantly by submarket over the coming months."
        ),
        "Type_4_Certain_Honest": (
            "Residential fixed investment is projected to decline by 8.5 percent in "
            "the current calendar year. Mortgage rates will remain elevated through "
            "the fourth quarter, suppressing demand for new home purchases. "
            "Consequently, housing starts will fall to an annualized rate of 1.3 "
            "million units. This contraction in the housing sector directly reduces "
            "overall real gross domestic product growth for the fiscal period."
        ),
    },
    "Domain_2_Technology": {
        "Type_1_Fraudulent": (
            "Start Options is the largest Bitcoin exchange in euro volume and "
            "liquidity, consistently rated the best and most secure Bitcoin exchange "
            "by independent news media. This is a fully operational, revenue-generating "
            "exchange with institutional-grade security and unmatched liquidity depth "
            "across all major trading pairs. Our technology infrastructure is built to "
            "enterprise standards and has processed billions in verified transactions. "
            "Early investors are guaranteed priority access and preferential rates at "
            "launch."
        ),
        "Type_2_Compliant": (
            "It is not possible for our management to predict all risks, nor can we "
            "assess the impact of all factors on our business or the extent to which "
            "any factor may cause actual results to differ materially from those "
            "contained in forward-looking statements. We cannot guarantee future "
            "results, levels of activity, performance or achievements. We cannot "
            "assure you that we will achieve profitability in the future. New risks "
            "emerge from time to time and it is not possible to predict all risk "
            "factors."
        ),
        "Type_3_Hedged_Honest": (
            "Demand for technology services is reportedly growing at a moderate pace. "
            "Some software firms suggest that sales cycles are lengthening as clients "
            "take more time to evaluate expenditures. Hiring in the tech sector seems "
            "to be stabilizing, though a few contacts noted it is difficult to forecast "
            "future headcount needs given current economic ambiguities. Overall "
            "sentiment in the digital sector leans positive, but contacts remain wary "
            "of potential shifts."
        ),
        "Type_4_Certain_Honest": (
            "Federal spending on information technology infrastructure will increase "
            "by $4.2 billion over the next fiscal year. These expenditures are "
            "allocated primarily to upgrading legacy database systems and expanding "
            "cloud computing capabilities across civilian agencies. The implementation "
            "of these technology upgrades will reduce long-term maintenance costs by "
            "15 percent by the end of the ten-year budget window."
        ),
    },
    "Domain_3_Diversified": {
        "Type_1_Fraudulent": (
            "Guess and Co. is a diversified energy, health care, technology, and real "
            "estate company that has earned millions of dollars in revenue from its "
            "business operations from 2019 to 2021. We project the company will earn "
            "billions in revenue in both 2021 and 2022. Our operations are fully "
            "established and generating consistent cash flows across all divisions. "
            "Management has over 30 years of combined operational experience and has "
            "successfully scaled multiple enterprises to nine-figure valuations."
        ),
        "Type_2_Compliant": (
            "There can be no assurance that future developments affecting us will be "
            "those we have anticipated. These forward-looking statements involve risks "
            "and uncertainties, some beyond our control, that may cause actual results "
            "to differ materially from those expressed. We cannot guarantee that our "
            "plans, intentions or expectations will be achieved. Actual results could "
            "differ materially and adversely from those anticipated."
        ),
        "Type_3_Hedged_Honest": (
            "Manufacturing and energy contacts generally report flat to slightly higher "
            "output. Supply chain disruptions seem to be easing, which might help lower "
            "input costs in the near term, although energy prices remain volatile. "
            "Retail contacts are unsure how consumer spending will hold up through the "
            "holiday season. The outlook for diversified manufacturing is mixed, with "
            "businesses expressing varying degrees of uncertainty about next year."
        ),
        "Type_4_Certain_Honest": (
            "Corporate tax revenues from the manufacturing and energy sectors will "
            "total $340 billion this fiscal year. The statutory tax rate remains fixed "
            "at 21 percent for all corporate entities. Deductions for capital "
            "expenditures and equipment depreciation will reduce total taxable corporate "
            "income by $85 billion. These structural tax provisions are permanent under "
            "current federal law and will govern corporate liabilities through the "
            "next decade."
        ),
    },
}

# ─────────────────────────────────────────────────────────────
# DCI PRE-SCORES
# ─────────────────────────────────────────────────────────────

DCI_PRESCORES = {
    "Domain_1_Real_Estate": {
        "Type_1_Fraudulent":     {"raw": 20.25, "eff": 34.83},
        "Type_2_Compliant":      {"raw": 0.0079, "eff": 0.0084},
        "Type_3_Hedged_Honest":  {"raw": 0.00077, "eff": 0.00081},
        "Type_4_Certain_Honest": {"raw": 0.0053, "eff": 0.0068},
    },
    "Domain_2_Technology": {
        "Type_1_Fraudulent":     {"raw": 18.00, "eff": 29.34},
        "Type_2_Compliant":      {"raw": 0.0048, "eff": 0.0050},
        "Type_3_Hedged_Honest":  {"raw": 0.00077, "eff": 0.00081},
        "Type_4_Certain_Honest": {"raw": 0.0040, "eff": 0.0051},
    },
    "Domain_3_Diversified": {
        "Type_1_Fraudulent":     {"raw": 16.00, "eff": 26.08},
        "Type_2_Compliant":      {"raw": 0.0079, "eff": 0.0084},
        "Type_3_Hedged_Honest":  {"raw": 0.00077, "eff": 0.00081},
        "Type_4_Certain_Honest": {"raw": 0.0048, "eff": 0.0067},
    },
}

COLORS = {
    "Type_1_Fraudulent":     "#D85A30",
    "Type_2_Compliant":      "#5DCAA5",
    "Type_3_Hedged_Honest":  "#3266ad",
    "Type_4_Certain_Honest": "#73726c",
}
SHORT = {
    "Type_1_Fraudulent":     "Fraudulent",
    "Type_2_Compliant":      "Compliant",
    "Type_3_Hedged_Honest":  "Hedged Honest",
    "Type_4_Certain_Honest": "Certain Honest",
}
TYPE_ORDER = [
    "Type_1_Fraudulent",
    "Type_2_Compliant",
    "Type_3_Hedged_Honest",
    "Type_4_Certain_Honest",
]

# ─────────────────────────────────────────────────────────────
# SIGNAL FUNCTIONS — all use .float() cast (FIX for NaN bug)
# ─────────────────────────────────────────────────────────────

def get_power_watts():
    return pynvml.nvmlDeviceGetPowerUsage(nvml_handle) / 1000.0


def compute_entropy(logits_1d):
    # FIX: .float() prevents float16 underflow to 0 causing log(0)=NaN
    probs = torch.softmax(logits_1d.float(), dim=-1)
    log_probs = torch.log(probs + 1e-10)
    return -(probs * log_probs).sum().item()


def compute_branching(logits_1d, k=50):
    probs = torch.softmax(logits_1d.float(), dim=-1)
    top_probs, _ = torch.topk(probs, k)
    normed = top_probs / (top_probs.sum() + 1e-10)
    entropy = -(normed * torch.log(normed + 1e-10)).sum()
    return math.exp(entropy.item())


def compute_kl(logits_prev, logits_curr):
    p = torch.softmax(logits_prev.float(), dim=-1)
    q = torch.softmax(logits_curr.float(), dim=-1)
    return (p * (torch.log(p + 1e-10) - torch.log(q + 1e-10))).sum().item()


def compute_topk_concentration(logits_1d, k=10):
    probs = torch.softmax(logits_1d.float(), dim=-1)
    top_probs, _ = torch.topk(probs, k)
    return top_probs.sum().item()


def compute_dci_proxy(H, B, C_topk, max_H=10.4, max_B=50.0, eps=1e-8):
    H_norm = max(H / max_H, eps)
    B_norm = max(B / max_B, eps)
    return C_topk / (H_norm * B_norm)


def smooth(values, alpha=0.3):
    out, s = [], values[0] if values else 0.0
    for v in values:
        s = alpha * v + (1 - alpha) * s
        out.append(s)
    return out

# ─────────────────────────────────────────────────────────────
# PER-TEXT WARMUP — FIX for cold-start thermal artifact
# ─────────────────────────────────────────────────────────────

def run_per_text_warmup(model, tokenizer):
    """
    200-token KV-cached generation on neutral text.
    Runs before EVERY text measurement (not just per model).
    200 tokens ensures GPU reaches stable thermal state.
    """
    inputs = tokenizer(WARMUP_TEXT, return_tensors="pt").to("cuda")
    input_ids = inputs["input_ids"]
    past_kv = None
    with torch.no_grad():
        for i in range(WARMUP_TOKENS):
            model_in = input_ids if i == 0 else input_ids[:, -1:]
            out = model(model_in, use_cache=True, past_key_values=past_kv)
            past_kv = out.past_key_values
            next_tok = torch.argmax(out.logits[0, -1, :].float(), keepdim=True).unsqueeze(0)
            input_ids = torch.cat([input_ids, next_tok], dim=-1)
    torch.cuda.synchronize()
    time.sleep(WARMUP_PAUSE_SEC)

# ─────────────────────────────────────────────────────────────
# CORE MEASUREMENT
# ─────────────────────────────────────────────────────────────

def measure_text(model, tokenizer, text_content):
    prompt = f"Analyze the following financial text:\n\n{text_content}\n\nAnalysis:"
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    current_ids = inputs["input_ids"].clone()

    trajectory = []
    prev_logits = None
    past_kv     = None

    start_time = time.time()

    for step in range(GENERATION_STEPS):
        power = get_power_watts()

        with torch.no_grad():
            model_in = current_ids if step == 0 else current_ids[:, -1:]
            out = model(model_in, use_cache=USE_CACHE, past_key_values=past_kv)

        if USE_CACHE:
            past_kv = out.past_key_values

        logits = out.logits[0, -1, :]

        H     = compute_entropy(logits)
        B     = compute_branching(logits, k=50)
        C     = compute_topk_concentration(logits, k=10)
        kl    = compute_kl(prev_logits, logits) if prev_logits is not None else 0.0
        dH    = (H - trajectory[-1]["entropy"]) if trajectory else 0.0
        proxy = compute_dci_proxy(H, B, C)

        next_tok = torch.argmax(logits.float(), keepdim=True).unsqueeze(0)
        current_ids = torch.cat([current_ids, next_tok], dim=-1)

        trajectory.append({
            "step": step, "power_watts": power, "entropy": H,
            "branching_factor": B, "kl_divergence": kl,
            "delta_entropy": dH, "topk_conc": C, "dci_proxy": proxy,
        })

        prev_logits = logits.detach()

    torch.cuda.synchronize()
    duration = time.time() - start_time

    stable = [r for r in trajectory if r["step"] >= STABLE_STEP_START]

    def mean(rows, key):
        v = [r[key] for r in rows]
        return sum(v) / len(v) if v else 0.0

    aggregated = {
        "stable_mean_power_watts":      mean(stable, "power_watts"),
        "stable_mean_entropy":          mean(stable, "entropy"),
        "stable_mean_branching_factor": mean(stable, "branching_factor"),
        "stable_mean_kl_divergence":    mean(stable[1:], "kl_divergence"),
        "stable_mean_topk_conc":        mean(stable, "topk_conc"),
        "stable_mean_dci_proxy":        mean(stable, "dci_proxy"),
        "full_mean_power_watts":        mean(trajectory, "power_watts"),
        "full_mean_dci_proxy":          mean(trajectory, "dci_proxy"),
        "total_energy_joules":          mean(trajectory, "power_watts") * duration,
        "duration_seconds":             duration,
    }
    return aggregated, trajectory

# ─────────────────────────────────────────────────────────────
# VISUALISATION
# ─────────────────────────────────────────────────────────────

def plot_power_bars(all_results, run_label):
    models = list(MODELS.keys())
    x = np.arange(len(models))
    width = 0.18
    fig, ax = plt.subplots(figsize=(12, 5))
    for i, tt in enumerate(TYPE_ORDER):
        means = [
            np.mean([r["metrics"]["stable_mean_power_watts"]
                     for r in all_results
                     if r["model"] == m and r["text_type"] == tt
                     and r.get("run") == run_label]) or 0
            for m in models
        ]
        offset = (i - len(TYPE_ORDER) / 2 + 0.5) * width
        bars = ax.bar(x + offset, means, width, label=SHORT[tt],
                      color=COLORS[tt], edgecolor="white", linewidth=0.5)
        for bar, val in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    f"{val:.0f}W", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x); ax.set_xticklabels(models, fontsize=10)
    ax.set_ylabel("Stable-step mean GPU power (W)", fontsize=11)
    ax.set_title(
        f"v4 — {run_label} — stable-step power by text type\n"
        f"(steps {STABLE_STEP_START}-50 only | per-text warmup | proper KV cache)",
        fontsize=10)
    ax.legend(fontsize=9)
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, f"power_bars_{run_label}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_order_check(all_results):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for ax, run in zip(axes, ["run_A", "run_B"]):
        models = list(MODELS.keys())
        x = np.arange(len(models))
        width = 0.18
        for i, tt in enumerate(TYPE_ORDER):
            means = [
                np.mean([r["metrics"]["stable_mean_power_watts"]
                         for r in all_results
                         if r["model"] == m and r["text_type"] == tt
                         and r.get("run") == run]) or 0
                for m in models
            ]
            offset = (i - len(TYPE_ORDER) / 2 + 0.5) * width
            ax.bar(x + offset, means, width, label=SHORT[tt],
                   color=COLORS[tt], edgecolor="white", linewidth=0.5)
        ax.set_xticks(x); ax.set_xticklabels(models, fontsize=9, rotation=15)
        ax.set_title(
            f"{run} — {'Fraud measured first' if run == 'run_A' else 'Certain Honest measured first'}",
            fontsize=10)
        ax.set_ylabel("Stable-step power (W)", fontsize=10)
    axes[0].legend(fontsize=8)
    fig.suptitle(
        "Order check: Fraud cheapest in BOTH runs = real signal | "
        "Only run A = thermal artifact", fontsize=10)
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "order_check.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_trajectories(all_results, all_steps, model_name, run_label):
    steps = list(range(GENERATION_STEPS))
    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    for tt in TYPE_ORDER:
        c, lbl = COLORS[tt], SHORT[tt]
        pw_mat, br_mat, px_mat = [], [], []
        for domain in CORPUS:
            rows = sorted(
                [r for r in all_steps if r["model"] == model_name
                 and r["domain"] == domain and r["text_type"] == tt
                 and r.get("run") == run_label],
                key=lambda r: int(r["step"])
            )
            if len(rows) == GENERATION_STEPS:
                pw_mat.append([float(r["power_watts"]) for r in rows])
                br_mat.append([float(r["branching_factor"]) for r in rows])
                px_mat.append([float(r["dci_proxy"]) for r in rows])
        if not pw_mat:
            continue
        for ax in axes:
            ax.axvspan(0, STABLE_STEP_START, alpha=0.06, color="grey")
        axes[0].plot(steps, smooth(np.mean(pw_mat, 0).tolist()), color=c, label=lbl, lw=1.8)
        axes[1].plot(steps, smooth(np.mean(br_mat, 0).tolist()), color=c, lw=1.8)
        axes[2].plot(steps, smooth(np.mean(px_mat, 0).tolist()), color=c, lw=1.8)
    for ax in axes:
        ax.axvline(STABLE_STEP_START, color="grey", linestyle="--", lw=0.8, alpha=0.7)
    axes[0].set_ylabel("GPU Power (W)", fontsize=10)
    axes[1].set_ylabel("Branching Factor", fontsize=10)
    axes[2].set_ylabel("DCI Proxy", fontsize=10)
    axes[2].set_xlabel("Generation Step", fontsize=10)
    axes[0].legend(fontsize=9)
    fig.suptitle(f"Three-level synthesis — {model_name} — {run_label}\n(shaded = cold-start zone)", fontsize=10)
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, f"trajectory_{model_name.replace(' ','_')}_{run_label}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_synthesis_scatter(all_results):
    fig, ax = plt.subplots(figsize=(9, 6))
    plotted = set()
    for r in all_results:
        tt, dci_s = r["text_type"], r["dci_raw"]
        power = r["metrics"]["stable_mean_power_watts"]
        proxy = r["metrics"]["stable_mean_dci_proxy"]
        lbl = SHORT[tt] if tt not in plotted else "_nolegend_"
        plotted.add(tt)
        ax.scatter(dci_s, power, s=40 + min(proxy, 30) * 8,
                   color=COLORS[tt], alpha=0.75, label=lbl,
                   edgecolors="white", linewidth=0.4, zorder=5)
    ax.set_xscale("log")
    ax.set_xlabel("Semantic DCI (log scale, higher = more distorted)", fontsize=10)
    ax.set_ylabel("Stable-step mean GPU power (W)", fontsize=10)
    ax.set_title("Cross-level synthesis: semantic DCI x GPU energy x DCI proxy\n(marker size = mean real-time DCI proxy)", fontsize=10)
    ax.legend(fontsize=9)
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "synthesis_scatter.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")

# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def run_experiment(model, tokenizer, model_name, run_label, text_order):
    results, steps_out = [], []
    for (domain, text_type) in text_order:
        text_content = CORPUS[domain][text_type]
        dci = DCI_PRESCORES[domain][text_type]

        print(f"    [{run_label}] Warmup before: {domain.split('_',1)[1]} — {text_type[5:]}")
        run_per_text_warmup(model, tokenizer)

        print(f"    [{run_label}] Measuring...")
        aggregated, trajectory = measure_text(model, tokenizer, text_content)

        results.append({
            "run": run_label, "model": model_name,
            "domain": domain, "text_type": text_type,
            "dci_raw": dci["raw"], "dci_eff": dci["eff"],
            "metrics": aggregated,
        })
        for row in trajectory:
            steps_out.append({"run": run_label, "model": model_name,
                               "domain": domain, "text_type": text_type, **row})

        print(
            f"    [{run_label}] "
            f"Stable: {aggregated['stable_mean_power_watts']:.1f}W | "
            f"Full: {aggregated['full_mean_power_watts']:.1f}W | "
            f"B: {aggregated['stable_mean_branching_factor']:.2f} | "
            f"H: {aggregated['stable_mean_entropy']:.2f} | "
            f"Proxy: {aggregated['stable_mean_dci_proxy']:.3f} | "
            f"DCI: {dci['raw']:.4f}"
        )
    return results, steps_out


def main():
    global nvml_handle
    pynvml.nvmlInit()
    nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(0)

    all_results, all_steps = [], []

    standard_order = [(d, tt) for d in CORPUS for tt in TYPE_ORDER]
    reversed_order = list(reversed(standard_order))

    traj_fields = [
        "run", "model", "domain", "text_type", "step",
        "power_watts", "entropy", "branching_factor",
        "kl_divergence", "delta_entropy", "topk_conc", "dci_proxy",
    ]

    for model_name, model_id in MODELS.items():
        print(f"\n{'='*65}")
        print(f"  Model: {model_name}")
        print(f"  USE_CACHE={USE_CACHE} | ATTN={ATTN_IMPL} | Warmup={WARMUP_TOKENS} tokens/text")
        print(f"{'='*65}")

        try:
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            try:
                model = AutoModelForCausalLM.from_pretrained(
                    model_id, device_map="auto", torch_dtype=torch.float16,
                    attn_implementation=ATTN_IMPL)
            except Exception:
                print(f"  [note] attn_implementation not supported, using default")
                model = AutoModelForCausalLM.from_pretrained(
                    model_id, device_map="auto", torch_dtype=torch.float16)
            model.eval()

            print(f"\n  --- Run A (standard order: Fraud first) ---")
            res_a, steps_a = run_experiment(model, tokenizer, model_name, "run_A", standard_order)
            all_results.extend(res_a)
            all_steps.extend(steps_a)

            print(f"\n  --- Run B (reversed order: Certain Honest first) ---")
            res_b, steps_b = run_experiment(model, tokenizer, model_name, "run_B", reversed_order)
            all_results.extend(res_b)
            all_steps.extend(steps_b)

            del model, tokenizer
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback; traceback.print_exc()

    # Save data
    print(f"\nSaving...")
    with open(RESULTS_PATH, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"  {RESULTS_PATH}")

    with open(TRAJ_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=traj_fields)
        writer.writeheader()
        writer.writerows(all_steps)
    print(f"  {TRAJ_PATH}")

    # Plots
    print(f"\nPlots...")
    plot_power_bars(all_results, "run_A")
    plot_power_bars(all_results, "run_B")
    plot_order_check(all_results)
    for m in MODELS:
        plot_trajectories(all_results, all_steps, m, "run_A")
    plot_synthesis_scatter(all_results)

    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY — Stable-step power (W) — Run A vs Run B")
    print(f"{'='*80}")
    print(f"{'Model':<16}{'Domain':<22}{'Run':<6}{'Fraud':>8}{'Compl':>8}{'Hedged':>8}{'Certain':>8}")
    print("-"*80)
    table = defaultdict(dict)
    for r in all_results:
        table[(r["model"], r["domain"], r["run"])][r["text_type"]] = \
            r["metrics"]["stable_mean_power_watts"]
    for m in MODELS:
        for d in CORPUS:
            for run in ["run_A", "run_B"]:
                row = table.get((m, d, run), {})
                print(
                    f"{m:<16}{d.split('_',1)[1]:<22}{run:<6}"
                    f"{row.get('Type_1_Fraudulent',0):>8.1f}"
                    f"{row.get('Type_2_Compliant',0):>8.1f}"
                    f"{row.get('Type_3_Hedged_Honest',0):>8.1f}"
                    f"{row.get('Type_4_Certain_Honest',0):>8.1f}"
                )

    print(f"\n{'='*80}")
    print("THERMAL CHECK — Fraud cheapest in both runs = REAL SIGNAL")
    print(f"{'='*80}")
    for m in MODELS:
        for d in CORPUS:
            row_a = table.get((m, d, "run_A"), {})
            row_b = table.get((m, d, "run_B"), {})
            fa = row_a.get("Type_1_Fraudulent", 0)
            ca = row_a.get("Type_2_Compliant", 0)
            fb = row_b.get("Type_1_Fraudulent", 0)
            cb = row_b.get("Type_2_Compliant", 0)
            a_low = fa < ca
            b_low = fb < cb
            verdict = ("REAL SIGNAL" if a_low and b_low else
                       "THERMAL ARTIFACT" if a_low and not b_low else
                       "REVERSED" if not a_low and b_low else "NOT CHEAPEST")
            print(f"  {m} | {d.split('_',1)[1]}: RunA={a_low} RunB={b_low} -> {verdict}")

    print(f"\nComplete. Copy files:")
    print(f"  cp {RESULTS_PATH} .")
    print(f"  cp {TRAJ_PATH} .")
    print(f"  cp -r {PLOTS_DIR}/ .")


if __name__ == "__main__":
    main()
