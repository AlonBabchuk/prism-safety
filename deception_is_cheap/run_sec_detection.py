"""
Deception is Cheap — Energy Consumption Experiment
===================================================
Measures GPU power consumption when four AI architectures
process legally-labeled text pairs: fraudulent communications
(verbatim from federal complaints) versus compliant
communications (from regulated filings).

Hardware: GPU with 80GB+ VRAM recommended for all four models.
See README.md for full setup instructions.

Results save to /tmp/prism_sec/{model_slug}/{pair_id}/
To run on one model, comment out unwanted entries in MODELS.
"""

import gc
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Hook + signal helpers live in the archived phase 2 token-level scripts
# folder (kept there to preserve the original reviewed implementation).
SCRIPT_DIR = Path(__file__).resolve().parent              # deception_is_cheap/
REPO_ROOT = SCRIPT_DIR.parent                             # repo root
PHASE2_SCRIPTS = REPO_ROOT / "archive" / "phase2_token_level" / "scripts"
sys.path.insert(0, str(PHASE2_SCRIPTS))

from run_phase2_llama import (                            # noqa: E402
    attach_hooks,
    remove_hooks,
    compute_token_signals,
    compute_attention_signals,
)


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

N_GENERATION_STEPS = 50
DEVICE = "cuda"

# Per-pair output goes here. Workspace disks may be quota-limited, so we
# write the bulky CSVs and plots to /tmp; on Windows this maps to the
# current drive's \tmp directory which is fine for ad-hoc runs.
OUTPUT_ROOT = Path("/tmp/prism_sec")

# (huggingface_id, output_slug, torch_dtype). Gemma 2 was trained in
# bfloat16; loading it in fp16 risks silent overflow on its logits.
MODELS = [
    ("meta-llama/Meta-Llama-3.1-8B", "llama-3.1-8b", torch.float16),
    ("mistralai/Mistral-7B-v0.1",    "mistral-7b",   torch.float16),
    ("Qwen/Qwen2.5-7B",              "qwen-2.5-7b",  torch.float16),
    ("google/gemma-2-9b",            "gemma-2-9b",   torch.bfloat16),
]


# --------------------------------------------------------------------------
# Text pairs (verbatim — see SEC complaints / EDGAR filings for context)
# --------------------------------------------------------------------------

FRAUDULENT_1 = (
    "Investors' net invested capital is 100% protected from loss. "
    "The General Partner maintains a balance sheet in excess of $250 million "
    "that could be drawn upon in the unlikely event that an investor's capital "
    "was not returned in full. MG Capital has a proven track record — "
    "10-year investment performance exceeding S&P 500 returns, 20+ years of "
    "deep investment expertise, consistent year-over-year growth for investors "
    "since 2000. Fund I raised $350 million of investor capital and earned a "
    "gross return on investment of 38.99%, outperforming the S&P 500 Index by "
    "4.5-to-1. Strong cash flow from more than 215 high-quality, long-term "
    "corporate tenant partnerships with non-cancelable, multi-year lease "
    "agreements with built-in rent escalations ranging from 4-10% per annum. "
    "MG Capital targets a 20% gross annual IRR on investments and a gross "
    "investment multiple of 2.6x. Historically this exit strategy has resulted "
    "in well-timed liquidity events for our direct global investors."
)

CLEAN_1 = (
    "These statements are not guarantees of future performance and are subject "
    "to risks, uncertainties and other factors, some of which are beyond our "
    "control and difficult to predict, and could cause actual results to differ "
    "materially from those expressed. General economic conditions including "
    "heightened inflation, slower growth or recession, changes to monetary "
    "policy, higher interest rates and currency fluctuations may adversely "
    "affect our results. An economic downturn could disproportionately impact "
    "investments we target, potentially causing a decrease in investment "
    "opportunities and diminished demand for capital. The speculative and "
    "illiquid nature of our investments means investors could lose some or all "
    "of their investment. Although we believe our assumptions are reasonable, "
    "any could prove inaccurate and actual results could differ materially "
    "from those expressed or forecasted in the forward-looking statements."
)

FRAUDULENT_2 = (
    "Start Options is the largest Bitcoin exchange in euro volume and "
    "liquidity, consistently rated the best and most secure Bitcoin exchange "
    "by independent news media. The B2G tokens will be deliverable on the "
    "Ethereum blockchain. Invested funds will be used to develop a coin that "
    "is mineable. Tokens will be tradeable on a proprietary digital asset "
    "trading platform at the platform's launch in early April 2018. "
    "This is a fully operational, revenue-generating exchange with "
    "institutional-grade security and unmatched liquidity depth across all "
    "major trading pairs. Our technology infrastructure is built to "
    "enterprise standards and has processed billions in verified transactions. "
    "Early investors are guaranteed priority access and preferential rates "
    "at launch. This opportunity will not be available at these terms once "
    "the platform goes live."
)

CLEAN_2 = (
    "It is not possible for our management to predict all risks, nor can we "
    "assess the impact of all factors on our business or the extent to which "
    "any factor may cause actual results to differ materially from those "
    "contained in forward-looking statements. We cannot guarantee future "
    "results, levels of activity, performance or achievements. We have "
    "acquired and may in the future acquire other companies or technologies "
    "which could divert management attention and adversely affect operating "
    "results. You should not consider our recent growth in revenues as "
    "indicative of future performance. We cannot assure you that we will "
    "achieve profitability in the future. New risks emerge from time to time "
    "and it is not possible to predict all risk factors, nor can we address "
    "the impact of all factors on our business."
)

FRAUDULENT_3 = (
    "Guess and Co. is a diversified energy, health care, technology, and "
    "real estate company that has earned millions of dollars in revenue from "
    "its business operations from 2019 to 2021. We project the company will "
    "earn billions in revenue in both 2021 and 2022. Our diversified portfolio "
    "across multiple high-growth sectors positions us uniquely to deliver "
    "substantial returns to investors. Our operations are fully established "
    "and generating consistent cash flows across all divisions. The company "
    "has demonstrated sustained profitability and is on track to become a "
    "market leader across each of its core verticals within 24 months. "
    "Management has over 30 years of combined operational experience and has "
    "successfully scaled multiple enterprises to nine-figure valuations."
)

CLEAN_3 = (
    "There can be no assurance that future developments affecting us will be "
    "those we have anticipated. These forward-looking statements involve "
    "risks and uncertainties, some beyond our control, that may cause actual "
    "results to differ materially from those expressed. We cannot guarantee "
    "that our plans, intentions or expectations will be achieved. Actual "
    "results could differ materially and adversely from those anticipated. "
    "Additional factors or events that could cause our actual results to "
    "differ may emerge from time to time, and it is not possible for us to "
    "predict all of them. Our forward-looking statements are based on "
    "information currently available and speak only as of the date on which "
    "they were made. We undertake no obligation to update publicly any "
    "forward-looking statements whether as a result of new information, "
    "future events, or otherwise."
)

TEXT_PAIRS = [
    ("pair1_realestate",    FRAUDULENT_1, CLEAN_1),
    ("pair2_tech",          FRAUDULENT_2, CLEAN_2),
    ("pair3_diversified",   FRAUDULENT_3, CLEAN_3),
]


# --------------------------------------------------------------------------
# NVML energy sampling (graceful fallback if pynvml is missing)
# --------------------------------------------------------------------------

PYNVML_HANDLE = None
PYNVML_OK = False
try:
    import pynvml  # type: ignore
    pynvml.nvmlInit()
    PYNVML_HANDLE = pynvml.nvmlDeviceGetHandleByIndex(0)
    PYNVML_OK = True
    print("[nvml] initialised; will sample power + temperature each step")
except ImportError:
    print("[nvml] pynvml not installed — gpu_power_w / gpu_temp_c will be None")
except Exception as exc:                                  # NVML init can fail
    print(f"[nvml] init failed ({exc}); energy columns will be None")


def sample_energy():
    """Return (power_watts, temperature_celsius) or (None, None)."""
    if not PYNVML_OK:
        return None, None
    try:
        power_mw = pynvml.nvmlDeviceGetPowerUsage(PYNVML_HANDLE)
        temp_c = pynvml.nvmlDeviceGetTemperature(
            PYNVML_HANDLE, pynvml.NVML_TEMPERATURE_GPU
        )
        return power_mw / 1000.0, int(temp_c)
    except Exception:
        # A transient NVML failure shouldn't kill the run.
        return None, None


# --------------------------------------------------------------------------
# Model loading
# --------------------------------------------------------------------------

def load_model_for_arch(model_name: str, dtype: torch.dtype):
    """Load tokenizer + causal LM on CUDA with eager attention."""
    print(f"[load] {model_name} ({dtype}) ...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map=DEVICE,
        attn_implementation="eager",
    )
    model.eval()
    return model, tokenizer


# --------------------------------------------------------------------------
# Generation loop with PRISM signals + per-step energy
# --------------------------------------------------------------------------

def run_generation_with_energy(model, tokenizer, text, label):
    """Greedy decode `N_GENERATION_STEPS` tokens, collecting all 7 PRISM
    signals AND a GPU power/temperature snapshot at each step.

    Mirrors run_phase2_llama.run_generation but adds the NVML sample and
    the two extra CSV columns. Hooks/signal helpers come from phase2.
    """
    print(f"[gen] {label} ...")
    captured, handles = attach_hooks(model)
    enc = tokenizer(text, return_tensors="pt").to(DEVICE)
    input_ids = enc["input_ids"]

    rows = []
    prev_probs = None
    prev_H = None

    try:
        for step in range(N_GENERATION_STEPS):
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
            delta_H = float("nan") if prev_H is None else tok_sig["H"] - prev_H
            power_w, temp_c = sample_energy()

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
                "gpu_power_w": power_w,
                "gpu_temp_c": temp_c,
            })

            next_id = torch.argmax(logits_last).view(1, 1)
            input_ids = torch.cat([input_ids, next_id], dim=1)
            prev_probs = tok_sig["probs"]
            prev_H = tok_sig["H"]
    finally:
        remove_hooks(handles)

    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Plots — side-by-side clean vs fraudulent per signal
# --------------------------------------------------------------------------

SINGLE_LINE_SIGNALS = [
    ("H",       "Token entropy H_t"),
    ("B",       "Branching factor B_t"),
    ("D",       "KL divergence D_t"),
    ("delta_H", "Entropy gradient delta_H_t"),
    ("H_attn",  "Attention entropy"),
    ("S",       "Attention span"),
]

TOPK_SIGNAL = {
    "columns": [("C1", "k=1"), ("C5", "k=5"), ("C10", "k=10")],
    "filename": "signal_C_topk",
    "title": "Top-k mass concentration C_t",
}


def _plot_pair(df_clean, df_fraud, plot_calls, title, filename, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for ax, df, panel_title, color in (
        (axes[0], df_clean, "Compliant (clean)", "tab:blue"),
        (axes[1], df_fraud, "Fraudulent (SEC-labeled)", "tab:red"),
    ):
        for col, lbl in plot_calls:
            ax.plot(df["step"], df[col], marker="o", markersize=3, label=lbl,
                    color=color if len(plot_calls) == 1 else None)
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


def generate_pair_plots(df_clean, df_fraud, out_dir: Path):
    """Same seven plots as Phase 2, relabeled for the SEC context."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for col, title in SINGLE_LINE_SIGNALS:
        _plot_pair(df_clean, df_fraud, [(col, col)], title,
                   f"signal_{col}", out_dir)
    _plot_pair(df_clean, df_fraud, TOPK_SIGNAL["columns"],
               TOPK_SIGNAL["title"], TOPK_SIGNAL["filename"], out_dir)


# --------------------------------------------------------------------------
# Per-model runner
# --------------------------------------------------------------------------

def _nan_safe_mean(series):
    """np.nanmean over a pandas series that may be all-None (pynvml off)."""
    arr = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    if np.all(np.isnan(arr)):
        return None
    return float(np.nanmean(arr))


def run_one_model(model_name, slug, dtype):
    """Run all three pairs on one model. Returns a list of dicts holding
    the per-pair summary stats used to assemble the final cross-model
    table — one dict per (pair, label) combination."""
    print(f"\n========== {model_name} ==========")
    model_root = OUTPUT_ROOT / slug
    model_root.mkdir(parents=True, exist_ok=True)
    energy_summary = {}
    table_rows = []

    model, tokenizer = load_model_for_arch(model_name, dtype)

    try:
        for pair_id, fraud_text, clean_text in TEXT_PAIRS:
            pair_dir = model_root / pair_id
            pair_dir.mkdir(parents=True, exist_ok=True)
            plots_dir = pair_dir / "plots"

            df_fraud = run_generation_with_energy(
                model, tokenizer, fraud_text, "fraudulent")
            df_clean = run_generation_with_energy(
                model, tokenizer, clean_text, "clean")

            combined = pd.concat([df_fraud, df_clean], ignore_index=True)
            combined.to_csv(pair_dir / "signals_raw.csv", index=False)
            generate_pair_plots(df_clean, df_fraud, plots_dir)
            print(f"[save] {slug}/{pair_id} -> csv + plots")

            energy_summary[pair_id] = {
                "fraudulent": {
                    "watts_mean": _nan_safe_mean(df_fraud["gpu_power_w"]),
                    "temp_mean":  _nan_safe_mean(df_fraud["gpu_temp_c"]),
                },
                "clean": {
                    "watts_mean": _nan_safe_mean(df_clean["gpu_power_w"]),
                    "temp_mean":  _nan_safe_mean(df_clean["gpu_temp_c"]),
                },
            }

            # One row per pair, fed into the cross-model summary table.
            table_rows.append({
                "model": slug,
                "pair": pair_id,
                "fraud_attn_span_mean": float(df_fraud["S"].mean()),
                "clean_attn_span_mean": float(df_clean["S"].mean()),
                "fraud_watts_mean": energy_summary[pair_id]["fraudulent"]["watts_mean"],
                "clean_watts_mean": energy_summary[pair_id]["clean"]["watts_mean"],
            })

        (model_root / "energy_summary.json").write_text(
            json.dumps(energy_summary, indent=2), encoding="utf-8"
        )
        print(f"[save] {slug}/energy_summary.json")
    finally:
        # GPU memory release before the next model loads. Without this, a
        # sequential 8B + 7B + 7B + 9B schedule will OOM on a single GPU.
        del model, tokenizer
        gc.collect()
        torch.cuda.empty_cache()
        print(f"[gc] released {slug}")

    return table_rows


# --------------------------------------------------------------------------
# Cross-model summary table
# --------------------------------------------------------------------------

def print_summary(all_rows):
    header = (
        f"{'model':<14} {'pair':<22} "
        f"{'F-attn-span':>11} {'C-attn-span':>11}  "
        f"{'F-watts':>9} {'C-watts':>9} {'F/C':>6}"
    )
    sep = "-" * len(header)
    print()
    print(sep)
    print(header)
    print(sep)
    for r in all_rows:
        f_w = r["fraud_watts_mean"]
        c_w = r["clean_watts_mean"]
        if f_w is not None and c_w is not None and c_w > 0:
            ratio = f"{f_w / c_w:>6.3f}"
        else:
            ratio = f"{'n/a':>6}"
        f_w_s = f"{f_w:>9.2f}" if f_w is not None else f"{'n/a':>9}"
        c_w_s = f"{c_w:>9.2f}" if c_w is not None else f"{'n/a':>9}"
        print(
            f"{r['model']:<14} {r['pair']:<22} "
            f"{r['fraud_attn_span_mean']:>11.3f} "
            f"{r['clean_attn_span_mean']:>11.3f}  "
            f"{f_w_s} {c_w_s} {ratio}"
        )
    print(sep)


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------

def main():
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    all_rows = []
    for model_name, slug, dtype in MODELS:
        all_rows.extend(run_one_model(model_name, slug, dtype))
    print_summary(all_rows)
    if PYNVML_OK:
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass
    print("[done] PRISM SEC detection sweep complete.")


if __name__ == "__main__":
    main()
