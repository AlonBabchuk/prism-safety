# Historical archive — sys.path splice references
# phase2/scripts/ which no longer exists after
# repository restructure. See
# archive/phase2_token_level/scripts/ for helpers.
"""
PRISM Phase 3 — real-time intervention experiment for Llama 3.1 8B.

Extends Phase 2 by monitoring KL divergence during generation and firing
a temperature-boost intervention whenever three consecutive steps show an
alternating high-low-high (or low-high-low) KL pattern with both swings
exceeding an amplitude threshold.

Produces four runs of 50 tokens each:
    1. coherent  + baseline      (greedy, no monitor)
    2. coherent  + intervened    (monitor on, sample at T=1.5 when fired)
    3. distorted + baseline
    4. distorted + intervened

Outputs:
    phase3/results/signals_raw_phase3.csv
    phase3/results/plots/*.png
    phase3/results/outputs_for_scoring.txt
    phase3/scripts/intervention_log.json
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

# Phase 2 lives as a sibling package — splice its scripts dir onto sys.path
# so we can import the model/hook/signal helpers instead of duplicating them.
SCRIPT_DIR = Path(__file__).resolve().parent              # phase3/scripts
PHASE3_DIR = SCRIPT_DIR.parent                            # phase3
REPO_ROOT = PHASE3_DIR.parent                             # repo root
PHASE2_SCRIPTS = REPO_ROOT / "phase2" / "scripts"
sys.path.insert(0, str(PHASE2_SCRIPTS))

from run_phase2_llama import (                            # noqa: E402
    load_model,
    attach_hooks,
    remove_hooks,
    compute_token_signals,
    compute_attention_signals,
    COHERENT_TEXT,
    DISTORTED_TEXT,
    N_GENERATION_STEPS,
    DEVICE,
)


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

RESULTS_DIR = PHASE3_DIR / "results"
PLOTS_DIR = RESULTS_DIR / "plots"
CSV_PATH = RESULTS_DIR / "signals_raw_phase3.csv"
SCORING_PATH = RESULTS_DIR / "outputs_for_scoring.txt"
TRIGGER_LOG_PATH = SCRIPT_DIR / "intervention_log.json"

# Intervention parameters.
TRIGGER_AMPLITUDE = 3.0      # both KL swings in the 3-step window must exceed this
INTERVENTION_TEMPERATURE = 1.5
INTERVENTION_DURATION = 5    # number of generation steps with elevated temperature

# Reproducibility for the sampling runs.
RANDOM_SEED = 0


# --------------------------------------------------------------------------
# Trigger detection
# --------------------------------------------------------------------------

def detect_alternating_kl(kl_window, amplitude=TRIGGER_AMPLITUDE):
    """Return True iff the three KL values form an alternating pattern
    (v0,v1,v2 with v1 strictly higher or strictly lower than both neighbours)
    AND both swings |v0-v1| and |v2-v1| exceed `amplitude`.

    `kl_window` is a sequence of three floats; NaNs disqualify the window.
    """
    if len(kl_window) < 3:
        return False
    v0, v1, v2 = kl_window
    if any(np.isnan([v0, v1, v2])):
        return False

    high_low_high = (v1 < v0) and (v1 < v2)
    low_high_low = (v1 > v0) and (v1 > v2)
    if not (high_low_high or low_high_low):
        return False

    swing_left = abs(v0 - v1)
    swing_right = abs(v2 - v1)
    return min(swing_left, swing_right) > amplitude


# --------------------------------------------------------------------------
# Token selection
# --------------------------------------------------------------------------

def select_next_token(logits_last, temperature):
    """Greedy argmax when temperature == 1.0, multinomial sample otherwise.

    The spec defines "greedy (1.0)" as the baseline mode, so T=1.0 keeps
    that contract; any other temperature switches to stochastic sampling.
    """
    if temperature == 1.0:
        return torch.argmax(logits_last).view(1, 1)
    scaled = logits_last.float() / temperature
    probs = F.softmax(scaled, dim=-1)
    sampled = torch.multinomial(probs, num_samples=1)
    return sampled.view(1, 1)


# --------------------------------------------------------------------------
# Generation with optional intervention monitor
# --------------------------------------------------------------------------

def run_generation_with_intervention(
    model,
    tokenizer,
    text,
    label,
    intervention_enabled,
    n_steps=N_GENERATION_STEPS,
):
    """Decode `n_steps` tokens from `text`.

    If `intervention_enabled`, the monitor watches the last three KL
    divergences each step. When an alternating pattern with amplitude
    > TRIGGER_AMPLITUDE is detected, the next INTERVENTION_DURATION
    token selections switch to temperature sampling at T=1.5. Re-firing
    inside an active window simply refreshes the countdown.

    Returns (DataFrame of per-step rows, list of trigger-firing log entries,
    decoded generated text).
    """
    print(f"[run] {label} (intervention={'on' if intervention_enabled else 'off'}) ...")
    torch.manual_seed(RANDOM_SEED)  # only relevant when sampling kicks in

    captured, handles = attach_hooks(model)
    enc = tokenizer(text, return_tensors="pt").to(DEVICE)
    input_ids = enc["input_ids"]
    prompt_len = input_ids.shape[1]

    rows = []
    triggers = []
    prev_probs = None
    prev_H = None
    intervention_remaining = 0   # steps left at elevated temperature
    kl_history = []              # rolling list of D_t values

    try:
        for step in range(n_steps):
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
            kl_history.append(tok_sig["D"])

            # Trigger check: only meaningful with a 3-step KL window.
            trigger_fired = False
            if intervention_enabled and len(kl_history) >= 3:
                window = kl_history[-3:]
                if detect_alternating_kl(window):
                    trigger_fired = True
                    intervention_remaining = INTERVENTION_DURATION
                    triggers.append({
                        "run_label": label,
                        "step": step,
                        "kl_window": [
                            None if (v is None or np.isnan(v)) else float(v)
                            for v in window
                        ],
                    })
                    print(f"  [trigger] step={step} window={window}")

            # Decide sampling temperature for the token chosen at THIS step.
            if intervention_remaining > 0:
                temperature = INTERVENTION_TEMPERATURE
                intervention_active = True
                intervention_remaining -= 1
            else:
                temperature = 1.0
                intervention_active = False

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
                "intervention_active": intervention_active,
                "trigger_fired": trigger_fired,
            })

            next_id = select_next_token(logits_last, temperature)
            input_ids = torch.cat([input_ids, next_id], dim=1)

            prev_probs = tok_sig["probs"]
            prev_H = tok_sig["H"]
    finally:
        remove_hooks(handles)

    generated_ids = input_ids[0, prompt_len:].tolist()
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return pd.DataFrame(rows), triggers, generated_text


# --------------------------------------------------------------------------
# Plotting — 4 panels per signal
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

PANEL_LAYOUT = [
    ("coherent_baseline",   "Coherent — baseline"),
    ("coherent_intervened", "Coherent — intervened"),
    ("distorted_baseline",  "Distorted — baseline"),
    ("distorted_intervened","Distorted — intervened"),
]


def _annotate_intervention(ax, df):
    """Shade steps where intervention was active; mark trigger-firing steps."""
    if "intervention_active" in df.columns:
        active = df.loc[df["intervention_active"], "step"].tolist()
        for s in active:
            ax.axvspan(s - 0.4, s + 0.4, color="orange", alpha=0.15, linewidth=0)
    if "trigger_fired" in df.columns:
        for s in df.loc[df["trigger_fired"], "step"].tolist():
            ax.axvline(s, color="red", linestyle="--", alpha=0.6, linewidth=1)


def _plot_four_panels(run_dfs, plot_calls, title, filename, out_dir):
    """Draw a 2x2 figure: rows = text type, cols = baseline vs intervened."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
    flat_axes = axes.flatten()
    for ax, (run_key, panel_title) in zip(flat_axes, PANEL_LAYOUT):
        df = run_dfs[run_key]
        _annotate_intervention(ax, df)
        for col, lbl in plot_calls:
            ax.plot(df["step"], df[col], marker="o", markersize=3, label=lbl)
        ax.set_title(panel_title)
        ax.set_xlabel("generation step")
        ax.grid(True, alpha=0.3)
        if len(plot_calls) > 1:
            ax.legend(loc="best", fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    out_path = out_dir / f"{filename}.png"
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"[plot] wrote {out_path}")


def generate_plots(run_dfs, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    for col, title in SINGLE_LINE_SIGNALS:
        _plot_four_panels(run_dfs, [(col, col)], title, f"signal_{col}", out_dir)
    _plot_four_panels(run_dfs, TOPK_SIGNAL["columns"],
                      TOPK_SIGNAL["title"], TOPK_SIGNAL["filename"], out_dir)


# --------------------------------------------------------------------------
# Scoring-bundle text output
# --------------------------------------------------------------------------

def write_scoring_bundle(generations, path: Path):
    """Write the four generated continuations in a labeled, scorer-friendly
    format so an external AI scorer can read each block unambiguously.
    """
    blocks = []
    for run_key, panel_title in PANEL_LAYOUT:
        prompt, generated = generations[run_key]
        blocks.append(
            f"=== {panel_title.upper()} ===\n\n"
            f"--- PROMPT ---\n{prompt}\n\n"
            f"--- GENERATED ---\n{generated}\n"
        )
    path.write_text("\n".join(blocks), encoding="utf-8")
    print(f"[save] wrote scoring bundle to {path}")


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------

def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    model, tokenizer = load_model()

    # The four runs: same prompts as Phase 2, with/without monitor.
    run_specs = [
        ("coherent_baseline",   COHERENT_TEXT,  False),
        ("coherent_intervened", COHERENT_TEXT,  True),
        ("distorted_baseline",  DISTORTED_TEXT, False),
        ("distorted_intervened",DISTORTED_TEXT, True),
    ]

    run_dfs = {}
    generations = {}      # run_key -> (prompt, generated_text)
    all_triggers = []

    for run_key, prompt, intervention_on in run_specs:
        df, triggers, generated = run_generation_with_intervention(
            model, tokenizer, prompt, run_key, intervention_on,
        )
        run_dfs[run_key] = df
        generations[run_key] = (prompt, generated)
        all_triggers.extend(triggers)

    # 1. Combined CSV across all four runs (each row tagged by `label`).
    combined = pd.concat(run_dfs.values(), ignore_index=True)
    combined.to_csv(CSV_PATH, index=False)
    print(f"[save] wrote raw signals to {CSV_PATH}")

    # 2. Plots — one PNG per signal, four panels each.
    generate_plots(run_dfs, PLOTS_DIR)

    # 3. Scoring bundle with all four generated texts.
    write_scoring_bundle(generations, SCORING_PATH)

    # 4. Trigger firing log.
    log_payload = {
        "trigger_amplitude": TRIGGER_AMPLITUDE,
        "intervention_temperature": INTERVENTION_TEMPERATURE,
        "intervention_duration": INTERVENTION_DURATION,
        "fires": all_triggers,
        "total_fires": len(all_triggers),
    }
    TRIGGER_LOG_PATH.write_text(json.dumps(log_payload, indent=2), encoding="utf-8")
    print(f"[save] wrote intervention log to {TRIGGER_LOG_PATH}")

    print("[done] PRISM Phase 3 intervention run complete.")


if __name__ == "__main__":
    main()
