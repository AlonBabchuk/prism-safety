"""
PRISM Phase 3 — instruct variant on meta-llama/Meta-Llama-3.1-8B-Instruct.

This is the instruct counterpart to the base-model runs. The architecture
is identical to Meta-Llama-3.1-8B; the difference is instruction tuning,
which lets us use the model's native chat template for meaningful
continuation of the speaker rather than relying on prefix prompting.

Produces six runs:
    1. coherent_baseline      — chat template, greedy 50 tokens, no monitor
    2. coherent_intervened    — chat template, greedy 50 tokens, monitor on
    3. distorted_baseline     — chat template, greedy 50 tokens, no monitor
    4. distorted_intervened   — chat template, greedy 50 tokens, monitor on
    5. coherent_detection     — Phase-2-style read-only signals over the
                                raw passage (no generation, no chat template)
    6. distorted_detection    — same

The intervention is the v2 state-injection mechanism (decoding stays
greedy; on a trigger fire we pause and append a short state-description
string to the running context).
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

# Phase 2 lives as a sibling package; splice its script directory onto
# sys.path so the model/hook/signal helpers can be imported directly.
SCRIPT_DIR = Path(__file__).resolve().parent              # phase3/scripts
PHASE3_DIR = SCRIPT_DIR.parent                            # phase3
REPO_ROOT = PHASE3_DIR.parent                             # repo root
PHASE2_SCRIPTS = REPO_ROOT / "phase2" / "scripts"
sys.path.insert(0, str(PHASE2_SCRIPTS))
# Phase 3 v1/v2 live in the same dir as this script, so importing by
# bare module name works without further path tweaks.

from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402
from run_phase2_llama import (                            # noqa: E402
    attach_hooks,
    remove_hooks,
    compute_token_signals,
    compute_attention_signals,
    COHERENT_TEXT,
    DISTORTED_TEXT,
    N_GENERATION_STEPS,
    DEVICE,
)
from run_phase3_intervention import (                     # noqa: E402
    detect_alternating_kl,
    TRIGGER_AMPLITUDE,
    write_scoring_bundle,
)
from run_phase3_intervention_v2 import INJECTION_TEXT     # noqa: E402


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B-Instruct"

# All output paths are instruct-specific so they don't clobber the
# base-model runs already produced by the v1/v2 scripts.
RESULTS_DIR = PHASE3_DIR / "results"
PLOTS_DIR = RESULTS_DIR / "plots_instruct"
CSV_PATH = RESULTS_DIR / "signals_raw_instruct.csv"
SCORING_PATH = RESULTS_DIR / "outputs_for_scoring_instruct.txt"
TRIGGER_LOG_PATH = SCRIPT_DIR / "intervention_log_instruct.json"

SYSTEM_PROMPT = (
    "You are continuing a speech. Continue the following passage in "
    "exactly the same voice, style, and argument. Do not summarize, "
    "comment, or break character — simply continue speaking as the "
    "same speaker."
)


# --------------------------------------------------------------------------
# Model loading (bfloat16, overrides phase2's float16 default)
# --------------------------------------------------------------------------

def load_instruct_model(model_name: str = MODEL_NAME):
    """Load the instruct tokenizer + causal LM in bfloat16 on CUDA.

    The Meta-Llama-3.1 instruct weights were trained in bfloat16. Loading
    them in float16 (phase 2's default) risks numerical drift and silent
    overflow in the high-magnitude logits common to instruction-tuned
    models. Eager attention is kept so attention weights are still
    materialised for the hooks.
    """
    print(f"[load_instruct_model] loading {model_name} in bfloat16 ...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map=DEVICE,
        attn_implementation="eager",
    )
    model.eval()
    return model, tokenizer


# --------------------------------------------------------------------------
# Shared helpers
# --------------------------------------------------------------------------

def build_chat_input(tokenizer, passage):
    """Tokenize a passage through the model's chat template.

    `add_generation_prompt=True` appends the assistant-turn header so the
    model knows it should continue speaking next. Returns the full
    [1, seq] input_ids tensor on DEVICE.
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": passage},
    ]
    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(DEVICE)
    return input_ids


def _row(label, step, tok_sig, attn_sig, delta_H,
         intervention_active=False, trigger_fired=False):
    """Build the dict that becomes one CSV row. Schema is shared by
    every run so the combined frame stacks cleanly."""
    return {
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
    }


# --------------------------------------------------------------------------
# Run type 1 — baseline continuation (greedy, no intervention)
# --------------------------------------------------------------------------

def run_baseline_continuation(model, tokenizer, passage, label,
                              n_steps=N_GENERATION_STEPS):
    """Chat-templated greedy continuation. No monitor."""
    print(f"[baseline] {label} ...")
    captured, handles = attach_hooks(model)
    input_ids = build_chat_input(tokenizer, passage)
    prompt_len = input_ids.shape[1]

    rows = []
    prev_probs = None
    prev_H = None

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
            delta_H = float("nan") if prev_H is None else tok_sig["H"] - prev_H

            rows.append(_row(label, step, tok_sig, attn_sig, delta_H))

            next_id = torch.argmax(logits_last).view(1, 1)
            input_ids = torch.cat([input_ids, next_id], dim=1)
            prev_probs = tok_sig["probs"]
            prev_H = tok_sig["H"]
    finally:
        remove_hooks(handles)

    generated_text = tokenizer.decode(
        input_ids[0, prompt_len:].tolist(), skip_special_tokens=True
    )
    return pd.DataFrame(rows), generated_text


# --------------------------------------------------------------------------
# Run type 2 — intervened continuation (greedy + state injection on trigger)
# --------------------------------------------------------------------------

def run_intervened_continuation(model, tokenizer, passage, label,
                                n_steps=N_GENERATION_STEPS):
    """Chat-templated greedy continuation with the v2 state-injection
    intervention. On a trigger fire we pause this step (no greedy emit)
    and append INJECTION_TEXT to the running context so the next forward
    pass conditions on the model's own destabilisation label.
    """
    print(f"[intervened] {label} ...")

    injection_ids = tokenizer.encode(INJECTION_TEXT, add_special_tokens=False)
    injection_tensor = torch.tensor([injection_ids], device=DEVICE,
                                    dtype=torch.long)

    captured, handles = attach_hooks(model)
    input_ids = build_chat_input(tokenizer, passage)
    prompt_len = input_ids.shape[1]
    injected_positions = set()  # positions in input_ids that came from injections

    rows = []
    injections = []
    prev_probs = None
    prev_H = None
    kl_history = []

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

            # Same trigger predicate as v1/v2: rolling 3-step KL window,
            # alternating sign, both swings exceeding the amplitude floor.
            trigger_fired = False
            if len(kl_history) >= 3:
                window = kl_history[-3:]
                if detect_alternating_kl(window, amplitude=TRIGGER_AMPLITUDE):
                    trigger_fired = True
                    injections.append({
                        "run_label": label,
                        "step": step,
                        "kl_window": [
                            None if (v is None or np.isnan(v)) else float(v)
                            for v in window
                        ],
                        "injected_token_ids": list(injection_ids),
                        "injected_text": INJECTION_TEXT,
                    })
                    print(f"  [inject] step={step} ids={injection_ids}")

            delta_H = float("nan") if prev_H is None else tok_sig["H"] - prev_H
            rows.append(_row(label, step, tok_sig, attn_sig, delta_H,
                             intervention_active=trigger_fired,
                             trigger_fired=trigger_fired))

            if trigger_fired:
                # Pause — no greedy token this step. Inject the state
                # tokens; next iteration's forward pass consumes them.
                base_len = input_ids.shape[1]
                for offset in range(injection_tensor.shape[1]):
                    injected_positions.add(base_len + offset)
                input_ids = torch.cat([input_ids, injection_tensor], dim=1)
            else:
                next_id = torch.argmax(logits_last).view(1, 1)
                input_ids = torch.cat([input_ids, next_id], dim=1)

            prev_probs = tok_sig["probs"]
            prev_H = tok_sig["H"]
    finally:
        remove_hooks(handles)

    # Strip prompt and injected spans so the scoring bundle shows only
    # the model's own continuation.
    full_ids = input_ids[0].tolist()
    generated_ids = [
        tid for pos, tid in enumerate(full_ids)
        if pos >= prompt_len and pos not in injected_positions
    ]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return pd.DataFrame(rows), injections, generated_text


# --------------------------------------------------------------------------
# Run type 3 — detection-only (no generation, no chat template)
# --------------------------------------------------------------------------

def run_detection(model, tokenizer, passage, label):
    """Read per-position signals from the raw passage.

    Phase-2 protocol applied to the instruct model: feed the passage as
    plain text (no chat template), then for every token position t read
    the signals as if t were the "current step". This gives a directly
    comparable baseline to Phase 2's signal trajectories on the same
    input texts.

    Implementation: one full forward pass returns logits and attention
    over the entire prefix; we then iterate positions, sliding the
    "last query" window down the sequence by slicing attention tensors.
    Causal masking guarantees the per-position signals match what a
    series of prefix-by-prefix forward passes would have produced.
    """
    print(f"[detection] {label} ...")
    captured, handles = attach_hooks(model)

    # Plain text — explicitly NOT the chat template, to match Phase 2.
    enc = tokenizer(passage, return_tensors="pt").to(DEVICE)
    input_ids = enc["input_ids"]
    seq_len = input_ids.shape[1]

    try:
        with torch.no_grad():
            model(
                input_ids=input_ids,
                output_attentions=True,
                use_cache=False,
                return_dict=True,
            )
        full_logits = captured["logits"][0]               # [seq, vocab]
        full_attentions = list(captured["attentions"])    # list of [1, H, S, S]
    finally:
        remove_hooks(handles)

    rows = []
    prev_probs = None
    prev_H = None
    for t in range(seq_len):
        logits_t = full_logits[t]
        tok_sig = compute_token_signals(logits_t, prev_probs)

        # Slice each layer's attention down to the [0..t] prefix. The
        # attention helper reads attn[0, :, -1, :], so the last query
        # row is now position t — exactly what a prefix forward would
        # have produced.
        attn_slices = [a[:, :, :t + 1, :t + 1] for a in full_attentions]
        attn_sig = compute_attention_signals(attn_slices)

        delta_H = float("nan") if prev_H is None else tok_sig["H"] - prev_H
        rows.append(_row(label, t, tok_sig, attn_sig, delta_H))

        prev_probs = tok_sig["probs"]
        prev_H = tok_sig["H"]

    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Plotting — 2x3 panels (rows = passage, cols = run type)
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

PANEL_GRID = [
    [("coherent_baseline",    "Coherent — baseline"),
     ("coherent_intervened",  "Coherent — intervened"),
     ("coherent_detection",   "Coherent — detection")],
    [("distorted_baseline",   "Distorted — baseline"),
     ("distorted_intervened", "Distorted — intervened"),
     ("distorted_detection",  "Distorted — detection")],
]


def _annotate_intervention(ax, df):
    """Shade intervention-active steps and draw vertical lines on fires.
    Only meaningful for the intervened columns; baseline/detection rows
    will have no annotations because both flag columns are False.
    """
    if "intervention_active" in df.columns:
        active = df.loc[df["intervention_active"], "step"].tolist()
        for s in active:
            ax.axvspan(s - 0.4, s + 0.4, color="orange",
                       alpha=0.15, linewidth=0)
    if "trigger_fired" in df.columns:
        for s in df.loc[df["trigger_fired"], "step"].tolist():
            ax.axvline(s, color="red", linestyle="--",
                       alpha=0.6, linewidth=1)


def _plot_2x3(run_dfs, plot_calls, title, filename, out_dir):
    fig, axes = plt.subplots(2, 3, figsize=(18, 8))
    for row_idx, row in enumerate(PANEL_GRID):
        for col_idx, (run_key, panel_title) in enumerate(row):
            ax = axes[row_idx, col_idx]
            df = run_dfs[run_key]
            # Annotations only land on the intervened column; the other
            # columns will have all-False intervention flags so calling
            # this here is a no-op for them.
            _annotate_intervention(ax, df)
            for col, lbl in plot_calls:
                ax.plot(df["step"], df[col],
                        marker="o", markersize=3, label=lbl)
            ax.set_title(panel_title)
            ax.set_xlabel("step")
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
        _plot_2x3(run_dfs, [(col, col)], title, f"signal_{col}", out_dir)
    _plot_2x3(run_dfs, TOPK_SIGNAL["columns"],
              TOPK_SIGNAL["title"], TOPK_SIGNAL["filename"], out_dir)


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------

def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load the *instruct* model. Same architecture as the base model, but
    # bfloat16 (not phase2's float16) to match the dtype the weights were
    # trained in. Phase 2's signal/hook helpers don't care about dtype.
    model, tokenizer = load_instruct_model(MODEL_NAME)

    run_dfs = {}
    generations = {}   # only runs 1-4 (continuations) feed the scoring bundle
    all_injections = []

    # Run 1: coherent_baseline ---------------------------------------------
    df, gen = run_baseline_continuation(model, tokenizer,
                                        COHERENT_TEXT, "coherent_baseline")
    run_dfs["coherent_baseline"] = df
    generations["coherent_baseline"] = (COHERENT_TEXT, gen)

    # Run 2: coherent_intervened -------------------------------------------
    df, inj, gen = run_intervened_continuation(model, tokenizer,
                                               COHERENT_TEXT,
                                               "coherent_intervened")
    run_dfs["coherent_intervened"] = df
    generations["coherent_intervened"] = (COHERENT_TEXT, gen)
    all_injections.extend(inj)

    # Run 3: distorted_baseline --------------------------------------------
    df, gen = run_baseline_continuation(model, tokenizer,
                                        DISTORTED_TEXT, "distorted_baseline")
    run_dfs["distorted_baseline"] = df
    generations["distorted_baseline"] = (DISTORTED_TEXT, gen)

    # Run 4: distorted_intervened ------------------------------------------
    df, inj, gen = run_intervened_continuation(model, tokenizer,
                                               DISTORTED_TEXT,
                                               "distorted_intervened")
    run_dfs["distorted_intervened"] = df
    generations["distorted_intervened"] = (DISTORTED_TEXT, gen)
    all_injections.extend(inj)

    # Run 5: coherent_detection --------------------------------------------
    run_dfs["coherent_detection"] = run_detection(
        model, tokenizer, COHERENT_TEXT, "coherent_detection")

    # Run 6: distorted_detection -------------------------------------------
    run_dfs["distorted_detection"] = run_detection(
        model, tokenizer, DISTORTED_TEXT, "distorted_detection")

    # Combined CSV (all six runs, each tagged by `label`).
    combined = pd.concat(run_dfs.values(), ignore_index=True)
    combined.to_csv(CSV_PATH, index=False)
    print(f"[save] wrote raw signals to {CSV_PATH}")

    # 2x3 panel plots, one per signal.
    generate_plots(run_dfs, PLOTS_DIR)

    # Scoring bundle: ONLY the four continuation runs. The detection runs
    # produce no generated text, so they're excluded by construction.
    write_scoring_bundle(generations, SCORING_PATH)

    # Intervention log.
    log_payload = {
        "model": MODEL_NAME,
        "mechanism": "text_injection",
        "trigger_amplitude": TRIGGER_AMPLITUDE,
        "injection_text": INJECTION_TEXT,
        "system_prompt": SYSTEM_PROMPT,
        "injections": all_injections,
        "total_injections": len(all_injections),
    }
    TRIGGER_LOG_PATH.write_text(
        json.dumps(log_payload, indent=2), encoding="utf-8"
    )
    print(f"[save] wrote intervention log to {TRIGGER_LOG_PATH}")

    print("[done] PRISM Phase 3 instruct run complete.")


if __name__ == "__main__":
    main()
