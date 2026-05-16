"""
PRISM Phase 2 — multi-architecture replication.

Runs the identical Phase 2 detection protocol from run_phase2_llama.py on
three additional architectures from three different organisations:

    1. mistralai/Mistral-7B-v0.1   (Mistral AI)
    2. Qwen/Qwen2.5-7B             (Alibaba)
    3. google/gemma-2-9b           (Google)

Per-model outputs mirror the layout used for Llama:
    phase2/results/{slug}/signals_raw.csv
    phase2/results/{slug}/plots/signal_{H,B,D,delta_H,H_attn,S,C_topk}.png
    phase2/scripts/{slug}_thresholds.json

After all three models complete, a summary table is printed so the
distorted/coherent alert rates and the best-performing thresholds can
be compared at a glance across architectures.
"""

import gc
import json
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Same directory as run_phase2_llama, so a bare-name import works once we
# add this directory to sys.path. (Running this file as __main__ already
# puts SCRIPT_DIR at sys.path[0], but we splice explicitly so the script
# also works if invoked from elsewhere.)
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from run_phase2_llama import (                            # noqa: E402
    COHERENT_TEXT,
    DISTORTED_TEXT,
    attach_hooks,                  # noqa: F401  (re-exported per spec)
    remove_hooks,                  # noqa: F401
    compute_token_signals,         # noqa: F401
    compute_attention_signals,     # noqa: F401
    run_generation,
    run_calibration_sweep,
    generate_plots,
    N_GENERATION_STEPS,
)


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

PHASE2_DIR = SCRIPT_DIR.parent                            # phase2
RESULTS_ROOT = PHASE2_DIR / "results"

# (huggingface_id, output_slug). Order matters — runs proceed in this order
# so a Mistral OOM doesn't block Qwen/Gemma from completing.
MODELS = [
    ("mistralai/Mistral-7B-v0.1", "mistral-7b"),
    ("Qwen/Qwen2.5-7B",            "qwen-2.5-7b"),
    ("google/gemma-2-9b",          "gemma-2-9b"),
]


# --------------------------------------------------------------------------
# Model loading
# --------------------------------------------------------------------------

def load_model_for_arch(model_name: str):
    """Load tokenizer + causal LM with arch-appropriate precision.

    - Gemma 2 9B was trained in bfloat16; Google recommends loading it
      in bfloat16 too. fp16 risks silent overflow on Gemma's logits.
    - Mistral and Qwen use float16 to match the phase 2 base setting.

    Eager attention is forced everywhere so the forward hooks see real
    attention weights, matching run_phase2_llama.
    """
    if "gemma" in model_name.lower():
        dtype = torch.bfloat16
    else:
        dtype = torch.float16

    print(f"[load] {model_name} ({dtype}) ...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map="cuda",
        attn_implementation="eager",
    )
    model.eval()
    return model, tokenizer


# --------------------------------------------------------------------------
# Per-model run
# --------------------------------------------------------------------------

def run_phase2_for_model(model_name: str, slug: str):
    """Execute the full Phase 2 protocol on one model.

    Mirrors run_phase2_llama.main(): two 50-token greedy generations
    (coherent + distorted), a threshold sweep, and the seven side-by-side
    signal plots. All artefacts are written under per-model paths so they
    never clobber each other or the Llama run.

    Returns the `best` thresholds dict so the caller can build the
    cross-model summary table.
    """
    model_results_dir = RESULTS_ROOT / slug
    plots_dir = model_results_dir / "plots"
    csv_path = model_results_dir / "signals_raw.csv"
    thresholds_path = SCRIPT_DIR / f"{slug}_thresholds.json"

    model_results_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    model, tokenizer = load_model_for_arch(model_name)

    try:
        df_coherent = run_generation(model, tokenizer, COHERENT_TEXT,
                                     N_GENERATION_STEPS, label="coherent")
        df_distorted = run_generation(model, tokenizer, DISTORTED_TEXT,
                                      N_GENERATION_STEPS, label="distorted")

        # Persist raw per-step signals (both runs concatenated).
        import pandas as pd
        combined = pd.concat([df_coherent, df_distorted], ignore_index=True)
        combined.to_csv(csv_path, index=False)
        print(f"[save] {slug} -> {csv_path}")

        # Threshold sweep + persist best combo.
        best = run_calibration_sweep(df_coherent, df_distorted)
        thresholds_path.write_text(json.dumps(best, indent=2), encoding="utf-8")
        print(f"[save] {slug} -> {thresholds_path}")

        # Side-by-side trajectory plots for each of the 7 PRISM signals.
        generate_plots(df_coherent, df_distorted, plots_dir)
    finally:
        # Free GPU memory before the next model loads. Phase 2's eager
        # attention + use_cache=False footprint is significant; without
        # this, sequential 7B/9B loads risk OOM on a single GPU.
        del model, tokenizer
        gc.collect()
        torch.cuda.empty_cache()
        print(f"[gc] released {slug}")

    return best


# --------------------------------------------------------------------------
# Cross-model summary
# --------------------------------------------------------------------------

def print_summary(rows):
    """Pretty-print a fixed-width table across all model results.

    `rows` is a list of (model_name, slug, best_dict). The summary makes
    it easy to eyeball whether the distorted/coherent separation that
    holds on Llama generalises to other architectures.
    """
    header = (
        f"{'model':<28} {'distorted':>10} {'coherent':>10} "
        f"{'ratio':>8}   thresholds (H<, B<, D>)"
    )
    sep = "-" * len(header)
    print()
    print(sep)
    print(header)
    print(sep)
    for model_name, slug, best in rows:
        thresholds = (
            f"H<{best['entropy_threshold']:<4} "
            f"B<{best['branching_threshold']:<3} "
            f"D>{best['kl_threshold']}"
        )
        print(
            f"{slug:<28} "
            f"{best['distorted_alert_rate']:>10.3f} "
            f"{best['coherent_alert_rate']:>10.3f} "
            f"{best['ratio']:>8.3f}   {thresholds}"
        )
    print(sep)


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------

def main():
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    summary_rows = []
    for model_name, slug in MODELS:
        print(f"\n========== {model_name} ==========")
        best = run_phase2_for_model(model_name, slug)
        summary_rows.append((model_name, slug, best))

    print_summary(summary_rows)
    print("[done] PRISM Phase 2 multi-architecture run complete.")


if __name__ == "__main__":
    main()
