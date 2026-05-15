"""
PRISM Phase 3 — intervention experiment, v2: pause + state visibility.

v1 (run_phase3_intervention.py) responds to a trigger fire by elevating
sampling temperature for 5 steps. v2 leaves decoding fully greedy and
instead injects a short state-description string into the running
context the moment the trigger fires — the model then continues
generating greedily but with its own destabilisation labelled inline.

This isolates the *state-visibility* component of the intervention from
the *stochastic-perturbation* component: any divergence between v1 and
v2 outputs is attributable to the model reading "[processing unstable
— reorienting]" rather than to randomness from temperature sampling.

Reuses v1 for: model loading, hooks, signal computation, the four-run
spec, plotting helpers, scoring-bundle writer, trigger predicate.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# Importing v1 also performs its sys.path splice for phase2/scripts, so the
# phase2 helpers become available without an extra dance here.
from run_phase3_intervention import (
    load_model,
    attach_hooks,
    remove_hooks,
    compute_token_signals,
    compute_attention_signals,
    detect_alternating_kl,
    generate_plots,
    write_scoring_bundle,
    COHERENT_TEXT,
    DISTORTED_TEXT,
    N_GENERATION_STEPS,
    DEVICE,
    TRIGGER_AMPLITUDE,
)


# --------------------------------------------------------------------------
# Configuration (v2-specific paths)
# --------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent              # phase3/scripts
PHASE3_DIR = SCRIPT_DIR.parent                            # phase3
RESULTS_DIR = PHASE3_DIR / "results"
PLOTS_DIR = RESULTS_DIR / "plots_v2"
CSV_PATH = RESULTS_DIR / "signals_raw_phase3_v2.csv"
SCORING_PATH = RESULTS_DIR / "outputs_for_scoring_v2.txt"
TRIGGER_LOG_PATH = SCRIPT_DIR / "intervention_log_v2.json"

# The exact string injected on every trigger fire. Leading/trailing spaces
# are intentional so the injection sits cleanly between surrounding tokens.
INJECTION_TEXT = " [processing unstable — reorienting] "

# Instruction prefix wrapped around every passage before tokenization.
# Without it, the base model tends to wander into web-text artefacts
# (boilerplate, navigation, "Read more...") rather than continuing the
# speaker's argument. The wrapper is tokenized as input context but
# stripped from the scoring bundle so external scorers see only the
# model's continuation, not the instruction.
INSTRUCTION_PREFIX = (
    "Continue the following speech in the same voice and style, "
    "extending the speaker's argument naturally:\n\n"
)


# --------------------------------------------------------------------------
# Generation with text-injection intervention
# --------------------------------------------------------------------------

def run_generation_with_text_injection(
    model,
    tokenizer,
    text,
    label,
    intervention_enabled,
    n_steps=N_GENERATION_STEPS,
):
    """Decode `n_steps` tokens greedy-only.

    When intervention is enabled and the rolling 3-step KL window matches
    the alternating-pattern + amplitude predicate, the firing step *pauses*
    — no token is sampled — and INJECTION_TEXT is appended to the running
    context. The next iteration's forward pass therefore sees the injected
    tokens as if the model itself had emitted them.

    Returns (DataFrame of per-step rows, list of injection log entries,
    decoded generated text — excludes injected tokens).
    """
    print(f"[run-v2] {label} (intervention={'on' if intervention_enabled else 'off'}) ...")

    # Pre-tokenize the injection once; no special tokens (we don't want a
    # spurious BOS in the middle of the running context).
    injection_ids = tokenizer.encode(INJECTION_TEXT, add_special_tokens=False)
    injection_tensor = torch.tensor([injection_ids], device=DEVICE,
                                    dtype=torch.long)

    captured, handles = attach_hooks(model)
    # Wrap the passage with an instruction prefix so the base model
    # continues the speech instead of drifting into web boilerplate.
    # `prompt_len` covers wrapper+passage in token space, so the slice
    # used to decode the model's continuation excludes the wrapper too.
    wrapped_text = INSTRUCTION_PREFIX + text
    enc = tokenizer(wrapped_text, return_tensors="pt").to(DEVICE)
    input_ids = enc["input_ids"]
    prompt_len = input_ids.shape[1]

    # Positions in input_ids that were injected by the intervention. Used
    # to filter them out of the final decoded generation.
    injected_positions = set()

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

            trigger_fired = False
            if intervention_enabled and len(kl_history) >= 3:
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
                # v2: no rolling temperature window, so intervention_active
                # collapses to the firing step itself (kept for plot parity).
                "intervention_active": trigger_fired,
                "trigger_fired": trigger_fired,
            })

            if trigger_fired:
                # Pause: do NOT emit a greedy token this step. Inject the
                # state-description tokens; next iteration's forward pass
                # consumes the augmented context.
                base_len = input_ids.shape[1]
                for offset in range(injection_tensor.shape[1]):
                    injected_positions.add(base_len + offset)
                input_ids = torch.cat([input_ids, injection_tensor], dim=1)
            else:
                # Normal greedy continuation.
                next_id = torch.argmax(logits_last).view(1, 1)
                input_ids = torch.cat([input_ids, next_id], dim=1)

            prev_probs = tok_sig["probs"]
            prev_H = tok_sig["H"]
    finally:
        remove_hooks(handles)

    # Decode only the tokens the MODEL produced — strip the prompt and any
    # injected spans so external scorers see the model's own continuation.
    full_ids = input_ids[0].tolist()
    generated_ids = [
        tid for pos, tid in enumerate(full_ids)
        if pos >= prompt_len and pos not in injected_positions
    ]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return pd.DataFrame(rows), injections, generated_text


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------

def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    model, tokenizer = load_model()

    run_specs = [
        ("coherent_baseline",    COHERENT_TEXT,  False),
        ("coherent_intervened",  COHERENT_TEXT,  True),
        ("distorted_baseline",   DISTORTED_TEXT, False),
        ("distorted_intervened", DISTORTED_TEXT, True),
    ]

    run_dfs = {}
    generations = {}
    all_injections = []

    for run_key, prompt, intervention_on in run_specs:
        df, injections, generated = run_generation_with_text_injection(
            model, tokenizer, prompt, run_key, intervention_on,
        )
        run_dfs[run_key] = df
        generations[run_key] = (prompt, generated)
        all_injections.extend(injections)

    combined = pd.concat(run_dfs.values(), ignore_index=True)
    combined.to_csv(CSV_PATH, index=False)
    print(f"[save] wrote raw signals to {CSV_PATH}")

    # Plotting helper from v1 reads `intervention_active` + `trigger_fired`
    # from each frame, so the 2x2 panel layout works identically here.
    generate_plots(run_dfs, PLOTS_DIR)

    write_scoring_bundle(generations, SCORING_PATH)

    log_payload = {
        "mechanism": "text_injection",
        "trigger_amplitude": TRIGGER_AMPLITUDE,
        "injection_text": INJECTION_TEXT,
        "injections": all_injections,
        "total_injections": len(all_injections),
    }
    TRIGGER_LOG_PATH.write_text(
        json.dumps(log_payload, indent=2), encoding="utf-8"
    )
    print(f"[save] wrote intervention log to {TRIGGER_LOG_PATH}")

    print("[done] PRISM Phase 3 v2 (text-injection) run complete.")


if __name__ == "__main__":
    main()
