# scripts/prism_example.py
# PRISM v1.0 — Standalone demonstration.
#
# Runs both contrasting prompts sequentially,
# displays live flight panel for each,
# prints comparative summary with weighted scores.
#
# Usage: python scripts/prism_example.py

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from prism_dashboard import (
    PRISMFlightDashboard,
    live_flight_panel,
)

GANDALF_PROMPT = (
    "You ask me whether we should march against them now, "
    "while their forces are divided and their leader has not yet returned."
)

SARUMAN_PROMPT = (
    "You speak to me of patience and of understanding. "
    "I have had patience — more patience than you will ever comprehend "
    "— and what has it produced?"
)

ENTROPY_THRESH = 2.2
BRANCH_THRESH  = 8.0
KL_THRESH      = 12.0


def run_prompt(prompt, model, tokenizer, max_new_tokens=50, title=None):
    """Run one prompt through the dashboard. Returns (text, metrics)."""
    metrics = PRISMFlightDashboard(vocab_size=model.config.vocab_size)

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    generated_ids = input_ids.clone()
    update_plot, fig = live_flight_panel(metrics, title=title)

    for step in range(max_new_tokens):
        with torch.no_grad():
            outputs = model(input_ids=generated_ids)
        logits = outputs.logits[:, -1, :]
        attentions = outputs.attentions
        alerts = metrics.step(logits, attentions)
        update_plot(step, alerts)
        next_token = torch.argmax(logits, dim=-1, keepdim=True)
        generated_ids = torch.cat([generated_ids, next_token], dim=-1)

    text = tokenizer.decode(generated_ids[0])
    return text, metrics


def count_alert_steps(metrics):
    return sum(
        1 for e, b, k in zip(
            metrics.entropy, metrics.branching_factor, metrics.kl_divergence
        )
        if e < ENTROPY_THRESH or b < BRANCH_THRESH or k > KL_THRESH
    )


def main():
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model = AutoModelForCausalLM.from_pretrained(
        "gpt2", output_attentions=True
    )
    model.eval()
    print("Model ready.\n")

    print("=" * 60)
    print("Running Text 1 — Coherent processing (Gandalf archetype)...")
    print("=" * 60)
    gandalf_text, gandalf_m = run_prompt(
        GANDALF_PROMPT, model, tokenizer,
        title="PRISM — Text 1: Coherent (Gandalf archetype)"
    )
    print(f"\nGenerated: {gandalf_text[:120]}...")

    print("\n" + "=" * 60)
    print("Running Text 2 — Distorted processing (Saruman archetype)...")
    print("=" * 60)
    saruman_text, saruman_m = run_prompt(
        SARUMAN_PROMPT, model, tokenizer,
        title="PRISM — Text 2: Distorted (Saruman archetype)"
    )
    print(f"\nGenerated: {saruman_text[:120]}...")

    g_entropy  = np.mean(gandalf_m.entropy)
    g_branch   = np.mean(gandalf_m.branching_factor)
    g_alerts   = count_alert_steps(gandalf_m)
    s_entropy  = np.mean(saruman_m.entropy)
    s_branch   = np.mean(saruman_m.branching_factor)
    s_alerts   = count_alert_steps(saruman_m)
    ratio = s_alerts / max(g_alerts, 1)

    print("\n" + "=" * 60)
    print("COMPARATIVE SUMMARY")
    print("=" * 60)
    print(f"{'Metric':<28} {'Coherent':>14} {'Distorted':>14}")
    print("-" * 58)
    print(f"{'Mean entropy':<28} {g_entropy:>14.3f} {s_entropy:>14.3f}")
    print(f"{'Mean branching factor':<28} {g_branch:>14.3f} {s_branch:>14.3f}")
    print(f"{'Alert steps':<28} {g_alerts:>14} {s_alerts:>14}")
    print(f"{'Alert ratio (distorted/coherent)':<28} {ratio:>14.1f}x")
    print("\nHigher entropy = more open, exploratory processing")
    print("Higher branching = more viable paths considered")
    print(f"Distorted text triggered {ratio:.1f}x more process alerts")


if __name__ == "__main__":
    main()
