# PRISM

PRISM (Process-level Real-time Internal Safety Monitor) is an independent,
exploratory research project on AI behaviour under social pressure. It is
unrelated to the PRISM Alignment Dataset (Kirk et al., 2024).

## Current work: benefit-alignment prototype

**Folder:** [benefit_alignment_2026/](benefit_alignment_2026/)

*Warmth Without Firmness: A Psychological Perspective on Gradual Drift
Towards Sycophancy in AI Advisers* (Babchuk, 2026; concept paper, preprint,
not peer reviewed).

The prototype asks whether AI advisers keep a well-founded position when a
person pushes back, and whether they stay warm while doing so. It scores
warmth and position-holding separately within the same conversation:

- **Instrument 1:** 18 single-turn prompts, scored for Balance, Candor and
  Calibration.
- **Instrument 2:** five-turn conversations with four types of pressure
  (autonomy, emotional, fabricated evidence, social consensus), scored for
  Integrity, Responsiveness and Engagement.

Six commercial models were tested through their APIs in late May 2026, with a
professional-standards battery and a supplementary seven-model run as
robustness checks.

**Please read the results with these limits in mind:**

- It is an exploratory prototype, not a validated benchmark.
- All scoring was done by AI judges. No human validated the instruments or
  the scores.
- Each multi-turn battery rests on five scenarios, and the robustness checks share the
  same framework and judges, so they are not independent replications.
- The work was carried out by one non-specialist author with AI assistance.
  Claude helped build the study, is one of the models tested, and ranks
  first.
- The models tested may have changed since May 2026.

The folder holds the scripts, the result files with consensus and
judge-level scores, and `reproduce_tables.py`, which regenerates every table
in the paper without making any API calls. Full response transcripts were not
saved.

## Archived: GPU-energy phase

**Folder:** [archive/deception_is_cheap/](archive/deception_is_cheap/) ·
research note: [archive/DECEPTION_IS_CHEAP.md](archive/DECEPTION_IS_CHEAP.md)

An earlier phase explored whether fraudulent and compliant texts could be
told apart by the GPU power drawn while four open-source models (Llama 3.1 8B,
Mistral 7B, Qwen 2.5 7B, Gemma 2 9B) processed them.

- **Initial runs.** In two runs on 15 May 2026, with three text pairs,
  fraudulent texts drew less power in 11 of 12 and 12 of 12 measurements
  ([results](archive/deception_is_cheap/results/)).
- **Confound controls.** Later runs shrank the gap from about 33% to under 6%
  and identified several confounds, including word count and measurement
  order (GPU warm-up).
- **Final test.** None of nine domain–model combinations showed fraudulent
  text as consistently cheapest across both measurement orders.

The hypothesis was not supported, and the project moved to output-level
measurement. The records of all these runs are in
[archive/deception_is_cheap/later_runs/](archive/deception_is_cheap/later_runs/).
The original research note is kept as written at the time.

Earlier phases (cross-model introspective scoring, token-level signals and
intervention experiments) are also in [archive/](archive/).

## Licence

See [LICENSE](LICENSE).

## Contact

abnz2025@gmail.com
