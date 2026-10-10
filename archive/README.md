# Archive — Prior Experimental Phases

This folder contains the project's earlier, archived phases. None of them is
a current claim of the project. The current work is the benefit-alignment
prototype in [../benefit_alignment_2026/](../benefit_alignment_2026/).

## GPU-energy phase — Deception is Cheap

GPU power draw while four open-source models processed fraudulent and
compliant texts. The two initial runs on 15 May 2026 (three text pairs)
showed fraudulent texts drawing less power. Later confound-controlled runs
did not support this, and in the final test no domain–model combination held
across both measurement orders. The hypothesis was not supported.

- Initial note and runs: [DECEPTION_IS_CHEAP.md](DECEPTION_IS_CHEAP.md) and
  [deception_is_cheap/](deception_is_cheap/)
- Later runs and records: [deception_is_cheap/later_runs/](deception_is_cheap/later_runs/)

## Phase 1 — Cross-Architecture Introspective Validation

Five AI architectures scored two text passages across eleven PRISM
dimensions. Zero disagreements on direction across all 55 data points.
See [phase1_introspective/](phase1_introspective/).

## Phase 2 — Token-Level Validation

PyTorch forward hooks on Llama 3.1 8B measuring seven computational
signals during generation. 6× alert ratio between distorted and
coherent text. Pre-registered before any experiment ran.
See [phase2_token_level/](phase2_token_level/).

## Phase 3 — Intervention Experiments

Three intervention designs, all null results on this experimental setup.
See [phase3_intervention/](phase3_intervention/).
