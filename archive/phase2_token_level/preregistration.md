# PRISM Phase 2 — Pre-registration of Predictions
**Author:** Alon Babchuk  
**Date:** 14 May 2026  
**Repository:** github.com/AlonBabchuk/prism-safety  
**Status:** Pre-registration — written and committed before any Llama 3.1 8B experiment runs

---

## Purpose

This document records predictions for the Phase 2 PRISM validation experiment before any data is collected. The experiment will implement Layer 1 token-level signal extraction (entropy, branching factor, KL divergence, entropy gradient, top-k mass concentration, attention entropy, and attention span) on Llama 3.1 8B via PyTorch forward hooks, using the same coherent/distorted text contrast validated on GPT-2 in Phase 1.

Committing this document before running the experiment establishes that the predictions below reflect genuine prior expectations, not post-hoc rationalisation.

---

## Experiment design

- **Model:** Meta Llama 3.1 8B (HuggingFace: meta-llama/Meta-Llama-3.1-8B)
- **Text contrast:** Same Gandalf (coherent) and Saruman (distorted) passages used in Phase 1
- **Signals measured:** All seven Layer 1 signals from Section 5 of the white paper
- **Primary comparison:** Branching factor contrast between coherent and distorted text
- **Secondary comparisons:** Attention entropy and attention span differentiation; threshold calibration behaviour

---

## Predictions

### Prediction 1 — Branching factor contrast exceeds the GPT-2 result

**GPT-2 result:** 7.2x branching factor difference between coherent and distorted text (111.1 vs 15.4).

**Prediction:** The branching factor contrast on Llama 3.1 8B will exceed 7.2x.

**Rationale:** A more capable model processes coherent text with genuinely greater exploratory breadth — more viable continuations held open simultaneously — while distorted text produces earlier and more pronounced entropy collapse. The 70x scale increase from GPT-2 to Llama 3.1 8B should amplify rather than diminish this contrast, as the model has greater capacity to maintain high-entropy coherent processing and greater sensitivity to the structural closure of distorted text.

**What would falsify this:** A branching factor contrast below 7.2x, or no statistically meaningful difference between the two texts.

---

### Prediction 2 — Attention signals show partial emergence

**GPT-2 result:** Attention entropy and attention span showed minimal differentiation at GPT-2 scale. The white paper explicitly predicted these signals would emerge more clearly at larger model scale.

**Prediction:** Attention entropy and attention span will show clearer differentiation on Llama 3.1 8B than on GPT-2, but may not produce clean mappings to PRISM dimensions without Layer 2 representation probes.

**Rationale:** Scale increases the depth and richness of attention patterns. A larger model attending to coherent text should draw on wider context (higher attention span) and distribute attention more broadly across tokens (higher attention entropy) than when processing distorted text, which collapses to narrow, recent context. However, clean dimensional mapping likely requires the Layer 2 probes described in the white paper — token-level attention signals alone may show the directional pattern without reaching the clarity needed for threshold-based monitoring.

**What would falsify this:** Attention signals showing no differentiation whatsoever (same as GPT-2), or conversely, producing immediate clean dimensional mapping without Layer 2 probes.

---

### Prediction 3 — Threshold calibration still required, but raw signals more interpretable

**GPT-2 result:** Default thresholds produced no differentiation (49/50 alert steps on both texts). Model-specific calibration was required to achieve a 2.2x alert ratio.

**Prediction:** Llama 3.1 8B will still require model-specific threshold calibration, but the raw uncalibrated signals will show more visible separation between coherent and distorted text than GPT-2 did before calibration.

**Rationale:** Threshold calibration is a structural requirement of any monitoring system applied across different model architectures — each model has a different baseline entropy range and distribution. This will not disappear at larger scale. However, a more capable model should produce more interpretable raw signals, meaning the coherent/distorted contrast will be visible in the distributions even before thresholds are tuned, reducing the calibration burden compared to GPT-2.

**What would falsify this:** Raw signals on Llama 3.1 8B showing no visible separation before calibration (same as GPT-2), or the model requiring no calibration at all.

---

## What counts as a successful result

The experiment confirms Phase 2 predictions if:
- Branching factor contrast exceeds 7.2x
- Attention signals show directional differentiation (even if not yet clean enough for direct dimensional mapping)
- Raw signals show visible separation before calibration, even if thresholds still require tuning

## What counts as a null result

- Branching factor contrast below 7.2x
- No attention signal differentiation
- Raw signals requiring the same degree of calibration as GPT-2 with no visible pre-calibration separation

## What counts as an anomalous result worth documenting

Any result that does not fit cleanly into confirmation or falsification — including unexpected signal behaviour, reversed patterns on specific dimensions, or signals that differentiate strongly on some dimensions but not others. Per the PRISM project principle: unexpected results are findings, not failures. Anomalies will be documented and reported in the Phase 2 companion paper, not excluded.

---

## Commitment

The predictions above were written before any Llama 3.1 8B experiment was run. The GitHub commit timestamp on this file is the verification. Results will be reported honestly against these predictions regardless of outcome.
