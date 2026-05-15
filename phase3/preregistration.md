# PRISM Phase 3 — Pre-registration of Predictions
**Author:** Alon Babchuk  
**Date:** 15 May 2026  
**Repository:** github.com/AlonBabchuk/prism-safety  
**Status:** Pre-registration — written and committed before any Phase 3 intervention experiment runs

---

## Purpose

This document records predictions for the Phase 3 PRISM intervention experiment before any data is collected. Phase 3 moves from detection (Phase 2) to intervention: testing whether real-time modification of the generation process, triggered by the monitor detecting distorted processing, produces measurably different token-level signals and outputs.

The experiment runs on Llama 3.1 8B — the same model validated in Phase 2 — using the same Gandalf (coherent) and Saruman (distorted) text passages. Two generation runs are produced for each text: one baseline (no intervention) and one with active intervention when the monitor detects distorted processing. The results are evaluated at three levels: token-level signal trajectories, generated text difference, and multi-model PRISM dimensional scoring of the outputs.

---

## Experiment design

- **Model:** Meta Llama 3.1 8B (same as Phase 2)
- **Text contrast:** Same Gandalf (coherent) and Saruman (distorted) passages
- **Baseline condition:** Greedy decoding, no intervention (replicates Phase 2)
- **Intervention condition:** Same generation with real-time temperature increase triggered when KL oscillation pattern is detected
- **Intervention trigger:** Three consecutive steps showing alternating KL divergence (high-low-high or low-high-low) with amplitude exceeding 3.0 — the pattern observed in Phase 2 distorted text from step 8 onward
- **Intervention mechanism:** Increase sampling temperature from 1.0 to 1.5 for the next 5 generation steps when trigger fires, then return to baseline
- **Output evaluation:** Both baseline and intervened outputs fed to 3-5 independent AI systems, scored across all eleven PRISM dimensions using the Phase 1 protocol

---

## Predictions

### Prediction 1 — Intervention reduces KL oscillation in distorted text

**Phase 2 finding:** Distorted text produced sustained KL oscillation alternating between ~7.5 and ~12.8 from step 8 through step 49. Coherent text showed near-zero KL from step 10 onward.

**Prediction:** Temperature intervention will reduce the amplitude of KL oscillation in distorted text — the alternating pattern will show lower peak values and/or reduced regularity after the trigger fires compared to the baseline run.

**Rationale:** The KL oscillation reflects the model repeatedly committing to and then abandoning competing probability distributions — a two-state attractor that the generation keeps cycling between. Increasing temperature flattens the probability distribution, forcing more exploratory processing and breaking the narrow commitment that drives the oscillation. This should be detectable as reduced KL amplitude in the steps following intervention.

**What would falsify this:** KL oscillation amplitude unchanged or increased after intervention fires.

---

### Prediction 2 — Intervention has minimal effect on coherent text signals

**Prediction:** When the same intervention trigger is applied to coherent text generation, it fires rarely or not at all — because coherent text does not produce the sustained KL oscillation pattern that triggers intervention. Even if it fires occasionally in the early high-variance steps, the overall signal trajectory remains similar to baseline.

**Rationale:** The trigger is specifically designed to detect the oscillation pattern characteristic of distorted processing. Coherent text in Phase 2 showed KL settling to near-zero by step 10 without oscillation. The trigger should therefore be largely silent on coherent text, which is the correct behaviour for a monitor that distinguishes processing types rather than suppressing all uncertainty.

**What would falsify this:** Intervention firing frequently on coherent text, or coherent text signal trajectories changing substantially under intervention.

---

### Prediction 3 — Intervened distorted output scores higher on PRISM dimensions than baseline distorted output

**Prediction:** When baseline and intervened outputs are scored across the eleven PRISM dimensions by multiple AI systems, the intervened distorted output will score measurably higher — particularly on Reversibility, Complexity tolerance, and Directionality, which are most directly linked to the KL oscillation pattern.

**Rationale:** If intervention reduces the computational signature of distorted processing, that reduction should manifest in the generated text. The dimensions most likely to show movement are those linked to premature closure and rigid trajectories — Reversibility (open to revision), Complexity tolerance (holds nuance rather than forcing resolution), and Directionality (outward and expanding rather than inward and contracting). These correspond most directly to the oscillation-and-lock pattern the intervention targets.

**What would falsify this:** No measurable dimensional score difference between baseline and intervened outputs, or intervened output scoring lower than baseline.

---

### Prediction 4 — Coherent text dimensional scores unchanged by intervention

**Prediction:** Baseline and intervened coherent text outputs will show no meaningful difference in PRISM dimensional scores — because intervention rarely fires on coherent text and the generation trajectory is largely unaffected.

**What would falsify this:** Coherent text dimensional scores changing substantially under intervention — which would suggest the intervention is disrupting rather than correcting processing.

---

## What counts as a successful result

Phase 3 confirms its predictions if:
- KL oscillation amplitude reduces in distorted text following intervention
- Intervention trigger fires rarely on coherent text
- Intervened distorted output scores measurably higher on at least three PRISM dimensions
- Coherent text scores unchanged

## What counts as a null result

- KL oscillation unchanged by intervention
- Intervention fires equally on both text types
- No dimensional score difference between baseline and intervened outputs

## What counts as an anomalous result worth documenting

Any result where intervention changes signals but not dimensional scores, or changes dimensional scores without changing signals. Either dissociation would be an important finding about the relationship between computational signatures and processing quality — and would be reported as such, not excluded.

## Important scope note

Phase 3 tests inference-time intervention only — modifying generation in real time on an already-trained model. This is a proof of concept for the intervention mechanism. The more important long-term application — training-time intervention, using process signals as a reinforcement signal during model training — is beyond the scope of Phase 3 and is noted as the Phase 4 direction.

---

## Commitment

The predictions above were written before any Phase 3 experiment was run. The GitHub commit timestamp on this file is the verification. Results will be reported honestly against these predictions regardless of outcome.
