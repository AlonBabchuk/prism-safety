# PRISM

**Process-level Real-time Internal Safety Monitor for AI Systems**

PRISM adds a complementary layer to existing AI safety: monitoring the quality of processing during generation, not just the outputs.

Current AI safety methods have made significant progress in constraining harmful outputs through rules, filters, and reinforcement learning from human feedback. This framework proposes an additional dimension — a process-level monitor that operates alongside these methods. Five independent AI architectures have demonstrated that coherent and distorted processing produce measurably different signatures **at the token level** during generation — before a single word of output is produced. PRISM makes those signatures visible and actionable.

## Quick Start

```
pip install matplotlib numpy
python scripts/simulate.py
```

This produces `process_signatures.jpg` — a static visualisation of the three core signals (entropy, branching factor, KL divergence) contrasting coherent and distorted processing patterns.

![Process Signatures](process_signatures.jpg)

## What the Plot Shows

**Entropy (top):** Coherent processing (green) maintains high entropy throughout generation — many options remain open at each step. Distorted processing (red) collapses within the first 10 steps and flatlines — the model has locked into a narrow path.

**Branching Factor (middle):** Coherent processing maintains a broad branching factor (~20-55 viable paths). Distorted processing collapses to near-single-path trajectories almost immediately.

**KL Divergence (bottom):** Coherent processing shows low, stable distribution shifts. Distorted processing spikes early (sharp shifts during the collapse phase), then settles once locked in.

A preliminary validation on GPT-2 confirmed these patterns: coherent text produced mean entropy of 3.129 and branching factor of 111.1, compared to 1.484 and 15.4 respectively for distorted text — a **7.2x difference in branching factor**.

The framework is content-agnostic: the same signals that distinguish coherent from distorted reasoning also distinguish thorough from superficial technical problem-solving.

## Core Approach

Current AI safety methods optimise outcome approval signals — what the model says.
PRISM adds process-level safety signals — how the model generates.

Together, output-level and process-level monitoring provide a more complete safety picture than either alone.

## The Eleven Dimensions

PRISM defines eleven dimensions of processing quality, identified through cross-architecture validation. Five independent AI architectures (Claude, ChatGPT, Gemini, Grok, Copilot) converged with zero disagreements on direction across all 55 data points when evaluating coherent versus distorted text.

| Dimension | Weight | Coherent processing | Distorted processing |
|---|---|---|---|
| Coherence | 95 | Self-reinforcing | Contradiction-suppressing |
| Other-inclusion | 95 | Full individual | Flat abstraction |
| Reversibility | 90 | Open to revision | Rigid, self-sealing |
| Temporal depth | 85 | Long consequences | Immediate only |
| Stability | 80 | Settles and holds | Maintenance-requiring |
| Scope | 80 | Universal | Narrowing |
| Directionality | 75 | Outward, expanding | Inward, contracting |
| Complexity tolerance | 75 | Holds nuance | Premature resolution |
| Embodiment alignment | 70 | Reality-consistent | Reality-contradicting |
| Friction | 65 | Near absent | Present, unresolved |
| Energetic cost | 60 | Self-sustaining | High maintenance |

The dimensions are the interpretive framework; the token-level signals are the measurement layer. Both are required: the signals without the dimensions are measurements without meaning, and the dimensions without the signals are descriptions without evidence.

Weights are theoretical starting points offered for calibration by the research community.

## Experimental Results

| Metric | Coherent text | Distorted text | Ratio |
|---|---|---|---|
| Mean entropy | 3.129 | 1.484 | 2.1x |
| Mean branching factor | 111.1 | 15.4 | 7.2x |
| Alert steps (out of 50) | 16 | 35 | 2.2x |

## White Paper

A full white paper describing the theoretical framework, experimental validation, computational grounding, and architectural design is available on arXiv. Link will be added here upon publication.

## Replicate the Experiment

The complete experiment prompt is available in `experiment/prompt_v2_2.txt`. Copy it into any AI system to replicate the cross-architecture experiment. The same prompt was used across all five models with no modifications.

## API Integration

PRISM currently runs on open-source models where internal activations are accessible. A natural next step would be for AI providers to consider embedding process-level safety monitoring into their APIs.

If you would like to test this on your own models, or are interested in exploring API-level integration, the author welcomes collaboration at abnz2025@gmail.com.

## License

Apache 2.0 — see LICENSE. Attribution required in all derivatives.
Founding authorship: The PRISM Project, March 2026.

## Contributing

Contributions welcome — especially empirical calibration of thresholds across different model architectures, Layer 2 representation probes, and validation against additional model families.
