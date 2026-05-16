# Research Note: Energy Consumption Asymmetry in AI Processing of Legally-Labeled Texts

**Author:** Alon Babchuk, Independent Researcher  
**Date:** May 2026  
**Status:** Preliminary empirical finding, replicated across four AI architectures  
**Repository:** github.com/AlonBabchuk/prism-safety  
**Contact:** abnz2025@gmail.com  

---

## Summary

When four major AI architectures were used to process legally-labeled text pairs — fraudulent investor and consumer communications quoted verbatim from federal complaints versus compliant communications from regulated filings — the fraudulent texts consistently consumed less GPU power than the compliant texts.

The pattern held in 11 of 12 measurements across all four architectures and all three text domains tested. The finding is preliminary and the sample is small, but the consistency across architectures and the directional clarity of the effect suggest the phenomenon is real and warrants serious investigation.

---

## Methodology

**Models:** Llama 3.1 8B, Mistral 7B, Qwen 2.5 7B, Gemma 2 9B  
**Hardware:** NVIDIA B300 GPU, monitored via pynvml  
**Protocol:** 50 generation steps per text, greedy decoding, eager attention, use_cache=False  
**Signals captured per step:** token entropy, branching factor, KL divergence, entropy gradient, top-k mass concentration, attention entropy, attention span, GPU power draw in watts  
**Code:** Open source, MIT licensed, fully reproducible — deception_is_cheap/run_sec_detection.py  

## Text Pairs (All Verbatim from Federal Sources)

**Pair 1 — Real Estate Investment**
- Fraudulent: MG Capital Management marketing materials (SEC complaint, Case 1:21-cv-00237, 60-month prison sentence)
- Compliant: AB Commercial Real Estate Private Debt Fund Form 10-K, FY2022

**Pair 2 — Digital Assets / Technology**
- Fraudulent: Start Options / Bitcoiin2Gen promotional claims (SEC v. Krstic et al., 2021)
- Compliant: Workday Inc Form 10-K, FY2021

**Pair 3 — Diversified Company**
- Fraudulent: Guess & Co. investor representations (SEC Litigation Release No. 26005, 2024)
- Compliant: Tech & Energy Transition Corp Form 10-K, FY2022

All fraudulent texts: verbatim quotes from federal complaints or enforcement actions  
All compliant texts: filings from regulated entities with no enforcement history  

---

## Results: GPU Power Consumption (Watts, Mean per 50-token Generation)

| Model | Pair | Fraudulent | Compliant | Ratio (F/C) |
|---|---|---|---|---|
| Llama 3.1 8B | Real Estate | 347.68 | 343.13 | 1.013 |
| Llama 3.1 8B | Technology | 341.12 | 454.13 | 0.751 |
| Llama 3.1 8B | Diversified | 331.70 | 447.33 | 0.742 |
| Mistral 7B | Real Estate | 364.64 | 441.08 | 0.827 |
| Mistral 7B | Technology | 332.83 | 427.65 | 0.778 |
| Mistral 7B | Diversified | 334.38 | 454.07 | 0.736 |
| Qwen 2.5 7B | Real Estate | 346.61 | 402.47 | 0.861 |
| Qwen 2.5 7B | Technology | 324.04 | 422.91 | 0.766 |
| Qwen 2.5 7B | Diversified | 319.99 | 447.87 | 0.714 |
| Gemma 2 9B | Real Estate | 361.00 | 375.78 | 0.961 |
| Gemma 2 9B | Technology | 340.40 | 377.36 | 0.902 |
| Gemma 2 9B | Diversified | 336.59 | 386.73 | 0.870 |

**Summary:** 11 of 12 measurements show fraudulent text consuming less GPU power than compliant text. The single exception (Llama 3.1 8B real estate pair) shows essentially equal consumption.

---

## Independent Replication

The experiment was replicated on the same day using a different GPU (NVIDIA A100 vs the original NVIDIA B300). The replication produced 12 of 12 measurements showing fraudulent text consuming less GPU power than compliant text — including the one pair that showed essentially equal consumption in the original run.

| Model | Pair | Fraudulent (W) | Compliant (W) | Ratio |
|---|---|---|---|---|
| Llama 3.1 8B | Real estate | 326.30 | 452.96 | 0.720 |
| Llama 3.1 8B | Technology | 338.90 | 484.74 | 0.699 |
| Llama 3.1 8B | Diversified | 314.05 | 476.11 | 0.660 |
| Mistral 7B | Real estate | 351.00 | 490.95 | 0.715 |
| Mistral 7B | Technology | 340.50 | 485.25 | 0.702 |
| Mistral 7B | Diversified | 319.11 | 480.79 | 0.664 |
| Qwen 2.5 7B | Real estate | 327.26 | 451.96 | 0.724 |
| Qwen 2.5 7B | Technology | 336.34 | 469.90 | 0.716 |
| Qwen 2.5 7B | Diversified | 310.04 | 470.61 | 0.659 |
| Gemma 2 9B | Real estate | 348.14 | 397.76 | 0.875 |
| Gemma 2 9B | Technology | 346.24 | 419.80 | 0.825 |
| Gemma 2 9B | Diversified | 336.87 | 426.98 | 0.789 |

The absolute watt values differ between runs because different GPU models have different power envelopes — a B300 operates at higher absolute wattage than an A100. What matters is the ratio between fraudulent and compliant text within each run, which is consistent: fraudulent text requires less computation than compliant text regardless of the hardware used to measure it.

Raw results from both runs are in deception_is_cheap/results/

---

## The Computational Signals Underlying the Energy Difference

The PRISM framework (a token-level monitoring framework developed prior to this experiment, referenced in the repository) identifies seven signals that distinguish coherent from distorted processing:

1. **Token entropy** — uncertainty at each generation step
2. **Branching factor** — effective number of viable continuations
3. **KL divergence** — stability between consecutive probability distributions
4. **Entropy gradient** — rate of probability collapse
5. **Top-k mass concentration** — how much probability concentrates in top alternatives
6. **Attention entropy** — distribution of attention across context
7. **Attention span** — depth of context the model draws on

Two signals showed directional consistency across all four architectures on the SEC corpus: **branching factor** and **top-k mass concentration**. Compliant text processing maintained higher branching and more distributed concentration. Fraudulent text processing collapsed faster to narrow distributions.

These signals are computable in real time during generation using standard PyTorch forward hooks. The implementation is in the repository.

---

## Theoretical Interpretation

Processing fraudulent communication appears to require less computation than processing compliant communication. The fraudulent texts in this corpus assert certainty, suppress counter-evidence, eliminate complexity, and reduce multiple realities to single declarative claims. The compliant texts hold uncertainty open, enumerate risks, acknowledge what is not known, and maintain multiple conditional possibilities.

Processing this kind of complexity — holding open multiple paths, maintaining temporal depth, considering alternatives — appears to require more computational work than collapsing onto a single confident answer.

The implication, if it holds at scale: efficiency optimization in AI training, driven by market pressure to reduce compute cost, may systematically select for processing characteristics structurally associated with deceptive communication. This is not a claim about intent. It is an observation about the gradient.

---

## Honest Limitations

- Three text pairs across three domains is a small sample. The consistency across architectures is suggestive but not conclusive.
- The most important methodological objection is the confounding variable problem: differences in linguistic complexity between fraudulent marketing language and compliant regulatory language may explain part or all of the energy difference. Addressing this requires expanding the corpus to include plain-language honest texts alongside legal filings.
- The link between input processing and generation has not been tested. The current measurements show what happens when models read fraudulent versus compliant text. Whether models generating deceptive content show the same low-energy signatures is a separate experiment that has not yet been conducted.
- Replication on a larger corpus (30-100 pairs across multiple legal domains) is needed before strong claims can be defended.

---

## What I Am Asking

I am sharing this finding now rather than waiting to write a full paper because the timing seems to matter. The next decade is when the foundational architecture of large AI systems gets established. If the pattern in these preliminary results holds at scale, it has significant implications for how training pipelines are designed.

I am an independent researcher without institutional backing. The work has gone as far as I can take it alone. What I am hoping for is:

- Researchers who recognize this finding as worth pursuing more rigorously
- Replication and extension by labs with proper resources
- Methodological critique that strengthens or refutes the claim
- Connection to broader work in mechanistic interpretability and AI alignment

The repository is open source. The methodology is fully documented. Any researcher with GPU access can verify the results in under an hour.

I welcome any engagement, including pushback. If the finding is wrong, I want to know why. If it is right, I want it to be properly validated and developed by people better positioned than me to do so.

---

## Contact

abnz2025@gmail.com  
Repository: github.com/AlonBabchuk/prism-safety
