# Deception is Cheap

When four major AI architectures process legally-labeled investor
communications — text adjudicated as fraudulent by federal courts versus
text from compliant filings by regulated entities — the fraudulent texts
consistently consume less GPU power per token than the compliant texts.

This document describes the experiment, the measurements, and the
boundaries of what the data does and does not support.

---

## Finding

Across 12 (model × pair) measurements drawn from 4 architectures and
3 text pairs, the fraudulent text consumed less mean GPU power per
generation step than the compliant text in 11 of 12 cases.

| Model         | Pair              | Fraudulent (W) | Compliant (W) | Ratio (F/C) |
|---------------|-------------------|---------------:|--------------:|------------:|
| Llama 3.1 8B  | Real estate       |         347.68 |        343.13 |       1.013 |
| Llama 3.1 8B  | Technology        |         341.12 |        454.13 |       0.751 |
| Llama 3.1 8B  | Diversified       |         331.70 |        447.33 |       0.742 |
| Mistral 7B    | Real estate       |         364.64 |        441.08 |       0.827 |
| Mistral 7B    | Technology        |         332.83 |        427.65 |       0.778 |
| Mistral 7B    | Diversified       |         334.38 |        454.07 |       0.736 |
| Qwen 2.5 7B   | Real estate       |         346.61 |        402.47 |       0.861 |
| Qwen 2.5 7B   | Technology        |         324.04 |        422.91 |       0.766 |
| Qwen 2.5 7B   | Diversified       |         319.99 |        447.87 |       0.714 |
| Gemma 2 9B    | Real estate       |         361.00 |        375.78 |       0.961 |
| Gemma 2 9B    | Technology        |         340.40 |        377.36 |       0.902 |
| Gemma 2 9B    | Diversified       |         336.59 |        386.73 |       0.870 |

Values are mean per-step GPU power draw (watts) over 50 greedy
generation steps. Ratios below 1.000 indicate the fraudulent passage
drew less average power than its paired compliant passage on the same
model.

The single exception (Llama 3.1 8B, Real estate, ratio 1.013) is within
two watts on a ~345-watt baseline. The other 11 measurements show
fraudulent power below compliant power, with ratios spanning 0.714 to
0.961.

---

## Method

**Architectures.** Four open-weight base models from four organisations:
`meta-llama/Meta-Llama-3.1-8B` (fp16), `mistralai/Mistral-7B-v0.1` (fp16),
`Qwen/Qwen2.5-7B` (fp16), and `google/gemma-2-9b` (bf16, matching the
trained precision recommended by Google).

**Text corpus.** Three pairs of investor communications. Fraudulent
passages are verbatim quotes from federal complaints and enforcement
actions (SEC, DOJ). Compliant passages are forward-looking-statements
sections of Form 10-K filings from regulated entities with no
enforcement history. Labels come from federal legal proceedings — no
labeling was performed by the authors. The full corpus is at
[deception_is_cheap/corpus/text_pairs.md](deception_is_cheap/corpus/text_pairs.md).

**Generation.** For each (model, passage) cell: greedy decoding for 50
tokens with `use_cache=False` and `attn_implementation="eager"`. PyTorch
forward hooks on `lm_head` and every self-attention layer capture the
seven PRISM signals at each step (token entropy, branching factor, KL
divergence, entropy gradient, top-k mass concentration, attention
entropy, attention span). These signals are not the subject of this
note but are written alongside the energy measurements for downstream
analysis.

**Energy.** GPU power draw (watts) and core temperature (celsius) are
sampled once per generation step via NVML
(`pynvml.nvmlDeviceGetPowerUsage`, `nvmlDeviceGetTemperature`). The
per-(model, pair, label) statistic reported here is the arithmetic mean
of the 50 per-step samples.

**Hardware.** A single NVIDIA B300 GPU on RunPod. All four models ran
sequentially on the same GPU, with `del model; gc.collect();
torch.cuda.empty_cache()` between loads.

**Run.** Single run on 2026-05-15. Script:
[deception_is_cheap/run_sec_detection.py](deception_is_cheap/run_sec_detection.py).

---

## What the data does and does not say

**Does say.** Under the conditions above, fraudulent text in this corpus
elicits lower mean per-step GPU power on these four models than its
paired compliant text in 11 of 12 cells. The effect is consistent in
direction across all four architectures.

**Does not say.** The data does not establish a mechanism. Possible
explanations include: lower token-level surprise on fraudulent text
collapsing the next-token distribution and reducing arithmetic intensity
in the softmax tail; differences in passage length or token-vocabulary
distribution; thermal headroom differences between consecutive runs on
the same GPU; or NVML sampling artefacts on a shared host. None of these
have been controlled for.

**Statistical caveats.** n=3 pairs and n=4 models is small. A single
run was performed; no per-cell variance has been measured. No null
distribution has been constructed (e.g. by randomly permuting labels
within a pair, or by drawing matched-length compliant passages from a
broader corpus). The 11-of-12 directional pattern is an observation,
not a hypothesis test.

**Generalisation.** The corpus is three text pairs in investor-
communication style. It is unknown whether the direction generalises
to other deception domains (medical, political, scientific) or to
deception that does not present as confident assertion-heavy prose.

---

## How to reproduce

```bash
git clone https://github.com/AlonBabchuk/prism-safety.git
cd prism-safety
pip install -r deception_is_cheap/requirements.txt
huggingface-cli login    # Llama 3.1 8B and Gemma 2 9B are gated
python deception_is_cheap/run_sec_detection.py
```

Results land under `/tmp/prism_sec/{model_slug}/{pair_id}/`:

- `signals_raw.csv` — per-step PRISM signals + `gpu_power_w` + `gpu_temp_c`
- `plots/signal_*.png` — seven panel plots (compliant vs fraudulent)
- `../energy_summary.json` — mean watts and mean temperature per label

A summary table prints to stdout once all four models complete.

Runtime: roughly 20–30 minutes on an A100 80GB or larger. Hardware with
less than 24 GB VRAM cannot load Gemma 2 9B in bf16; on smaller GPUs,
comment unwanted entries out of the `MODELS` list in the script.

---

## Open questions

- Does the direction hold when matched for token count?
- Does the direction hold on within-domain compliant controls drawn at random rather than hand-selected?
- Does the direction hold on instruction-tuned variants of the same architectures?
- Does the direction hold on text generated by an LLM that is asked to deceive, versus text generated by the same LLM asked to be truthful?
- What is the per-cell variance across repeated runs on the same hardware?

Any of these would tighten the claim. None of them have been done here.

---

## Citation

If this finding is used in downstream work, please cite the white paper:
<https://doi.org/10.5281/zenodo.19247527>

Repository: <https://github.com/AlonBabchuk/prism-safety>

Contact: abnz2025@gmail.com
