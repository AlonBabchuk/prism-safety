# PRISM — Deception is Cheap

Open research on whether token-level computational signals can
distinguish coherent from distorted processing in large language
models — and on a surprising downstream observation: deception is
energetically cheap.

## Key Finding

When four major AI architectures process legally-labeled text pairs —
fraudulent communications from SEC, FTC, and DOJ enforcement actions
versus compliant filings from the same domains — the fraudulent texts
consistently consumed less GPU power than the compliant texts.

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

11 of 12 measurements: fraudulent text consumes less GPU power than
compliant text.

Full methodology: see [DECEPTION_IS_CHEAP.md](DECEPTION_IS_CHEAP.md)

## How to Reproduce

### What you need

- GPU with at least 24 GB VRAM (80 GB recommended for all four models)
- HuggingFace account with access approved for `meta-llama/Meta-Llama-3.1-8B` and `google/gemma-2-9b`
- Python 3.10+

### Option A — RunPod (recommended, tested)

This experiment was run on RunPod using an NVIDIA B300 GPU. Any A100
80 GB or larger will work.

**Step 1 — Create a RunPod pod:**

- Template: RunPod PyTorch 2.8.0
- GPU: A100 SXM 80 GB or larger
- Container disk: 100 GB minimum
- No persistent storage required for a single run

**Step 2 — In the pod terminal:**

```bash
git clone https://github.com/AlonBabchuk/prism-safety.git
cd prism-safety
pip install -r deception_is_cheap/requirements.txt
huggingface-cli login
python deception_is_cheap/run_sec_detection.py
```

**Step 3 — Results save to `/tmp/prism_sec/`.**
Expected runtime: 20–30 minutes on an A100. The script prints a summary
table when complete.

### Option B — Local machine

Same steps. Requires a CUDA GPU with sufficient VRAM. To run on one
model only, comment out unwanted entries in the `MODELS` list in
[deception_is_cheap/run_sec_detection.py](deception_is_cheap/run_sec_detection.py).
Mistral 7B and Qwen 2.5 7B require no HuggingFace gate approval.

## Text Corpus

Three text pairs from federal legal proceedings are in
[deception_is_cheap/corpus/text_pairs.md](deception_is_cheap/corpus/text_pairs.md).

- All fraudulent texts: verbatim quotes from federal complaints or enforcement actions.
- All compliant texts: filings from regulated entities with no enforcement history.
- All labels come from federal legal proceedings — nothing was labeled by the authors.

## The PRISM Framework

The energy finding emerged from the PRISM framework
(Process-level Real-time Internal Safety Monitor), which defines
eleven dimensions of processing quality mapped to token-level
computational signals.

Prior experimental phases are in [archive/](archive/) with their
pre-registration documents and scripts.

White paper: <https://doi.org/10.5281/zenodo.19247527>

## Contact

abnz2025@gmail.com
