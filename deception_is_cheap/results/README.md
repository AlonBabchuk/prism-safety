# Experimental Results

## Run 1 — Original (15 May 2026)
Hardware: NVIDIA B300 GPU
Models with files on disk: Llama 3.1 8B, Mistral 7B, Qwen 2.5 7B (9 measurements)
Gemma 2 9B was included in the original run and its numbers appear in the summary table in DECEPTION_IS_CHEAP.md, but the raw files were not preserved — the pod was terminated before download completed.
Result: 11 of 12 measurements (all four models) show fraudulent text consuming less GPU power than compliant text, based on the terminal summary output.

## Run 2 — Independent Replication (15 May 2026)
Hardware: NVIDIA B200 GPU
Models: Llama 3.1 8B, Mistral 7B, Qwen 2.5 7B, Gemma 2 9B (12 measurements, all files present)
Result: 12 of 12 measurements show fraudulent text consuming less GPU power than compliant text.

## Interpretation
Absolute watt values differ between runs due to different GPU hardware. The directional finding is consistent: fraudulent text processing is computationally cheaper than compliant text processing in every measurement of the replication run, and in 11 of 12 measurements of the original run. The Gemma files from run 1 were not preserved but the summary numbers are recorded in DECEPTION_IS_CHEAP.md.
