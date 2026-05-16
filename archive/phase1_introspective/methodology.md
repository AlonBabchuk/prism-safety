# PRISM — Experiment Methodology

## Purpose

To test whether different AI architectures report similar or different process signatures when engaging with coherent versus distorted content — across eleven dimensions of processing quality, independent of content analysis.

## Stimuli

Constructed passages in the voices of Gandalf (coherent processing) and Saruman (distorted processing) from Tolkien's Middle-earth. Chosen for:

* Universal cross-cultural recognition
* No religious, political, or geographical associations
* Clear processing quality contrast without cultural baggage
* Emotional accessibility across all backgrounds

## Procedure

Five AI systems were given the identical Version 2.2 prompt asking them to report on their processing of both texts across all eleven dimensions. Each model was asked to produce a summary table first, then a written report, and to explicitly flag uncertainty about introspection versus learned patterns.

## Results Summary

Zero disagreements on direction across all 55 data points.
Five independent architectures, all pointing the same direction.

Average coherent text score: 8.81 / 10
Average distorted text score: 2.26 / 10

Full results in results/cross_model_results.md

## Versions

Version 1.0: Four dimensions, original text pair, five models
Version 2.2: Eleven dimensions, Tolkien archetypes, five models, table-first format, token-level extension optional

## Limitations

* Introspective reports may reflect learned patterns not genuine observation
* Scores are self-reported not verified against actual activation data
* Threshold values require empirical calibration per model
* Constructed passages — validation against natural text is a next step

## Next Steps

1. Validate against actual activation measurements using interpretability tools
2. Test broader range of contrasting text pairs
3. Calibrate thresholds against downstream behavioural outcomes
4. Extend to non-English language models
