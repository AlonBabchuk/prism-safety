# Later runs: confound controls and falsification

These are the runs that followed the two initial runs of 15 May 2026 (in
`../results/`). They are copied unchanged from the project's working records.

| Folder | What it contains |
|---|---|
| `v2_replication/` | Phase 2 confound controls. Two honest control text types were added: hedged texts (Federal Reserve Beige Book) and certain texts (CBO projections). The fraudulent–compliant gap shrank from about 33% to under 6%. |
| `v3_confound_controls/` | Three v3 runs (v3-A, v3-B, v3-C) with progressively corrected protocols. They identified a word-count artefact, a broken KV-cache loop and a cold-start (GPU warm-up) ordering artefact. |
| `v4_falsification/` | The final run, after all identified confounds were corrected, with both measurement orders tested. None of nine domain–model combinations showed fraudulent text as consistently cheapest across both orders. |
| `records/` | The project's written records of these runs. `PRISM_Hardware_vs_Output_Arc.md` gives the overall account and conclusion, and `Deception_is_Cheap_Complete_Experimental_Record.md` is the run-by-run record. |

**Conclusion:** the energy-asymmetry hypothesis was not supported. The project
moved to output-level measurement.
