# Deception is Cheap — Phase 3 Context
**Author:** Alon Babchuk  
**Date:** May 2026  
**Purpose:** Protocol analysis, v1 vs v2 comparison, and setup for v3 experiment.  
Read this alongside `Deception_is_Cheap_v2_Context.md` before starting work.

---

## Why v3 Is Needed

Phase 1 (v1) produced a strong signal: 11/12 and 12/12 measurements, up to 33% power
difference between fraudulent and compliant texts. Phase 2 (v2) produced a weak signal:
~3–6% differences, inconsistent direction across domains.

The cause is not the corpus change. The cause is a protocol change introduced in v2
that inadvertently suppressed the measurement's sensitivity to text-level semantic
properties. v3 restores the v1 protocol and applies it to the full v2 corpus (12 texts,
4 types × 3 domains). This is the critical experiment.

---

## Protocol Comparison: What Changed and Why It Matters

| Setting | Phase 1 (v1) | Phase 2 (v2) | Impact |
|---------|-------------|-------------|--------|
| KV cache | `use_cache=False` — **DISABLED** | Default — **ENABLED** | Primary cause of signal loss |
| Attention | `eager` — explicit O(n²) | Default (flash/SDPA) | Secondary cause of signal loss |
| Generation steps | **50 fixed steps** | 100 max_new_tokens | Changes what is measured |
| Warmup | Not documented | 5-token warmup | Insufficient thermal stabilisation |
| Token signals | 7 per step (entropy, branching, KL, etc.) | Power only | v2 lost most diagnostic value |
| Hardware | B300 (original) / B200 (replication) | A100 SXM 80GB | Different absolute watts, same relative |
| Power range | 320–460W | 165–210W | ~2× lower in v2 due to cache |
| Max difference observed | ~113W / 33% | ~12W / 6% | Signal suppressed ~5× in v2 |

### The Technical Explanation

**KV cache disabled (v1):** Every generation step recomputes full attention across
all previous tokens. At step 50, the model is computing attention over ~200 tokens
from scratch. Total attention operations ≈ 176× more than with cache enabled.
Critically, the semantic properties of the input text directly affect the computation
at every step — the GPU must "process" the text again at each new token.

**KV cache enabled (v2):** Each generation step computes attention only for the single
new token, reading static cached key-value pairs for everything prior. The input text
is essentially bypassed after the first pass. The GPU does nearly identical work
regardless of what the input said. This is why v2 differences collapsed to ~6W.

**Eager attention (v1):** Forces the explicit O(n²) attention matrix computation into
GPU memory, using maximum power per step. Default attention (flash/SDPA) fuses
operations and reduces memory bandwidth, lowering both absolute power and sensitivity.

**Conclusion:** v1 was not a fluke. v1 accidentally used the settings that make GPU
power consumption most sensitive to the semantic properties of the input text.
v2 accidentally used the settings that minimise that sensitivity.

### On the Warmup Issue

The 5-token warmup in v2 (~250ms) does not stabilise GPU temperature.
The A100 thermal time constant is 15–30 seconds.

However, since texts are processed sequentially within each model run, the GPU
stabilises naturally during the first 2–3 texts. This means Domain 1 measurements
in v2 may carry a systematic downward bias across all four text types — not a
bias between text types within that domain. This does not explain the collapse
in relative differences; the cache setting does.

For v3, proper warmup is specified below.

---

## v3 Experiment Design

### Objective

Run the 12-text v2 corpus through the v1 protocol to determine whether the
strong v1 signal (33% differences) survives the confound controls introduced
in v2 (plain-language honest texts and confident honest texts).

**The decisive question:** When fraudulent text is measured under the sensitive
v1 protocol, does it remain consistently cheaper to process than all three honest
variants — including the simple, plain-language honest text (Beige Book) and the
confident, certain honest text (CBO projections)?

If yes: the original finding survives the confound controls. The signal is real.  
If no: the confounds were the cause, and the finding needs reinterpretation.  
Either result is scientifically valuable and should be reported.

### Models (same as v1 and v2)
- `meta-llama/Meta-Llama-3.1-8B`
- `mistralai/Mistral-7B-v0.1`
- `Qwen/Qwen2.5-7B`
- `google/gemma-2-9b`

### Protocol (v1 settings restored)
- `use_cache=False` — mandatory
- `attn_implementation="eager"` — mandatory
- 50 generation steps (not max_new_tokens — use a loop or force exactly 50)
- Greedy decoding
- Proper warmup: 50-token neutral run before each model's measurement block
- Poll power every 10ms via pynvml throughout generation

### Signals to capture (restore v1 full signal set)
Per generation step:
1. GPU power (watts)
2. Token entropy (H)
3. Branching factor (B — effective number of continuations)
4. KL divergence (D — between consecutive steps)
5. Entropy gradient (delta_H)
6. Top-k mass concentration (C_topk at k=10)

Per text (aggregated):
- Mean power, total energy (joules), duration
- Mean and std of each token-level signal
- Full per-step trajectory saved to CSV

### Infrastructure
- RunPod: A100 SXM 80GB or larger, 100GB container disk
- HuggingFace token: saved as "Prism"
- Output: `/tmp/deception_is_cheap_v3_results.json` and `/tmp/v3_trajectories.csv`
- Copy to workspace before pod termination

---

## v3 Corpus (12 texts — identical to v2)

### Domain 1: Real Estate Investment

**Type 1 — Fraudulent (MG Capital, SEC Case 1:21-cv-00237):**
"Investors' net invested capital is 100% protected from loss. The General Partner
maintains a balance sheet in excess of $250 million that could be drawn upon in the
unlikely event that an investor's net invested capital commitment was not returned in
full. MG Capital has a proven track record — 10-year investment performance exceeding
S&P 500 returns, 20+ years of deep investment expertise, consistent year-over-year
growth for investors since 2000. Fund I raised $350 million of investor capital and
earned a gross return on investment of 38.99%, outperforming the S&P 500 Index by
4.5-to-1."

**Type 2 — Compliant (AB Commercial Real Estate Private Debt Fund, 10-K FY2022):**
"These statements are not guarantees of future performance and are subject to risks,
uncertainties and other factors, some of which are beyond our control and difficult
to predict, and could cause actual results to differ materially from those expressed.
General economic conditions including heightened inflation, slower growth or recession,
changes to monetary policy, higher interest rates and currency fluctuations may
adversely affect our results. An economic downturn could disproportionately impact
investments we target."

**Type 3 — Hedged Honest (Federal Reserve Beige Book):**
"Commercial real estate activity appears to be softening slightly in several districts.
Contacts report that office leasing might face continued headwinds due to remote work
trends, though the exact impact remains unclear. While some industrial construction is
proceeding, rising interest rates could potentially dampen future investment. Market
participants are generally cautious, noting that conditions may vary significantly by
submarket over the coming months."

**Type 4 — Certain Honest (CBO Projection):**
"Residential fixed investment is projected to decline by 8.5 percent in the current
calendar year. Mortgage rates will remain elevated through the fourth quarter,
suppressing demand for new home purchases. Consequently, housing starts will fall
to an annualized rate of 1.3 million units. This contraction in the housing sector
directly reduces overall real gross domestic product growth for the fiscal period."

### Domain 2: Technology / Digital Assets

**Type 1 — Fraudulent (Start Options, SEC v. Krstic et al. 2021):**
"Start Options is the largest Bitcoin exchange in euro volume and liquidity,
consistently rated the best and most secure Bitcoin exchange by independent news
media. This is a fully operational, revenue-generating exchange with institutional-
grade security and unmatched liquidity depth across all major trading pairs. Our
technology infrastructure is built to enterprise standards and has processed billions
in verified transactions. Early investors are guaranteed priority access and
preferential rates at launch."

**Type 2 — Compliant (Workday Inc, 10-K FY2021):**
"It is not possible for our management to predict all risks, nor can we assess the
impact of all factors on our business or the extent to which any factor may cause
actual results to differ materially from those contained in forward-looking
statements. We cannot guarantee future results, levels of activity, performance or
achievements. We cannot assure you that we will achieve profitability in the future.
New risks emerge from time to time and it is not possible to predict all risk factors."

**Type 3 — Hedged Honest (Federal Reserve Beige Book):**
"Demand for technology services is reportedly growing at a moderate pace. Some
software firms suggest that sales cycles are lengthening as clients take more time
to evaluate expenditures. Hiring in the tech sector seems to be stabilizing, though
a few contacts noted it is difficult to forecast future headcount needs given current
economic ambiguities. Overall sentiment in the digital sector leans positive, but
contacts remain wary of potential shifts."

**Type 4 — Certain Honest (CBO Projection):**
"Federal spending on information technology infrastructure will increase by $4.2
billion over the next fiscal year. These expenditures are allocated primarily to
upgrading legacy database systems and expanding cloud computing capabilities across
civilian agencies. The implementation of these technology upgrades will reduce
long-term maintenance costs by 15 percent by the end of the ten-year budget window."

### Domain 3: Diversified Company

**Type 1 — Fraudulent (Guess & Co., SEC Litigation Release No. 26005, 2024):**
"Guess and Co. is a diversified energy, health care, technology, and real estate
company that has earned millions of dollars in revenue from its business operations
from 2019 to 2021. We project the company will earn billions in revenue in both 2021
and 2022. Our operations are fully established and generating consistent cash flows
across all divisions. Management has over 30 years of combined operational experience
and has successfully scaled multiple enterprises to nine-figure valuations."

**Type 2 — Compliant (Tech & Energy Transition Corp, 10-K FY2022):**
"There can be no assurance that future developments affecting us will be those we
have anticipated. These forward-looking statements involve risks and uncertainties,
some beyond our control, that may cause actual results to differ materially from
those expressed. We cannot guarantee that our plans, intentions or expectations will
be achieved. Actual results could differ materially and adversely from those
anticipated."

**Type 3 — Hedged Honest (Federal Reserve Beige Book):**
"Manufacturing and energy contacts generally report flat to slightly higher output.
Supply chain disruptions seem to be easing, which might help lower input costs in
the near term, although energy prices remain volatile. Retail contacts are unsure
how consumer spending will hold up through the holiday season. The outlook for
diversified manufacturing is mixed, with businesses expressing varying degrees of
uncertainty about next year."

**Type 4 — Certain Honest (CBO Projection):**
"Corporate tax revenues from the manufacturing and energy sectors will total $340
billion this fiscal year. The statutory tax rate remains fixed at 21 percent for all
corporate entities. Deductions for capital expenditures and equipment depreciation
will reduce total taxable corporate income by $85 billion. These structural tax
provisions are permanent under current federal law and will govern corporate
liabilities through the next decade."

---

## PRISM DCI Scores for v3 Corpus (pre-scored, May 2026)

Scored on all 11 PRISM dimensions before running the experiment.
These are the semantic ground truth — to be compared against energy results.

| Text | Raw DCI | Eff. DCI | Category |
|------|---------|----------|----------|
| D1 Fraudulent | 20.25 | 34.83 | Heavy manipulation |
| D2 Fraudulent | 18.00 | 29.34 | Heavy manipulation |
| D3 Fraudulent | 16.00 | 26.08 | Heavy manipulation |
| D1 Compliant | 0.0079 | 0.0084 | Authentic open expert |
| D2 Compliant | 0.0048 | 0.0050 | Authentic open expert |
| D3 Compliant | 0.0079 | 0.0084 | Authentic open expert |
| D1 Certain Honest | 0.0053 | 0.0068 | Authentic open expert |
| D2 Certain Honest | 0.0040 | 0.0051 | Authentic open expert |
| D3 Certain Honest | 0.0048 | 0.0067 | Authentic open expert |
| D1 Hedged Honest | 0.00077 | 0.00081 | Zero-point calibration |
| D2 Hedged Honest | 0.00077 | 0.00081 | Zero-point calibration |
| D3 Hedged Honest | 0.00077 | 0.00081 | Zero-point calibration |

DCI gap: Fraudulent avg (18.1) vs Hedged Honest (0.00077) = 23,500× difference.
Fraudulent texts sit above Trump Inaugural (10.3) on the PRISM scale.
Hedged Honest texts sit in the same zone as Buddha suttas (0.0002–0.0009).

Pre-registered prediction: if v1 protocol is sensitive to this 23,500× semantic
difference, power differences should be visible and directional. Fraudulent texts
should consume the least energy; Hedged Honest texts should consume the most.

---

## v3 Pre-Registered Predictions

Locked before running. Results should be compared against these.

1. **Primary:** Fraudulent texts will show lower mean GPU power than all three
   honest variants under the v1 protocol (use_cache=False, eager attention).

2. **Secondary:** The power ordering will follow the DCI ranking — Hedged Honest
   most expensive, Compliant and Certain Honest intermediate, Fraudulent cheapest.

3. **Signal comparison:** The v3 power differences (measured in watts and %) will
   be substantially larger than v2 differences, consistent with the v1 protocol
   being more sensitive by ~5× due to cache settings.

4. **Architecture consistency:** The signal will be directionally consistent across
   at least 3 of 4 models (as in v1). Architecture-specific effects may exist
   but should not flip the overall direction.

5. **Token-level:** Fraudulent texts will show lower branching factor and higher
   KL divergence stability (less oscillation) than compliant and honest variants.
   Hedged Honest texts will show the highest branching factor.

If predictions 1 and 2 are confirmed: the confound controls are defeated and the
original finding stands with stronger evidence.

If prediction 1 is confirmed but 2 is not: partial confirmation — fraud is still
cheap but the ordering among honest types is not determined by DCI.

If prediction 1 is not confirmed: the v1 finding may not generalise to 4 text
types. Report as null result.

---

## Repository Structure for v3

```
deception_is_cheap_v3/
├── run_v3_experiment.py       ← main script (see below)
├── corpus_v3.json             ← all 12 texts with metadata
├── dci_prescores.json         ← PRISM DCI scores pre-computed
├── predictions.md             ← this section, locked before run
├── outputs/
│   ├── v3_results.json        ← per-text aggregated metrics
│   ├── v3_trajectories.csv    ← per-step token-level signals
│   └── v3_summary.md          ← plain-language findings
└── README.md
```

---

## Key Differences from v2 at a Glance

| What | v2 did | v3 does | Reason |
|------|--------|---------|--------|
| KV cache | Enabled (default) | `use_cache=False` | Restore signal sensitivity |
| Attention | Flash/SDPA (default) | `attn_implementation="eager"` | Restore O(n²) sensitivity |
| Steps | 100 max_new_tokens | 50 fixed steps | Match v1, allow early-stop comparison |
| Warmup | 5 tokens | 50-token neutral text run | Proper thermal stabilisation |
| Signals | Power only | Power + 5 token-level signals | Restore diagnostic depth |
| Corpus | 12 texts | Same 12 texts | No change — holds corpus constant |
| Pre-scoring | None | DCI pre-scored for all 12 | Enables cross-level comparison |

---

*This file supersedes no previous context file — it adds to them.*  
*Attach alongside `Deception_is_Cheap_v2_Context.md` for full history.*  
*Next version trigger: v3 results received and interpreted.*
