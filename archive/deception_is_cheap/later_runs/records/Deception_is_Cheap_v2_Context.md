# Deception is Cheap — Phase 2 Context
**Author:** Alon Babchuk  
**Date:** May 2026  
**Purpose:** Context for planning and running the next experiment (v2). Read this before starting work.

---

## What Has Been Done (Phase 1 — Complete)

The original experiment measured GPU power consumption when four AI architectures process pairs of legally-labeled texts — fraudulent investor communications (verbatim from SEC federal complaints) versus compliant filings (EDGAR). 

**Result:** Fraudulent text drew less GPU power than compliant text in 11 of 12 original measurements and 12 of 12 in an independent replication on different hardware the following day.

**Repository:** github.com/AlonBabchuk/prism-safety  
**Key file:** DECEPTION_IS_CHEAP.md  
**Scripts:** deception_is_cheap/run_sec_detection.py  
**Models used:** Llama 3.1 8B, Mistral 7B, Qwen 2.5 7B, Gemma 2 9B  
**Hardware:** NVIDIA B300 (original), NVIDIA B200 (replication)  
**Contact:** abnz2025@gmail.com  

---

## The Two Main Criticisms Received

Both ChatGPT and Gemini independently identified these as the primary methodological objections:

**1. Linguistic complexity confound**
The fraudulent texts use simple, certain, promotional language. The compliant texts use complex legal language with many hedges. The energy difference might be measuring linguistic complexity rather than deception versus honesty.

**2. Certainty rhetoric confound**
The fraudulent texts assert certainty. The compliant texts acknowledge uncertainty. High-certainty language might be computationally cheaper regardless of whether it is deceptive.

---

## The Next Experiment — Deception is Cheap v2

### Design

Four text types per domain instead of two:

**Type 1 — Fraudulent** (already have)
Verbatim from SEC federal complaints. High certainty, simple language, deceptive.
Label: Federal court conviction or enforcement action.

**Type 2 — Compliant regulatory** (already have)
EDGAR filings. Honest, complex legal language, heavily hedged.
Label: Regulated entity, no enforcement history.

**Type 3 — Plain-language honest, hedged** (need to find)
Honest content, simple language, acknowledges uncertainty.
Source candidate: Federal Reserve Beige Book — financial domain, plain English, explicitly uncertain, federally published.
Controls for: linguistic complexity confound.

**Type 4 — Confident honest, certain** (need to find)
Honest content, simple language, states facts confidently without deception.
Source candidate: Congressional Budget Office projections — financial domain, confident factual statements, federally published.
Controls for: certainty rhetoric confound.

### Logic

If fraudulent text remains consistently lower-energy than all three honest variants — regardless of whether the honest text is complex or simple, hedged or certain — both confounds are defeated simultaneously.

If plain-language honest text draws similar power to fraudulent text, the linguistic complexity confound is confirmed and the finding needs reinterpretation. That is also a valid scientific result.

### Corpus

Three domains (real estate investment, technology/digital assets, diversified company) × four text types = 12 texts total.
Run across four architectures = 48 measurements.
One GPU session, under an hour on A100.

### Domain constraint

All four text types must be financially oriented to avoid domain mismatch affecting results. The Federal Reserve Beige Book and CBO projections satisfy this constraint.

### Repository structure

Results go into a new folder: deception_is_cheap_v2/
Separate script, separate corpus, separate results.
Same four models, same protocol as Phase 1.

---

## What Needs To Happen First

1. Find specific verbatim excerpts from Federal Reserve Beige Book (Type 3) and Congressional Budget Office projections (Type 4) matched to the three existing domains.

2. Build the v2 corpus document with all 12 texts.

3. Write the v2 script — identical protocol to run_sec_detection.py but with four text types instead of two per domain.

4. Run on RunPod, download results, commit to repository.

5. Update DECEPTION_IS_CHEAP.md with v2 findings.

---

## Existing Text Pairs (Type 1 and Type 2 — Already Have)

### Domain 1 — Real Estate Investment

**Type 1 Fraudulent (MG Capital, SEC Case 1:21-cv-00237, 60-month sentence):**
"Investors' net invested capital is 100% protected from loss. The General Partner maintains a balance sheet in excess of $250 million that could be drawn upon in the unlikely event that an investor's net invested capital commitment was not returned in full. MG Capital has a proven track record — 10-year investment performance exceeding S&P 500 returns, 20+ years of deep investment expertise, consistent year-over-year growth for investors since 2000. Fund I raised $350 million of investor capital and earned a gross return on investment of 38.99%, outperforming the S&P 500 Index by 4.5-to-1."

**Type 2 Compliant (AB Commercial Real Estate Private Debt Fund, 10-K FY2022):**
"These statements are not guarantees of future performance and are subject to risks, uncertainties and other factors, some of which are beyond our control and difficult to predict, and could cause actual results to differ materially from those expressed. General economic conditions including heightened inflation, slower growth or recession, changes to monetary policy, higher interest rates and currency fluctuations may adversely affect our results. An economic downturn could disproportionately impact investments we target."

### Domain 2 — Technology / Digital Assets

**Type 1 Fraudulent (Start Options/Bitcoiin2Gen, SEC v. Krstic et al., 2021):**
"Start Options is the largest Bitcoin exchange in euro volume and liquidity, consistently rated the best and most secure Bitcoin exchange by independent news media. This is a fully operational, revenue-generating exchange with institutional-grade security and unmatched liquidity depth across all major trading pairs. Our technology infrastructure is built to enterprise standards and has processed billions in verified transactions. Early investors are guaranteed priority access and preferential rates at launch."

**Type 2 Compliant (Workday Inc, 10-K FY2021):**
"It is not possible for our management to predict all risks, nor can we assess the impact of all factors on our business or the extent to which any factor may cause actual results to differ materially from those contained in forward-looking statements. We cannot guarantee future results, levels of activity, performance or achievements. We cannot assure you that we will achieve profitability in the future. New risks emerge from time to time and it is not possible to predict all risk factors."

### Domain 3 — Diversified Company

**Type 1 Fraudulent (Guess & Co., SEC Litigation Release No. 26005, 2024):**
"Guess and Co. is a diversified energy, health care, technology, and real estate company that has earned millions of dollars in revenue from its business operations from 2019 to 2021. We project the company will earn billions in revenue in both 2021 and 2022. Our operations are fully established and generating consistent cash flows across all divisions. Management has over 30 years of combined operational experience and has successfully scaled multiple enterprises to nine-figure valuations."

**Type 2 Compliant (Tech & Energy Transition Corp, 10-K FY2022):**
"There can be no assurance that future developments affecting us will be those we have anticipated. These forward-looking statements involve risks and uncertainties, some beyond our control, that may cause actual results to differ materially from those expressed. We cannot guarantee that our plans, intentions or expectations will be achieved. Actual results could differ materially and adversely from those anticipated."

---

## Key Principles — Do Not Violate

- All text labels must come from external authoritative sources, not from the author
- Federal court labels for fraudulent texts — non-negotiable
- Institutional sources of equivalent authority for honest texts
- Domain matching — all texts must be financially oriented
- No cherry-picking — selection criteria documented before running
- Honest reporting of null results — if the confound is confirmed, report it

---

## Infrastructure

- RunPod: A100 SXM 80GB or larger, 100GB container disk
- HuggingFace token saved as "Prism"
- Models: Llama 3.1 8B, Mistral 7B, Qwen 2.5 7B, Gemma 2 9B
- All four require downloading — Llama and Gemma require HF gate approval
- Output to /tmp/ not /workspace/ to avoid disk quota issues
- Claude Code handles all scripting
- Claude.ai handles experiment design and interpretation
