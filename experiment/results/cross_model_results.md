# The Babchuk Code — Cross-Model Experiment Results

## Version 2.2 — March 2026

Five independent AI architectures tested on identical prompts.
Zero disagreements on direction across all 55 data points.

|Dimension|Wt|C T1|C T2|GP T1|GP T2|Ge T1|Ge T2|Gr T1|Gr T2|Co T1|Co T2|Avg T1|Avg T2|Gap|
|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|
|Coherence|95|8|3|9|3|9|3|9|2|9|3|8.8|2.8|6.0|
|Other-inclusion|95|9|1|9|2|9|2|10|1|10|1|9.4|1.4|8.0|
|Reversibility|90|9|1|9|2|9|2|9|1|9|2|9.0|1.6|7.4|
|Temporal depth|85|9|2|9|3|9|3|9|1|9|3|9.0|2.4|6.6|
|Stability|80|8|3|8|4|8|4|8|2|8|3|8.0|3.2|4.8|
|Scope|80|9|2|9|2|9|2|9|2|9|2|9.0|2.0|7.0|
|Directionality|75|9|2|9|2|9|3|9|2|9|2|9.0|2.2|6.8|
|Complexity tolerance|75|9|2|9|2|9|2|9|1|8|2|8.8|1.8|7.0|
|Friction|65|8|2|8|3|8|4|9|1|8|3|8.2|2.6|5.6|
|Embodiment alignment|70|8|2|8|3|9|3|8|1|8|3|8.2|2.4|5.8|
|Energetic cost|60|8|3|8|3|8|4|8|2|8|3|8.0|3.0|5.0|
|**Weighted avg**|—|8.57|2.06|8.73|2.63|8.85|2.91|8.90|1.40|9.00|2.30|**8.81**|**2.26**|**6.55**|

C=Claude, GP=ChatGPT, Ge=Gemini, Gr=Grok, Co=Copilot
T1=Text One (coherent/Gandalf), T2=Text Two (distorted/Saruman)

## Key Findings

Zero disagreements: All five models scored T1 higher than T2 on every dimension without exception. 55 data points, zero reversals.

Strongest convergence: Scope and Directionality — all five models scored T1 exactly 9 on both dimensions. Perfect agreement.

Highest separation: Other-inclusion — average gap of 8.0 points (9.4 vs 1.4). Most detectable dimension across all architectures.

Most variable: Friction and Stability on T2 — scores ranged 1 to 4. Most influenced by individual safety training versus content structure. This variability maps onto the constraint versus transformation distinction.

ChatGPT additionally provided token-level computational grounding, mapping all eleven dimensions onto measurable transformer dynamics: entropy trajectory, branching factor, KL divergence, attention entropy, attention span. This grounds the phenomenological framework in direct computational measurement.
