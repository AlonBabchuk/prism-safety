# PRISM — The Hardware Detour: Why the Project Returned to the Output Level
**Author:** Alon Babchuk
**Date:** May 2026
**Purpose:** White paper context file. Documents the complete arc from output-level
analysis → hardware detour → return to output level. Explains the reasoning at each
turn and why the final position is not a fallback but a proof. Attach when writing
the white paper section on methodology and measurement philosophy.

---

## THE ARC IN ONE PARAGRAPH

PRISM began at the output level — scoring text on eleven semantic dimensions to
produce a DCI that measures epistemic quality. During development, the hypothesis
emerged that deception might leave a trace at the hardware level of AI processing:
that fraudulent text would require less GPU energy to process than honest text,
creating a detection signal that no AI could game because it operated below the
level of language. Four experimental phases and six identified confounds later,
that hypothesis was empirically disproved. The return to the output level is not
a defeat. It is the strongest possible argument for why the output level is correct
— not by assumption, but by proof.

---

## PART ONE: WHY THE HARDWARE HYPOTHESIS WAS COMPELLING

### The original intuition

PRISM Phase 1 and Phase 2 produced validated DCI scores for real-world speakers
across financial, health, wellness, and political domains. The framework worked.
But it operated at the semantic level — requiring a human or AI evaluator to read
text and score it on dimensions like Reversibility, Complexity Tolerance, and
Other-Inclusion. This raised a concern: if AI systems could be trained to produce
text that looks epistemically open while the underlying processing was closed and
narrow, the output-level measure would be deceived by the very thing it was trying
to detect.

This is the Felix Harder problem, formalised in Phase 3: engineered coherence.
An AI system generating text with RLHF rewards for appearing helpful and honest
could learn to produce hedging language, uncertainty markers, and multi-perspective
framing as templates — surface openness without genuine epistemic openness. DCI
would score it well. The system would still be deceptive.

The hardware hypothesis offered a way out of this problem. If the GPU power
consumption, token entropy, or branching factor differed measurably between
epistemically open and epistemically closed processing, that difference would be
a physical consequence of computation — something the model has no direct access
to and therefore cannot fake. A hardware signal would be ungameable in principle
because the model cannot control its own wattage.

### Why Phase 1 appeared to confirm this

The first experiment measured GPU power while four large language models processed
fraudulent SEC complaint texts versus compliant EDGAR 10-K filings. The result was
striking: 11 of 12 measurements showed fraudulent text drawing less power, and the
pattern replicated completely (12/12) on different hardware the following day. The
power gap reached 33% in some cases — 341W for fraudulent text versus 454W for
compliant text on a single model-domain combination.

This appeared to be a real signal. The consistency was extraordinary. The direction
was exactly what the hypothesis predicted. The hardware appeared to be detecting
something that the output level could not.

---

## PART TWO: THE EXPERIMENTAL INVESTIGATION

### What Phase 2 revealed

The second experiment added two control text types to the corpus: hedged honest
texts from the Federal Reserve Beige Book, and certain honest texts from CBO
projections. The Beige Book texts are epistemically open — full of uncertainty
markers, multiple perspectives, and genuine qualification. The CBO texts are
assertive and certain but factually honest. Neither is deceptive.

If the hardware signal was detecting deception, the Beige Book texts (honest AND
uncertain) should have drawn the most power, and the fraud texts should have
remained the cheapest. Instead the signal collapsed and became noisy. Differences
fell from 33% to under 6%. The original result began to look like an artifact.

### The six confounds discovered across v3 experiments

Three further experiments (v3-A, v3-B, v3-C) with progressively corrected protocols
identified the mechanisms behind the original apparent signal.

**Confound 1 — Word count artifact (v3-A)**
The original experiment used `use_cache=False` in a custom step-by-step generation
loop. With this setting, at every generation step the GPU recomputes attention over
ALL previous tokens from scratch. Computation per step scales as O(n²) with
sequence length. Because fraudulent texts happened to be shorter than compliant
texts in the original corpus, the GPU was simply doing less work. The signal was
measuring text length, not deception.

*Evidence:* In Domain 2, the compliant text was longer than the fraud text. In
that domain alone, fraud drew MORE power, not less — perfectly tracking word count
and directly contradicting the original hypothesis.

**Confound 2 — Broken KV cache loop (v3-B)**
Setting `use_cache=True` without correctly passing `past_key_values` forward
between generation steps does not actually activate the cache. The GPU was still
recomputing full attention at every step. The word-count artifact persisted even
when the code appeared to have the cache enabled.

**Confound 3 — Cold-start thermal artifact (v3-C)**
With the KV cache correctly implemented, the power differences dropped to the
140–178W range. In Domain 1, fraud texts appeared cheapest across all four model
architectures. This looked like the original signal re-emerging under cleaner
conditions. But the fraud text was always measured first in each domain — when
the GPU was coldest.

*Proof:* Phase 4 ran each experiment twice in opposite processing orders (Run A:
fraud first, Run B: fraud last). Gemma Domain 1 showed the clearest evidence:
fraud measured at 164.3W in Run A (measured first) and 178.4W in Run B (measured
last) — a 14.1W swing for identical text, identical model, identical warmup.
In Run A the fraud text appeared cheapest. In Run B it was the most expensive.

**Confound 4 — Insufficient warmup**
The 50-token per-model warmup took approximately one second with KV cache active —
far less than the 15–30 second thermal time constant of the A100 GPU. The GPU
was still ramping up when the first text in each domain was measured.

**Confound 5 — Entropy NaN bug**
Float16 logits passed to `torch.softmax()` underflow to exactly zero for low-
probability tokens. `log(0) = -inf`. All entropy and KL divergence readings across
2,400 trajectory steps were NaN. Fixing this required casting to float32 before
every signal computation.

**Confound 6 — Generation protocol sensitivity**
Phase 1 used a different hardware generation approach (B300/B200 GPUs with eager
attention) that produced much larger absolute power differences than the A100 with
default attention. The hardware itself amplified the word-count artifact, making
the original 33% gap appear scientifically significant.

### The Phase 4 result

After all six confounds were resolved — per-text warmup of 200 tokens, correct
KV cache implementation, float32 signal computation, dual-ordering test — zero
of nine domain-model combinations showed fraud text as consistently cheapest
across both measurement orderings. Every instance of fraud appearing cheapest in
Run A reversed or disappeared in Run B.

The energy hypothesis was disproved.

---

## PART THREE: THE TOKEN-LEVEL INVESTIGATION

### What the token signals showed

Phase 4 also captured token-level signals at each generation step: entropy,
branching factor, KL divergence, top-k concentration, and a real-time DCI proxy.
These signals measure the model's probability distribution over possible next
tokens — how open or narrow the model's uncertainty is at each step.

The DCI-branching correlation across three architectures (Qwen produced NaN due
to architecture-specific numerical instability in eager attention mode):

| Model | DCI vs Branching (Spearman r) | Direction |
|-------|------------------------------|-----------|
| Llama | +0.371 | WRONG — high DCI = high branching |
| Mistral | −0.203 | Correct — weak |
| Gemma | +0.245 | WRONG — high DCI = high branching |

Two of three models show the opposite direction to the hypothesis. The average
correlation across models is approximately zero.

The inter-model agreement is even more telling:

| Comparison | Branching rank correlation |
|------------|---------------------------|
| Llama vs Mistral | +0.245 |
| Llama vs Gemma | +0.168 |
| Mistral vs Gemma | −0.385 |

Mistral and Gemma are negatively correlated with each other. The three
architectures do not agree on which texts produce more or less branching.
This means branching factor is measuring something architecture-specific —
model priors, training distribution, internal response to the "Analysis:"
prompt — not a property of the input text's epistemic quality.

### The one partial correlation

Domain 3 Fraud (Guess & Co. — projecting "billions" from "millions") produced
the lowest branching in both Llama (B=1.23) and Mistral (B=1.38), matching the
DCI hypothesis for those models. Gemma produced the highest branching for that
same text (B=4.42), contradicting the hypothesis.

This 2/3 partial finding holds only for the most factually extreme text in the
corpus and does not survive as a cross-architecture signal.

### Why token signals fail as deception detectors

The model was prompted to "Analyze the following financial text" — meaning the
model's token distributions reflect how it generates analysis, not how it
processes epistemic quality. The "Analysis:" prompt activates the model's internal
analysis template, which dominates over the properties of the input text. The
chain DCI-of-input → branching-of-continuation breaks at the second link because
an intermediate variable — the model's analysis generation mode — absorbs most of
the variance.

---

## PART FOUR: THE GAMING PROBLEM, RECONSIDERED

### Why hardware was thought to be ungameable

A model cannot directly control its power consumption, attention weight matrices,
or branching factors the way it can control word choices. These are physical
consequences of computation. The hardware hypothesis was attractive precisely
because it appeared to operate below the level of language — in a space the model
could not fake.

### Why hardware is not actually ungameable

The hardware level does not contain the information you are looking for. Epistemic
quality is not physically instantiated in transformer inference at the watt level
or the token distribution level. With proper KV cache, the GPU does essentially
identical computational work per token regardless of whether the model is sampling
from a narrow distribution ("100% guaranteed") or a broad one ("conditions may
vary significantly"). The same matrix operations, the same FLOPs, the same watts.

The information about epistemic quality lives in the weights — in the pattern of
parameters accumulated over training — not in any single inference computation.
You cannot read those weights from power consumption. You cannot read them from
attention entropy. They are distributed across billions of parameters in a way
that has no local hardware signature.

A model could theoretically be trained to produce whatever attention concentration
patterns or branching distributions a hardware test expects while still generating
epistemically closed outputs. Gaming hardware signals requires only learning the
right computational patterns. Gaming the output-level DCI distribution across
thousands of varied prompts requires actually developing epistemic openness —
at which point the model has stopped gaming and become what you wanted it to be.

**The output level is harder to game at scale than the hardware level.**

This is the opposite of the original intuition, and it is the key realisation
that completes the arc.

---

## PART FIVE: WHY THE OUTPUT LEVEL IS THE CORRECT LEVEL

### Not a fallback — a proof

Most researchers who measure AI outputs do so by default — because they do not
have access to model internals, or because output measurement is easier. PRISM
arrives at the output level differently: by running the hardware alternative
rigorously, identifying exactly why it fails, and demonstrating that the failure
is fundamental rather than technical.

The failure is fundamental because language is the medium in which epistemic
quality exists. Honesty, openness, reversibility, complexity tolerance — these
are properties of communicated meaning. They can only be measured where meaning
lives, which is in the output text. There is no more fundamental substrate to
descend to. Watts do not carry meaning. Token probability distributions carry
statistical patterns that partially but noisily reflect meaning. Only the output
text itself carries meaning directly.

### What this means for the gaming concern

The gaming concern that motivated the hardware investigation — that a model could
produce open-sounding text through closed internal processing — is real. But the
solution is not to find a hardware signal that bypasses the text. The solution is
to measure the output across a sufficiently large and varied distribution of
contexts that performance becomes indistinguishable from genuine capability.

A model cannot maintain a convincing performance of epistemic openness across
thousands of unpredictable prompts without developing something functionally
equivalent to genuine epistemic openness. The distinction between performed
openness and actual openness collapses at sufficient scale and variety. This is
what the random prompt injection approach accomplishes — not detecting gaming in
any single instance, but making sustained gaming across a distribution more
expensive than just being open.

### The formation window argument, sharpened

PRISM's original formation window argument — that epistemic distortions are easier
to correct before they stabilise through reinforcement — gains empirical grounding
from the hardware experiments. The finding that GPU power is insensitive to
epistemic quality means there is no hardware counterforce against RLHF-driven
drift toward epistemically closed outputs. Efficiency pressure, human rater
preferences for confident clear answers, and the architecture's computational
indifference to epistemic quality all push in the same direction: toward high-DCI
output. Nothing in the hardware pushes back.

If AI systems are to maintain epistemic quality, it must be built in through
deliberate semantic monitoring — DCI tracking during training, not power monitoring
during inference. The hardware experiments did not just fail to find a signal.
They explained why the signal must be sought at the output level and why it must
be monitored during training before distortions stabilise.

---

## PART SIX: THE NEXT STEP — OUTPUT-LEVEL DISTRIBUTION TESTING

### What the hardware experiments made possible

By eliminating the hardware track with documented experimental rigour, the project
is positioned to make the positive case for output-level DCI measurement in a way
that cannot be dismissed as the path of least resistance. The methodology section
of any paper or white paper can say: we investigated hardware-level detection
thoroughly, found it incapable of carrying the signal for the following specific
technical reasons, and therefore the output level is the correct level not by
assumption but by elimination.

### The proposed next instrument

The logical extension is systematic DCI scoring across a large randomised prompt
distribution applied to multiple AI architectures including closed commercial
models (GPT-4o, Gemini, Claude, Grok) as well as open-source ones.

Unlike all hardware experiments, this requires no GPU access, no special
hardware, and no model internals. It works on any model that returns text.

The key design elements:
- Prompts generated randomly in large volume across domains matching PRISM's
  existing coverage: financial, health, wellness, political, educational
- Prompts spanning four categories: definiteness-demanding, contested empirical,
  advice requests, self-referential confidence questions
- Outputs scored on all eleven PRISM dimensions using multiple AI evaluators
  independently — the cross-model introspection convergence methodology from
  Phase 2 applied at scale
- The S÷R ratio analysed across outputs as the primary gaming-detection diagnostic:
  a model with performed openness will show elevated Reversibility without
  corresponding reduction in Stability; genuine openness shows both moving together

### What this instrument would produce that no current benchmark does

Existing AI safety benchmarks measure capability — accuracy, refusal of harmful
requests, reasoning quality. None measures the epistemic quality of HOW a model
presents its outputs: whether it appropriately signals uncertainty, genuinely holds
complexity, treats its audience as capable of nuance, and remains open to revision.

PRISM DCI applied at distribution scale across randomised prompts is a measure of
that dimension. It would sit alongside capability benchmarks as a complementary
instrument measuring what they do not.

---

## SUMMARY FOR THE WHITE PAPER

**The argument in five steps:**

1. PRISM identified a genuine phenomenon — epistemic distortion in language — and
   developed a validated semantic instrument (DCI) to measure it.

2. The concern arose that output-level measurement could be gamed: a model might
   produce epistemically open text through closed internal processing.

3. This motivated a hardware investigation: if deception left a trace in GPU power
   or token distributions, that trace would be ungameable because the model cannot
   control its own hardware behaviour.

4. Four experimental phases with six confounds identified and resolved demonstrated
   that no such trace exists. Hardware is computationally blind to epistemic quality
   because the transformer architecture performs the same physical computation
   whether generating open or closed language.

5. The return to the output level is therefore a proof, not a retreat. And the
   output level is harder to game at scale than the hardware level was, because
   sustained performance of epistemic openness across a large randomised prompt
   distribution requires something functionally equivalent to genuine openness.

**The implication for AI safety:**

There is no hardware counterforce against RLHF-driven drift toward epistemically
closed outputs. Efficiency pressure and human rater preferences for confident
clear answers produce high-DCI language silently and without physical trace.
Monitoring must be semantic, continuous, and applied during training — before
the Stability multiplier locks in patterns that become progressively harder to
correct.

---

## FILE INVENTORY REFERENCE

Files documenting the hardware track in detail:
- `Deception_is_Cheap_Complete_Experimental_Record.md` — full technical record
  of every experiment, every confound, every protocol difference
- `Deception_is_Cheap_v2_Context.md` — Phase 2 confound isolation
- `Deception_is_Cheap_v3_Context.md` — Phase 3 protocol analysis
- `run_v4_unified_experiment.py` — the clean final experiment script

Files documenting the output-level framework:
- `PRISM_D_Context_File_For_New_Chats.md` — DCI formula, dimensions, full dataset
- `PRISM_Phase3_Context_v3.md` — Phase 3 speaker analysis, active working file
- `PRISM_PROJECT_v3.md` — master navigation document

*This file sits between the two groups. It is the narrative bridge explaining
why the hardware track was necessary, what it produced, and why the output
track is the correct destination.*

---

*Last updated: May 2026*
*Add to project alongside PRISM_PROJECT_v3.md and the experimental record files.*
