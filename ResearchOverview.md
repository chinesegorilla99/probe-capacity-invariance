# Research Project Overview

### Working title (provisional — a scope-check tool, not the final title)
**"When 'Invariant' Means 'Linearly Inaccessible': A Controlled Study of Probe Capacity in Contrastive Learning"**

Title candidates (keep for the end; pick the final one *from your results*):
- *When "Invariant" Means "Linearly Inaccessible": A Controlled Study of Probe Capacity in Contrastive Learning* (current pick — hook + precise, no asserted finding)
- *Gone, or Just Nonlinear? Distinguishing Genuine Invariance from Linear Inaccessibility in Self-Supervised Representations*
- *What the Linear Probe Misses: Probe Capacity and the Measurement of Self-Supervised Invariance*
- *PRISM: Probing Representational Invariance via Scaling-capacity Measurement* (acronym route)
- *SCOPE: Selectivity-Controlled Probing of Encoded Invariance in Contrastive Representations* (acronym route)

> **How the title works in this project.** Three different things, don't confuse them:
> 1. **The question is fixed from day one** — "when an augmentation makes a factor linearly inaccessible, is the information gone or just nonlinearly hidden?" This is the spine and never changes.
> 2. **The working title above is a scope-check, not decoration.** When unsure whether an experiment belongs, test it against the title: does it serve "a controlled study of probe capacity"? If yes, keep it; if it's a fancier-encoder tangent, it's scope creep. The working title intentionally promises a *study*, not a verdict — no "phase transition," no "invariance is an illusion" — so a result that disagrees with your hunch is data, not failure.
> 3. **The final title is written last, from the results.** If the data is sharp, earn a punchy claim ("Linear Probes Overstate Invariance" or "Invariance Survives Probe Scaling"); if it's nuanced, keep the neutral "controlled study" framing. Don't fall in love with any hook before the data — a committed title quietly biases which experiments you run and emphasize.

---

## 1. One-sentence thesis

When self-supervised contrastive models are described as "invariant" to some factor (color, position, rotation), that invariance is almost always *measured* with a linear probe — so a linear probe's failure to recover the factor may reflect the probe's weakness rather than the representation's content; this project tests, under fully controlled conditions, whether the invariance survives stronger probes or dissolves into a measurement artifact.

---

## 2. The research question

**Primary question.** For an augmentation-induced invariance in a contrastive encoder, how does the recoverability of the "discarded" factor change as a function of probe capacity — and is the standard linear-probe verdict of "invariant" stable, or an artifact of limited probe expressivity?

**Secondary questions.**
- Does the answer depend on *which* factor / augmentation (e.g., is color invariance more "real" than positional invariance)?
- Does it depend on augmentation *strength* (weak vs. strong jitter)?
- Where does the information actually live — encoder vs. projector head — and does that change the verdict?

These are all things you can commit to today because every term is something you define and control, not something you hope to find. That is the property a foundation needs.

---

## 3. Why this matters

Self-supervised / contrastive learning is one of the dominant paradigms for learning representations without labels, and the central intuition — *the augmentations you choose determine the invariances the model acquires* — is now both folklore and the subject of formal theory. That intuition is the design principle practitioners use when picking augmentations.

But the claim is operationalized almost entirely through **linear probing**: train a linear classifier/regressor on the frozen representation to predict the factor; if it can't, the representation is called "invariant" to that factor. This is a measurement with a built-in confound. A linear probe's failure establishes only that the factor is not **linearly decodable** — not that the information has been removed. If a modestly nonlinear probe recovers it, then:

- the field's invariance claims are partly statements about probe weakness, not representation content;
- "choose augmentations to control invariances" is a leakier design lever than assumed;
- downstream conclusions that rely on a factor being "gone" (fairness/nuisance-removal arguments, disentanglement claims) inherit the confound.

Conversely, if stronger probes *cannot* recover the factor, you've produced the controlled evidence that the invariance is genuinely representational — a validation the folklore currently asserts but rarely measures cleanly. **Either outcome is a contribution.** That asymmetry-free payoff is the project's core virtue.

---

## 4. The gap and how this sits relative to prior work

This is the part you must verify yourself before committing (see Phase 0). My searches place the project in an open seam adjacent to several established lines:

- **Probing methodology (the move you're borrowing).** The "control task" idea — comparing a probe's performance on a trained representation against the same probe on a random/untrained representation to isolate what the *probe* contributes vs. what the *representation* contributes — comes from the NLP probing literature (Hewitt & Liang, "control tasks"; related: Pimentel et al. on probing as information). Your novelty is **applying this control-and-capacity lens specifically to the invariance claim in vision SSL**, on synthetic data with known factors. The methodology is borrowed and credited; the application is the contribution.

- **What contrastive learning encodes (adjacent, not the same).** Work on identifiability and "what makes good views" (von Kügelgen et al.; Tian et al.) and theoretical analyses of contrastive representations (e.g., spectral/geometric treatments, Balestriero & LeCun; HaoChen et al. spectral contrastive loss) studies *what is encoded*. They largely assume the measurement of invariance is unproblematic. You interrogate the measurement itself.

- **Encoder vs. projector (a confound you must absorb, not ignore).** Geometric analyses (e.g., Cosentino et al.) argue invariance is concentrated in the projector head while the encoder preserves more structure. This is not your question, but it is a rival explanation for any result you get, so your design must probe the **encoder** (the representation people actually use downstream) and you should include projector-vs-encoder as a reported axis. Engaging this head-on is exactly the kind of move that signals maturity to a reviewer.

**The honest status:** the broad topic (contrastive invariance) is heavily studied and theory-rich; the *specific, controlled, probe-capacity-vs-recoverability study with control tasks on known factors* appears open. "Appears" is doing real work in that sentence — Phase 0 exists to convert it to "is."

---

## 5. Core hypotheses (pre-register these)

State them now, in writing, so the project can't be quietly bent to fit whatever you find.

- **H1 (capacity dependence).** Recoverability of an augmentation-targeted factor is non-decreasing in probe capacity; for at least one factor it rises materially above the linear-probe level.
- **H2 (heterogeneity).** The size of that capacity effect differs across factors/augmentations (some invariances are "more linear-only" than others).
- **H3 (genuineness via control).** For factors where recoverability stays at the random-encoder control level across all probe capacities, the invariance is genuinely representational (information is absent, not merely nonlinearly hidden).

Note H1 and H3 are not mutually exclusive across factors — the interesting result is likely a *map*: some invariances are real, some are probe artifacts. That map is the paper.

---

## 6. Methodology

### 6.1 Data — synthetic, factor-controlled

Use a dataset with known, independent generative factors so "the true value of factor F" is defined by construction:

- **Primary: Shapes3D (3D Shapes)** — 6 independent factors: floor hue, wall hue, object hue, scale, shape, orientation. Multiple hue factors make it ideal for color-invariance tests.
- **Alternative / secondary: dSprites or Color-dSprites** — factors: shape, scale, orientation, x-position, y-position (+ color in the color variant). Lighter weight.
- Optional later: **MPI3D** for a second confirmation.

Why synthetic is the point: you know each factor's ground-truth value for every image, so recoverability is measured against a defined target, not an estimate. This is the same logical move that made the attribution-on-synthetic-data approach defensible — the ground truth is constructed, not inferred.

### 6.2 Encoders

- **SimCLR-style contrastive encoder** as the main object: backbone `f` (small ResNet or a compact CNN — keep it small for consumer GPU), projector head `g`, NT-Xent / InfoNCE loss, temperature τ.
- Train one encoder per **augmentation condition** (below). Each condition is designed to induce invariance to one targeted factor.
- **Probe the frozen encoder `f`**, not the projector `g` (the encoder is what downstream users consume). Report projector results separately as the Cosentino-style axis.

### 6.3 Augmentation conditions (the invariance "treatments")

Each condition applies one augmentation expected to induce invariance to its matching factor:

| Condition | Augmentation | Targeted factor (expected "discarded") |
|---|---|---|
| Color | hue/color jitter | object hue (and/or wall/floor hue) |
| Position | random crop / translation | x, y position |
| Orientation | rotation | orientation |
| Scale | random resized crop | scale |
| Control-aug | minimal/no augmentation | none (baseline encoder) |

Run each at ≥2 **strengths** (weak, strong) to test H-strength.

### 6.4 The probe-capacity ladder (the central instrument)

For each (encoder, targeted factor), train probes of increasing capacity to predict the factor from the frozen representation:

1. Linear probe (the field-standard measurement).
2. MLP, 1 hidden layer, small width.
3. MLP, 1 hidden layer, larger width.
4. MLP, 2 hidden layers.

Record probe performance (factor prediction accuracy for categorical factors; R²/MSE for continuous factors) at each rung. The output is a **recoverability-vs-capacity curve per factor per condition** — the binary "invariant?" becomes a measured quantity.

### 6.5 Controls (non-negotiable — this is what makes it rigorous)

- **Random-encoder control (Hewitt–Liang).** Run the identical probe ladder on a same-architecture, randomly-initialized (untrained) encoder. **Selectivity** = recoverability(trained) − recoverability(random) at each capacity. If a strong probe recovers the factor from the *random* encoder just as well, the probe is doing the work, not the representation — and your "recovery" is an artifact of the probe, which you must report honestly.
- **Supervised reference encoder.** An encoder trained with labels, as an upper-reference for how recoverable a factor can be.
- **Encoder-quality check.** Report downstream linear-probe accuracy on a held-out useful task to demonstrate the encoder is actually well-trained, so no reviewer can dismiss results as "your encoder is just bad."

---

## 7. Experimental design

A factorial sweep, fully crossed where feasible:

```
factors (5–6) × augmentation conditions (4 + control) × strengths (2)
            × probe capacities (4) × seeds (≥10)
```

- **Seeds** vary both encoder initialization and probe initialization; ≥10 per cell for confidence intervals.
- **Probe train/test discipline.** Split factor-label data into probe-train and probe-test; report probe-test performance only. Regularize probes; tune probe hyperparameters on a validation split, never on test.
- **Pre-registration.** Before running the full sweep, write down exactly which comparisons you will make and which would confirm/refute H1–H3. This protects the result's credibility and is a strong signal at student venues.

---

## 8. Metrics

- **Recoverability(F, capacity)** — probe-test performance predicting factor F at a given probe capacity (accuracy for categorical, R² for continuous).
- **Selectivity(F, capacity)** — recoverability(trained) − recoverability(random control). The real quantity of interest.
- **Capacity gap(F)** — recoverability at highest probe capacity − recoverability at linear probe. Large gap ⇒ "linear-only invariance" ⇒ artifact-prone.
- **Invariance verdict stability** — does the binary "invariant (below threshold)?" flip between linear and higher-capacity probes? Count of flips across factors/conditions is a headline number.

---

## 9. Statistical analysis

- **Bootstrap confidence intervals** on recoverability and selectivity across seeds.
- **Effect sizes with CIs** for the capacity gap (not just p-values).
- **Non-parametric tests** (e.g., Wilcoxon signed-rank across seeds) for "does capacity raise recoverability for factor F?" comparisons.
- **Multiple-comparison awareness** — you have many factor×condition cells; correct for it (e.g., Holm) and say so.
- Report everything per-cell; resist collapsing to a single average that hides the heterogeneity (the heterogeneity *is* the finding).

---

## 10. Expected outcomes (and why each is publishable)

- **Outcome A — invariance is partly a probe artifact.** Recoverability rises with capacity above the control for some factors. Contribution: a measured caution that linear-probe invariance overstates true invariance, with a map of which factors are affected.
- **Outcome B — invariance is genuinely representational.** Recoverability stays at control level across capacities. Contribution: first controlled validation of the augmentation→invariance folklore, turning assertion into evidence.
- **Outcome C (most likely) — a mixed map.** Some invariances real, some artifactual, modulated by augmentation strength. Contribution: a nuanced, reusable result the field can cite when choosing augmentations or making nuisance-removal claims.

There is no outcome that yields nothing. That is by design.

---

## 11. Threats to validity and how you preempt them

- **"Your strong probe just overfits."** → Random-encoder control + held-out probe test + regularization. Selectivity, not raw recoverability, is the headline.
- **"Synthetic ≠ real."** → Scope claims to controlled settings explicitly; add one small real-data sanity check (a natural dataset with a known nuisance factor) as a secondary, clearly-bounded result. Naming this limitation yourself is the move.
- **"It's the projector, not the encoder."** → Probe the encoder; report projector separately; cite and engage the projector-invariance literature.
- **"Your encoder is undertrained."** → Report downstream accuracy proving encoder quality before any invariance claim.
- **"Capacity and sample size are confounded."** → Hold probe-training-set size fixed across the capacity ladder.

---

## 12. Required background

**Mathematics:** linear algebra; probability and basic statistics (bootstrap, non-parametric tests, effect sizes); enough optimization to understand the contrastive objective. (You've said math is no obstacle — this is comfortably within reach.)

**Machine learning:** training small CNNs; the contrastive learning setup (positive/negative pairs, NT-Xent/InfoNCE, temperature, projector head); linear and MLP probing; calibration of an evaluation protocol. The one genuinely fiddly skill is **getting SimCLR to train stably at small scale** — plan to spend real time here before trusting any downstream number.

---

## 13. Compute and feasibility

- Consumer GPU is sufficient: small backbones, small synthetic images (64×64), modest batch sizes (use a memory-efficient contrastive setup or moderate batch + appropriate temperature).
- The cost driver is the **sweep size** (conditions × capacities × seeds), not any single run. Keep models small and the factor/condition set disciplined. If compute gets tight, cut augmentation *strengths* before cutting *seeds* — statistical credibility depends on seeds.

---

## 14. Risks and mitigations

| Risk | Level | Mitigation |
|---|---|---|
| SimCLR training instability at small scale | Medium-High | Budget 2–3 weeks up front; validate encoder quality before proceeding |
| Gap turns out already-published (Phase 0 fails) | Medium | Run Phase 0 first; have the narrowing/pivot ready (strength axis, second dataset, projector axis) |
| Sweep too large for timeline | Medium | Pre-scope factors/conditions; cut strengths not seeds |
| Result is "boringly all-real or all-artifact" | Low | Multiple factors/strengths make a flat result itself informative; design ensures *some* reportable structure |

---

## 15. Timeline (6–12 month shape; compress or expand to your calendar)

- **Weeks 1–2 — Phase 0 + foundation.** Confirmatory literature search (Section 17). Write the half-page foundation doc (Section 18). Lock H1–H3 and the pre-registration.
- **Weeks 3–5 — Encoder pipeline.** Get stable, reproducible SimCLR encoders on the synthetic data. Treat this as the gating risk; do not proceed until clean. Build the random-encoder and supervised-reference controls.
- **Weeks 6–8 — Probe instrument.** Implement the capacity ladder + selectivity computation. Validate on the control encoder first (you should see near-control recoverability).
- **Months 3–4 — Core sweep.** Run the full factorial with ≥10 seeds. Bootstrap CIs, effect sizes. Build the recoverability/selectivity maps.
- **Month 5 — Write + disseminate.** Draft the paper. Post an arXiv preprint. Submit to JEI (slow review — start early). Enter the work into your regional science fair on its own timeline.
- **Reserve (back third) — re-runs + one extension.** The inevitable re-runs, plus one stretch result (e.g., *does the recoverability profile predict downstream transfer performance?*) that becomes your "future work" and lifts the ceiling.

> **Venue calibration — read this so your internal bar stays honest.** The realistic top target for this work is a **NeurIPS/ICLR/CVPR *workshop* paper**, not a main-track paper. When you see programs advertising "our NeurIPS publications," the fine print is usually "Accepted to [Workshop Name] @ NeurIPS" — these are workshop tracks (Mech Interp, UniReps, ER, etc.), which are genuinely respected, realistic with a mentor, and the right aim for an independent high-schooler. That is a good outcome, not a consolation prize; just don't let the "NeurIPS" branding silently reset your expectation to main-track (where <0.2% of authors are high schoolers and the bar is full professional-grade work). Concretely: target a relevant workshop's call for papers + the arXiv preprint + JEI + your regional fair. Treat any main-track outcome as upside, not the plan.

---

## 16. What counts as success

- **Minimum (very likely):** a reproducible recoverability-vs-capacity map across factors and augmentations, with a clear, controlled statement about whether/when linear-probe invariance is artifactual — plus a clean, released codebase. This alone is a strong JEI/ISEF result.
- **Strong:** the above with at least one sharp, surprising finding (a factor whose "invariance" flips entirely under a small capacity increase, or one that's bulletproof across all probes), supported by the random-encoder control.
- **Exceptional:** the strong result *plus* the extension linking invariance profiles to downstream behavior, and a real-data sanity check showing the trend holds.

---

## 17. Phase 0 — confirmatory searches to run BEFORE committing

Do not write a line of training code until you've run these and confirmed the seam is open. If someone has already done the capacity-sweep-with-controls on SSL invariance, narrow (single augmentation, add the strength or projector axis) or pivot.

- `probing control tasks self-supervised representation invariance`
- `nonlinear probe recover discarded factor contrastive representation`
- `linear probe underestimates information representation selectivity`
- `augmentation invariance contrastive learning disentanglement measurement`
- Read directly: Hewitt & Liang (control tasks); Cosentino et al. (projector invariance); von Kügelgen et al. and Tian et al. (what views/augmentations encode).

Goal: confirm that the **specific** contribution — probe-capacity vs. recoverability, with random-encoder controls, on known synthetic factors, framed as an invariance-measurement question — is not already claimed.

---

## 18. Foundation document template (your immediate next action)

Fill this half page. It *is* your foundation; the title and everything else hang off it.

1. **Generative model / dataset:** which factor-controlled dataset, and the list of factors with their types (categorical/continuous).
2. **Definition of "true factor value":** the ground-truth target the probe predicts, per factor (defined by the dataset, not estimated).
3. **Definition of "invariance":** the precise statement — e.g., "the representation is ε-invariant to factor F at capacity c if probe-test performance ≤ baseline + ε." Commit to the baseline (random-encoder control) and the metric.
4. **Definition of "recoverability" and "selectivity":** the exact quantities and how computed.
5. **The probe-capacity ladder:** the specific rungs.
6. **Hypotheses H1–H3** and the comparisons that would confirm/refute each.

Send me this filled in and I'll pressure-test it like a reviewer before you spend a single GPU-hour.

---

## 19. Key references to read first (verify/expand during Phase 0)

- Hewitt & Liang — *Designing and Interpreting Probes with Control Tasks* (the methodological backbone).
- Chen et al. — *SimCLR* (the encoder you'll build).
- von Kügelgen et al. — self-supervised learning and content/style identifiability.
- Tian et al. — *What Makes for Good Views for Contrastive Learning*.
- Cosentino et al. — geometric analysis of contrastive invariance (encoder vs. projector).
- Locatello et al. — *Challenging Common Assumptions in Disentanglement* (rigor model + the "measurement assumptions matter" ethos).

> Citations here are from working knowledge and may be imprecise or dated — confirm exact authors/venues/results during Phase 0 before relying on any of them in your writeup.
