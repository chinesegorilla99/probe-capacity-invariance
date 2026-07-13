# Preregistration — Probe Capacity & Contrastive Invariance

> **STATUS: 🔒 LOCKED 2026-06-28.** Hypotheses and operational definitions below are frozen. Changes only via a dated **Amendment** appended at the bottom — never silent edits (per ResearchOverview §7/§17).
> **Provenance:** Socratic foundation → Phase 0 `deep-research` lit-review ([novelty verdict]) → repositioning → independent methods red-team (GO-WITH-CHANGES, 6 fixes adopted). Working notes: `lab-notebook/{foundation-summary,definitions,phase0/}`.
> **Scope:** claims are restricted to the controlled synthetic-factor setting; one bounded real-data sanity check is secondary and clearly labelled.

---

## 0. Locked design parameters

- **Datasets:** **Shapes3D primary**, **dSprites secondary**. (MPI3D optional later confirmation only.)
- **Factors + types (Shapes3D):** floor hue, wall hue, object hue, scale, orientation = **continuous** (R²); shape = **categorical** (normalized accuracy). dSprites supplies **x/y position** (continuous) for the position arm.
- **Headline contrast:** object hue (color-jitter, expected-genuine) vs. x/y position (random-crop, expected-artifact).
- **Augmentation conditions × strengths:** Color (hue/color jitter), Position (crop/translation), Orientation (rotation), Scale (random-resized-crop), + Control-aug (minimal); each at **≥2 strengths** (weak, strong).
- **Probe-capacity ladder:** (1) linear · (2) MLP 1-hidden small · (3) MLP 1-hidden large · (4) MLP 2-hidden. **Monotone capacity; capacity reported as an explicit measure (param count / effective DOF) per rung.** Exact widths fixed at probe-build (not a frozen quantity).
- **Seeds:** **≥10 per cell**, seed controls **both** encoder and probe init; bootstrap over seeds. Cell = `(factor, condition, strength, probe_capacity, seed)`.
- **Probe discipline:** probe-train / val / probe-test split; **probe-train size held FIXED across the ladder**; regularization tuned on val, **never** on test; report probe-test only.

## 1. Generative model / dataset
Factor-controlled synthetic datasets where each image's generative factors are known **by construction**. Shapes3D (6 factors above) primary; dSprites (shape, scale, orientation, x, y) secondary.

## 2. "True factor value"
The ground-truth generative factor value for each image, **defined by the dataset's construction, not estimated**. Each factor is the probe's prediction target.

## 3. Recoverability, encoder gain, probe selectivity (the metric layer)

- **Recoverability `R(F,c)`** — probe-**test** performance predicting factor F from the **frozen encoder** at capacity c, on a **normalized [0,1]** scale: categorical → `(acc−chance)/(1−chance)`; continuous → `R²`. **R²<0 retained, not clipped** (clipping biases G upward). ⚠️ orientation is cyclic → circular target / sin–cos at probe-build.
- **Probe selectivity `S(F,c) = R(real labels) − R(random labels)`** — Hewitt–Liang **random-LABEL** control task. Bounds the *probe* (S≈0 ⇒ probe is memorizing).
- **Encoder gain `G(F,c) = R(trained encoder) − R(random ENCODER)`** — matched-architecture untrained baseline. Bounds *representation-free* structure. **G is the headline metric.**
- **Commensurability (FIX 1):** normalized-accuracy and R² are different units. **G is interpreted per-factor** against that factor's own ε_G; cross-factor comparisons restricted to **within-type** factor pairs — R² and normalized-accuracy are never pooled or rank-compared.
- **Capacity gap `Δ_G(F) = G(F, top rung) − G(F, linear)`.**

## 4. "Invariance" (ε-invariance) — the dual-gate

- **Invariance verdict boolean (also the flip-count boolean):** trained encoder is **ε-invariant to F at capacity c iff `G(F,c) ≤ ε_G`**.
- **Genuine recovery requires BOTH gates:** `G(F,c) > ε_G` **AND** `S(F,c) > 0`. Three exhaustive cases (FIX 3):
  - `G ≤ ε_G` → **invariant**; the sub-case `G < 0` ("below-random / actively suppressed") is reported **distinctly**, not merged with "indistinguishable from random."
  - `G > ε_G, S > 0` → **genuine recovery** (a *linear-invariance artifact* when it appears only above the linear rung).
  - `G > ε_G, S ≤ 0` → **inconclusive / probe-driven** (dead zone): not genuine recovery; flagged separately. The invariance boolean keys on G alone, so the flip-count is well-defined.
- **ε_G primary = data-derived per-factor:** upper bound of the bootstrap 95% CI of G under the **random-vs-random-encoder null** (captures **encoder-init noise only**). Capacity-dependent probe over-fitting is controlled by **S**; **both gates use the same paired seed bootstrap**; the AND-gate is a conjunction of two one-sided tests, each component reported. **Fixed ε_G = 0.05 normalized = sensitivity analysis only** (FIX 2, 5).

## 5. Controls
- **Random-LABEL control task** → defines S (bounds probe).
- **Random-ENCODER control** (matched arch, untrained) → defines G (bounds representation-free structure).
- **Supervised-reference encoder** → recoverability ceiling; **diagnostic-only** (not in any decision rule).
- **Encoder-quality gate** → downstream linear-probe accuracy proving the encoder is well-trained **before** any invariance claim.

---

## 6. Hypotheses (LOCKED) + confirm/refute rules

All tests on **encoder gain G** with **probe selectivity S as the recovery gate**, **≥10 seeds**, bootstrap CIs, **Wilcoxon signed-rank** for paired capacity/factor comparisons, effect sizes with CIs, **Holm** correction across factor×condition cells. Probe-test only.

- **H1 (capacity dependence).** G(F,·) is non-decreasing in capacity and rises materially above linear for ≥1 factor.
  - **Confirm:** capacity gap `Δ_G(F)` bootstrap CI **> 0 and > ε_G** for ≥1 factor (with S>0 at the top rung). **Refute:** all `Δ_G` CIs include 0 or stay ≤ ε_G.
- **H2 (heterogeneity).** The capacity effect differs across factors.
  - **Confirm:** **paired per-seed difference** of `Δ_G` for ≥1 **within-type** factor pair has a Wilcoxon-significant / bootstrap-CI-excludes-0 difference (Holm across pairs). **Refute:** all within-type difference CIs include 0. *(FIX 4 — replaces the earlier non-overlapping-CI test.)*
- **H3 (genuine invariance via control).** Some factors stay invariant at every capacity.
  - **Confirm (existence):** ≥1 factor has `G(F,c) ≤ ε_G` at **every** rung (selectivity CI within ±ε_G across the ladder). **Refute:** no factor stays ε_G-invariant across all capacities.
- **H4 (encoder-vs-projector localization — a TEST of Cosentino et al. 2022, not a discovery).** Invariance concentrates in the projector.
  - **Confirm:** paired **`G(encoder) − G(projector) > 0`** for targeted factors (Wilcoxon, Holm), **widening with augmentation strength**. **Refute:** difference CI includes 0 / negative.

**Headline number — verdict-stability flip count:** # of (factor, condition) cells where the boolean `G ≤ ε_G` changes between the linear probe and the top rung; reported for **both** primary per-factor ε_G and the 0.05 sensitivity ε_G.

## 7. Analysis plan (LOCKED)
- Bootstrap CIs on G, S, and Δ_G across seeds; **paired seed resampling** shared by the G and S gates.
- Effect sizes with CIs for the capacity gap (not just p-values).
- Wilcoxon signed-rank across seeds for all paired comparisons (H1 capacity, H2 within-type differences, H4 encoder−projector).
- **Holm** correction across the factor×condition cell family; state the family explicitly.
- Report **per-cell**; never collapse to a single average that hides heterogeneity (the heterogeneity is the finding).
- **Power caveat (pre-registered):** Wilcoxon at n=10 has limited power for small effects after Holm; a pilot power check may raise seeds before the full sweep.
- **AI-assistance disclosure:** literature search and drafting used AI tools; recorded for transparency.

## 8. Deferred to later phases (NOT frozen quantities)
Exact probe widths (probe-build); the capacity measure plotted against G (param count vs. effective DOF — pre-register at probe-build, engaging Lee & Kondor 2026); orientation circular-regression handling; real-data sanity-check dataset.

## 9. Key references (verified Phase 0)
Hewitt & Liang 2019 (control tasks); Pimentel et al. 2020 (info-theoretic probing — argues *for* powerful probes, counterpoint to H&L); Lee & Kondor 2026 (arXiv:2605.11448, principled probe hierarchy); Foster, Pukdee & Rainforth 2021 (ICLR; arXiv:2010.09515, linear leakage of transform params); Cosentino, Sengupta, Avestimehr, Soltanolkotabi, Ortega, Willke & Tepper 2022 (arXiv:2205.06926, encoder/projector — basis of H4); Chen et al. 2020 (SimCLR); von Kügelgen et al. 2021; Tian et al. 2020; Locatello et al. 2019; HaoChen et al. 2021; Balestriero & LeCun 2022.

---

## Amendments

### A1 — 2026-07-13 — Realized grid (strong cross-section); confirm-rule clarifications; added reporting

**Unblinding status at the time of this amendment:** all 36 first-slice encoders were trained (color / control / position × strong × 12 seeds); probe recoverability stacks existed for the color/strong cell only (seeds 0–9, computed 2026-07-11 — superseded by the planned 12-seed re-probe); the H1–H4 statistics layer, flip counts, and figures had **not** been run on any real cell. Decision-procedure validation to date used hand-set synthetic fixtures only (`results/_synthetic/`, tagged `synthetic_world`).

**(a) Scope — realized grid.** Compute limits curtail the §0 condition × strength grid to the **strong cross-section**: Color, Position, Orientation, Scale, and Control-aug at strong strength, ≥12 seeds per cell (5 cells, 60 encoders). The weak strength axis is deferred to a follow-up study. Consequences, accepted in advance:
- **H4's confirm rule (encoder−projector gap widening with augmentation strength) is not evaluable at a single strength.** H4 is reported as its sign component only — paired `G(encoder) − G(projector)` at strong, Wilcoxon + Holm per §6 — with status **"partial" (descriptive)**. No confirmation or refutation of H4 is claimed.
- Claims are scoped to the realized cells; no generalization across augmentation strengths.
- All Holm families span the realized cells only; family membership is stated explicitly in the report.
- H1, H2, H3, both controls, and the headline flip count are unaffected.
- The §Scope secondary real-data sanity check remains deferred; if unrun at submission, it is omitted from the paper's claims rather than approximated.

**(b) Clarifying pins (implementation readings pinned 2026-07-04, before any real verdict; text now matches the implementation).**
- **H1:** the "> ε_G" threshold for the capacity gap `Δ_G` uses the same CI-of-mean estimator applied to the random-vs-random null **of the capacity gap itself** (the random stack's own top-minus-linear gap), not the top-rung level ε_G.
- **H3:** confirm keys on the §4 point boolean `G ≤ ε_G` at every rung; the §6 parenthetical band ("CI within ±ε_G") is co-reported as a diagnostic, **not** a gate; the suppressed sub-case (`G < 0`) remains reported distinctly.

**(c) Additional reporting (additive only; no decision rule changes).**
- **Flip-count uncertainty:** seed-bootstrap resampling of the paired per-seed G stack at *fixed* ε thresholds — per-factor flip fractions and a 95% interval on the headline count — co-reported for both the primary per-factor ε_G and the fixed-0.05 sensitivity ε_G. Threshold (ε) uncertainty is carried separately by the ε_G diagnostics, not folded into this bootstrap.
- **Absolute recoverability levels:** `R(trained)`, `R(random-encoder floor)`, and `R(projector)` (mean + bootstrap CI per factor × rung) are co-reported next to G in every per-cell report and figure table, so an "invariant (`G ≤ ε_G`)" verdict over a near-ceiling random floor is never read as "the factor is absent from the representation."
