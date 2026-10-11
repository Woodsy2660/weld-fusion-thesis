# 01 — Project brief

## Title

Multi-Modal Sensor Fusion for AI-Assisted Defect Classification in Structural Steel
Butt Welds.

## The problem

Post-fabrication inspection of structural steel welds relies on 100% visual
examination under AS/NZS 1554.1. Measured probability of detection for that method
averages 65% and varies between operators from 31% to 86% (Campbell et al., 2019).
The automated alternatives that fix this — radiography, phased-array ultrasonics,
industrial laser profilometry — carry capital and operator-qualification costs beyond
small-to-medium fabrication shops.

The modalities that *are* individually accessible are each blind to a distinct defect
class. A camera cannot interrogate the fusion faces. A profile measurement cannot
detect subsurface discontinuities. A volumetric acoustic method does not observe the
external bead geometry against which dimensional acceptance is judged.

**The problem is therefore one of combination, not of individual modality
performance.**

## Research question

Does evidence-theoretic fusion of surface-appearance, surface-geometry and
volumetric-acoustic evidence yield higher defect detection and classification
performance than any single modality or pairwise combination?

Three subsidiary questions:

- **RQ1** (analytical, no data required) — If each modality's measurement capability
  is declared as a set of focal elements, what does that declaration imply about the
  discriminating capability of the assembled system?
- **RQ2** (empirical) — What is the marginal contribution of each modality to fused
  performance, against ground truth on physical specimens?
- **RQ3** (empirical) — Does the fused system exceed the published human-inspector
  baseline by enough to justify use as a production screening tool?

## Scope

**In scope.** Single-V GMAW butt welds, Grade 250/350 mild steel, 10 mm plate, flat
(1G) position, single pass. Assessment against AS/NZS 1554.1 and the ISO 5817:2023
dimensional limits it invokes.

**Out of scope.** Other processes, materials, joint geometries or welding positions.
Real-time operation. Deployment hardware. A purpose-built laser profiler (specified as
costed future work).

**Positioning.** A screening tool, not a replacement for certified NDT Level II
inspection of Category SP welds.

## Deadline and consequences

Thesis due early November 2026, maximum 50 pages excluding preamble. **A hard
experimental freeze applies from 1 November** — after that date nothing new is run,
and anything unfinished is written up as a limitation.

The practical consequence for code: **prefer a working, correct, documented result
over a better one that arrives late.** If a refactor would improve the code but risks
the schedule, write down the improvement in `DECISIONS.md` as future work and move on.

## Deliverables that depend on this codebase

1. Per-cell BPA tables for all three modalities over a common cell index.
2. Fused predictions and belief intervals per cell.
3. Seven-condition ablation table with significance tests.
4. Algebraic ceiling per modality subset (derived, not measured).
5. Conflict mass K distribution.
6. Per-coupon defect maps along the weld axis.
7. Predicted versus measured geometric uncertainty.
8. Sequence-leakage audit table.
