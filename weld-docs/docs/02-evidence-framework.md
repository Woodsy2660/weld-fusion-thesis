# 02 — Evidence-theoretic framework

This is the specification the fusion code implements. If code and this document
disagree, this document is correct and the code is a bug.

## Frame of discernment

**EXPERIMENTAL FRAME (4-class):**
```
Θ = {N, U, P, L}
```

| Symbol | Meaning |
|---|---|
| N | No defect; acceptable to ISO 5817 |
| U | Undercut exceeding the ISO 5817 limit (rejectable) |
| P | Porosity |
| L | Lack of fusion (planar, sidewall) |

**NOTE:** C (crack) was removed from the experimental frame because no crack
surrogate specimens could be fabricated. The 5-class frame is retained in
`configs/frame_5class.yaml` for analytical ceiling derivation.

**L means lack of fusion only.** Lack of penetration is *not* represented: the
specimens are single pass, which makes root condition a constant rather than a
controlled variable. Do not write "LoF/LoP" anywhere.

Θ has 4 elements, so 2^Θ has 16 subsets. Mass is assigned only over the focal
elements declared below, plus Θ itself.

## Per-modality focal elements

A modality assigns mass **only** over sets it can physically resolve. This is the
central design idea of the thesis, not an implementation convenience.

### Vision — three heads

| Head | Focal element | Interpretation |
|---|---|---|
| V1 | `{P}` | Surface-breaking porosity is directly visible |
| V3 | `{N,U}` | Surface-only condition; cannot see inside |
| V4 | `{N,U,L}` | Cannot exclude a subsurface planar defect |
| — | `Θ` | Ignorance (low confidence, occlusion, missing data) |

Vision **never** assigns `{U}` alone (it cannot measure depth to the ISO limit) and
**never** assigns `{L}` alone (lack of fusion is subsurface and has no reliable
surface signature). A head that was originally planned for `{C}` was removed
(precision 0.19, insufficient support).

### Geometry

| Focal element | Interpretation |
|---|---|
| `{U}` | Measured undercut exceeds the ISO 5817 limit |
| `{N,P,L}` | Profile conforms; says nothing about internals |
| `Θ` | Insufficient valid depth pixels in the cell |

### Ultrasonic

| Focal element | Interpretation |
|---|---|
| `{P}` | Volumetric scatterer signature |
| `{L}` | Planar reflector at a fusion face |
| `{N,U}` | No subsurface indication; blind to surface profile |
| `Θ` | No coupling, no scan at this cell |

**NOTE:** U3 = `{C}` was removed from the experimental frame because C is not in Θ.

## The design property

**4-class experimental frame:**
```
{N,U,L} ∩ {N,P,L} ∩ {N,U} = {N}
```

Only the three-way conjunction isolates the sound-weld hypothesis. No pair does.
This is why the system needs three modalities and not two, and the argument is
structural — it holds regardless of how good any classifier is.

## Proposition: V+G cannot isolate {L}

**Statement.** Vision and geometry together cannot isolate L (lack of fusion).
Every focal element containing L also contains N, therefore:
- BetP(L) ≤ BetP(N) for any V+G BPA
- L can never be the unique pignistic winner from V+G alone

**Proof sketch.** Every vision focal element containing L also contains N:
- V1 = `{P}` — neither L nor N
- V3 = `{N,U}` — N but not L
- V4 = `{N,U,L}` — both N and L

Every geometry focal element containing L also contains N:
- G1 = `{U}` — neither L nor N
- G2 = `{N,P,L}` — both N and L

This property (L ⟹ N) is closed under intersection and Dempster combination.
Therefore Bel({L}) ≤ Bel({N}), Pl({L}) ≤ Pl({N}), and BetP(L) ≤ BetP(N).

**Implication.** Ultrasonic testing is required to isolate L because only UT has
U2 = `{L}` as a singleton focal element. This is not a classifier limitation — it
is a fundamental property of the sensing modalities.

**Note on the 5-class frame.** The ceiling proposition also holds for the 5-class
analytical frame: V4 = `{N,U,L,C}` and G2 = `{N,P,L,C}` both contain L and N
together, and both contain C and N together. V+G cannot isolate L or C.

## Core mathematics

### Basic probability assignment

A BPA is a function m : 2^Θ → [0,1] with

```
m(∅) = 0            and            Σ_{A ⊆ Θ} m(A) = 1
```

Sets with m(A) > 0 are **focal elements**. Mass on a non-singleton set represents
evidence for "one of these, cannot say which" — not a uniform distribution over them.

### Belief and plausibility

```
Bel(A) = Σ_{B ⊆ A} m(B)          lower bound: evidence that necessarily implies A
Pl(A)  = Σ_{B ∩ A ≠ ∅} m(B)      upper bound: evidence that does not rule out A
Pl(A)  = 1 − Bel(Ā)
```

`[Bel(A), Pl(A)]` is the belief interval. Its width is the committed ignorance about
A. **Report intervals alongside point decisions** — the interval is the thing a
probabilistic classifier cannot give you.

### Dempster's rule

```
                 Σ_{B ∩ C = A} m₁(B) m₂(C)
(m₁ ⊕ m₂)(A) = ───────────────────────────        for A ≠ ∅
                          1 − K

K = Σ_{B ∩ C = ∅} m₁(B) m₂(C)                      conflict mass
```

Normalising by 1 − K is what produces Zadeh's counterexample: with near-total
conflict (K → 1) the rule amplifies a hypothesis both sources consider almost
impossible. **Keep a Zadeh regression test in the test suite.**

Always return K. Its distribution across the specimen set is a reportable result.

### Murphy averaging — the primary rule

Average the n BPAs element-wise, then combine the average with itself n − 1 times
using Dempster's rule.

```
m_avg(A) = (1/n) Σ_i m_i(A)
m_fused  = m_avg ⊕ m_avg ⊕ … ⊕ m_avg        (n − 1 combinations)
```

Murphy is primary; raw Dempster is reported as an ablation.

### Shafer discounting

Applied to every BPA before combination, using measured modality reliability.

```
m^α(A) = (1 − α) m(A)              for A ≠ Θ
m^α(Θ) = α + (1 − α) m(Θ)
```

with `α_m = 1 − balanced_accuracy_m`, measured on validation folds only.

Discounting is **load-bearing, not decorative**: measured ECE for the vision modality
has been as poor as 0.68. A modality that is badly calibrated must be discounted
hard, and the framework does that natively.

### Pignistic transform

For the point decision:

```
BetP(θ) = Σ_{A ∋ θ} m(A) / |A|
```

Mass on a set is split evenly among its members. This is a decision rule, not a
belief — do not report BetP as if it were a posterior probability.

## Execution structure

**Two phases, deliberately.**

- **Phase 1 — batched, per fold.** Model inference, calibration fitting, reliability
  estimation. These are set-level operations and must see only the data their fold
  permits.
- **Phase 2 — per cell.** BPA construction → discounting → combination → decision.
  Pure function of one cell's three BPAs plus the fold's discount factors.

Keeping them separate is what makes the ablation cheap: the seven-condition ablation
is column selection over Phase 1 output, re-running only Phase 2.

## Invariants to assert in code

1. `m(∅) == 0` on every BPA, before and after combination.
2. `abs(sum(m.values()) - 1.0) < 1e-9`.
3. Every focal element of a modality's BPA is in that modality's declared set.
4. `Bel(A) <= Pl(A)` for all A tested.
5. `Pl(A) == 1 - Bel(complement(A))` to numerical tolerance.
6. Combining a BPA with vacuous mass (`m(Θ) = 1`) returns the original unchanged.
7. Zadeh's counterexample produces the documented pathological output under raw
   Dempster and does not under Murphy.
