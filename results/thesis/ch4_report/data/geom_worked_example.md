# Geometry BPA Worked Example

## Purpose
Demonstrate Dempster's rule combination with a **hypothetical** geometry BPA.
The actual geometry pipeline uses α_G = 1.0 (clamped), meaning geometry BPA is vacuous.
These examples show what would happen with α_G = 0.3.

## Example 1: C03-02 (true class: N)

### Stored UT BPA (discounted with α_U = 0.241)
- m({P}) = 0.045
- m({L}) = 0.589
- m({N,U}) = 0.125
- m(Θ) = 0.241

### Hypothetical Geometry BPA (α_G = 0.3)
Since C03-02 is labelled N, we use m({N,P,L}) = 0.8, m({U}) = 0, m(Θ) = 0.3 as a **hypothetical** example.

### Dempster Combination
| UT \ Geom | {N,P,L} (0.8) | Θ (0.3) |
|------------|---------------|---------|
| {P} (0.045) | {P} 0.036 | {P} 0.0135 |
| {L} (0.589) | {L} 0.471 | {L} 0.1767 |
| {N,U} (0.125) | {N} 0.100 | {N,U} 0.0375 |
| Θ (0.241) | {N,P,L} 0.193 | Θ 0.0723 |

K = 0 (no conflict)

Normalised:
- m({P}) = 0.050
- m({L}) = 0.648
- m({N}) = 0.100
- m({N,U}) = 0.038
- m({N,P,L}) = 0.193
- m(Θ) = 0.072

BetP:
- P: 0.050 + 0.193/3 + 0.072/4 = 0.132
- L: 0.648 + 0.193/3 + 0.072/4 = 0.730
- N: 0.100 + 0.038/2 + 0.193/3 + 0.072/4 = 0.201
- U: 0.038/2 + 0.072/4 = 0.037

**Decision: L** (incorrect - true is N)

## Example 2: C05-02 (true class: U)

### Stored UT BPA (discounted with α_U = 0.241)
- m({P}) = 0.035
- m({L}) = 0.584
- m({N,U}) = 0.140
- m(Θ) = 0.241

### Hypothetical Geometry BPA (α_G = 0.3)
Since C05-02 is labelled U, we use m({U}) = 0.8, m({N,P,L}) = 0, m(Θ) = 0.3 as a **hypothetical** example.

(Similar calculation yields Decision: U - correct)

---
**Note**: These are HYPOTHETICAL examples. The actual system uses α_G = 1.0 (geometry is vacuous).
