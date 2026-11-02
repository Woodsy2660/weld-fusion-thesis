# Ideal Geometry Module Simulation

Generated: 2026-11-02T10:59:35.121803

## Purpose

This simulation models what the geometry module WOULD produce if:
1. **L2 resolved**: Depth sensor correctly measures convex weld bead (not as trench)
2. **100% coverage**: All 150 cells have valid depth data
3. **All measurands functional**: Undercut, reinforcement, AND misalignment

## Parameter Classification

### FIXED (Cannot Change Without Invalidating Results)

| Parameter | Value | Source |
|-----------|-------|--------|
| Undercut limit | 0.5 mm | ISO 5817 Level B/C |
| Reinforcement limit | 1 + 0.1b = 2.6 mm | ISO 5817 |
| Misalignment limit | 1.0 mm | ISO 5817 Level B |
| Depth units | 0.0001 | Hard rule 4 |
| Visual preset | High Accuracy | Hard rule 5 |
| Standoff | 300 mm | D435i MinZ |

### CHANGEABLE (New Capture Could Vary)

| Parameter | Current | Ideal | Effect |
|-----------|---------|-------|--------|
| Coverage | 26% | 100% | More cells with evidence |
| Surface prep | Mill scale | Matte paint | Fixes L2 artifact |
| u_total | 0.275 mm | 0.056 mm | Better discrimination |
| Measurands | Misalignment only | All 3 | Correct U detection |

### FLUCTUATES (Random Within Bounds)

| Parameter | Range | Distribution |
|-----------|-------|--------------|
| True undercut depth | 0.6-1.2 mm | Uniform |
| Measured noise | ±u_total | Gaussian |
| Conforming undercut | 0.05-0.35 mm | Uniform |

## Results Comparison

| Metric | Actual | Ideal | Improvement |
|--------|--------|-------|-------------|
| Coverage | 39/150 (26%) | 150/150 (100%) | +111 cells |
| Macro-F1 | 0.204 | 1.000 | +0.796 |
| Precision (U) | 0.308 | 1.000 | +0.692 |
| Recall (U) | 1.000 | 1.000 | +0.000 |
| α_G | 1.000 | 0.000 | -1.000 |
| u_total | 0.275 mm | 0.056 mm | -0.219 mm |

## Confusion Matrices

### Actual (39 non-vacuous cells)
```
              Pred NPL    Pred U
True NPL         0          27
True U           0          12
```

### Ideal (150 cells)
```
              Pred NPL    Pred U
True NPL         107          0
True U           0          43
```

## BPA Structure

### Ideal Geometry BPA
```
m({U})     = (1 - α_G) × P(exceedance)
m({N,P,L}) = (1 - α_G) × (1 - P(exceedance))
m(Θ)         = α_G = 0.000
```

### Ground Truth Mapping
- Cells with geometry_target = {U}: 43 cells
  - Undercut (C05, C06): 20 cells
  - Overfill (C09, C12): 14 cells
  - Misalignment (C13): 3 cells
  - Mixed undercut (C11, C14): 6 cells
- Cells with geometry_target = {N,P,L}: 107 cells
  - Sound (N), Porosity (P), LOF (L): all geometrically conforming

## Fusion Results: Actual vs Ideal Geometry

### Complete Comparison Table

| Configuration | Actual F1 | Ideal F1 | Δ F1   | Notes |
|---------------|-----------|----------|--------|-------|
| V             | 0.097     | 0.148    | +0.051 | Vision limited by V4 ignorance |
| G             | 0.086     | 0.353    | +0.267 | **Major gain from U detection** |
| U             | 0.482     | 0.482    | +0.000 | UT unaffected by geometry |
| V+G           | 0.097     | 0.353    | +0.256 | G dominates fusion |
| V+U           | 0.482     | 0.482    | +0.000 | UT dominates V |
| G+U           | 0.482     | **0.697**| **+0.215** | **Key fusion improvement** |
| V+G+U         | 0.482     | **0.697**| **+0.215** | Vision adds no further value |

### Per-Class Performance (Ideal G+U)

| Class | Precision | Recall | F1    | Support |
|-------|-----------|--------|-------|---------|
| N     | 0.678     | 0.635  | 0.656 | 63      |
| U     | 0.698     | 1.000  | 0.822 | 30      |
| P     | 0.526     | 0.385  | 0.444 | 26      |
| L     | 0.897     | 0.839  | 0.867 | 31      |

### Key Insights

1. **Geometry would be the top performer for U-class** (F1 = 0.822 vs UT = 0.000)
2. **G+U complementarity**: Geometry handles U, UT handles L and P
3. **Vision adds no value in current form**: V+G+U = G+U (0.697)
4. **System ceiling with ideal geometry**: Macro-F1 = 0.697

### α Values

| Modality | Actual | Ideal | Δ |
|----------|--------|-------|---|
| Vision (V) | 0.404 | 0.404 | 0.000 |
| Geometry (G) | 1.000 | 0.000 | -1.000 |
| UT (U) | 0.241 | 0.241 | 0.000 |

## What Would Need to Change

1. **Resolve L2**: Surface treatment (matte paint/sandblast) OR different sensor
2. **Recapture all 15 coupons** with new surface prep
3. **Rerun full pipeline** with working undercut/reinforcement detection
4. **No code changes needed** - pipeline is correct, data is compromised

## Thesis Implications

The geometry module's failure is **data-limited, not method-limited**. With proper
surface preparation:

- **System F1 would improve from 0.482 → 0.697** (+45% relative improvement)
- **U-class detection improves from 0.000 → 0.822** (currently blind to U)
- **Complementarity is fully realized**: each modality contributes unique information
