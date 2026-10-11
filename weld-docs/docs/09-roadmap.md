# 09 — Roadmap

Target submission early November 2026. Dates assume a 5 November deadline; shift
uniformly if that is wrong.

## Sequencing principle

Order by **irreversibility**, not by importance. The instrument hire and the
couplant contamination are one-way doors. Everything else is software that can be
rerun at 2am. So the physical work is front-loaded and the first week contains no
writing.

## Week 1 — 9 to 15 Oct: acquisition

Nothing in this week can be redone.

| Days | Task |
|---|---|
| Thu 9 – Fri 10 | Collect coupons. Measure actual lengths. Stamp IDs. Drill datum holes. Punch-mark zone boundaries. |
| Thu 9 – Fri 10 | **Clean vision + depth capture, both faces, all 15 coupons.** Record USB type, firmware, SDK version, ASIC temperature. |
| Thu 9 – Sun 12 | In parallel: UT feature-extractor dry run on synthetic A-scans. Index sheet printed. Step block to the workshop. |
| Sat 11 – Sun 12 | Machine the notches. Build the cell-level label file from the parameter log. |
| Mon 13 – Wed 15 | **Ultrasonic scanning.** Belt-and-braces capture at every position. |

**Ordering constraint:** datum holes → clean optical capture → notch machining → UT.
The notches damage the surface and the couplant contaminates it, so optical capture
goes first.

## Week 2 — 16 to 22 Oct: three modalities to per-cell BPAs

| Days | Task |
|---|---|
| Thu 16 – Fri 17 | Geometry pipeline: 64-frame averaging, plane fit, dimensional extraction, step-block characterisation. Report predicted vs measured u_total. |
| Fri 17 – Sun 19 | UT: features from captured scans, Random Forest under nested GroupKFold, isotonic calibration. |
| Sat 18 – Tue 21 | Vision: L→V4 remap, single-source retrain, initialisation ablation on real coupons. Overnight GPU runs. |
| **Wed 22** | **Gate: three per-cell BPA tables over the same 150 cells, same index.** |
| Throughout | Write the specimen-manufacture and acquisition sections — these need no results. |

Rationale for this order within the week: geometry first because it has no training
loop and will surface registration problems while there is still time to re-scan; UT
second because if the waveform export failed you need to know immediately to fall
back to the BMPs; vision last of the three because it is the one that runs unattended.

**If the Wednesday gate is missed, the fusion result is at risk.** Start writing the
reduced-scope version rather than pushing on.

## Week 3 — 23 to 29 Oct: fusion, ablation, statistics

| Days | Task |
|---|---|
| Thu 23 – Fri 24 | Phase 2: BPA → discount → Murphy/Dempster → pignistic. Zadeh regression test. |
| Sat 25 – Sun 26 | Seven-subset ablation, empirical vs algebraic ceiling. Conflict K distribution. |
| Mon 27 | McNemar + Holm–Bonferroni on the declared family. Clustered bootstrap CIs. Measured ρ and d_eff. |
| Tue 28 – Wed 29 | Defect maps, 2.5D output figures, results tables. Write the results chapter. |

Fusion is last because it depends on all three modalities, but it is also fast — it
is column selection over tables that already exist. The ablation gets protected days
because it is the contribution.

## Week 4 — 30 Oct to 5 Nov: writing only

| Days | Task |
|---|---|
| Thu 30 – Sat 1 | Discussion and conclusions. All declared limitations. |
| **Sat 1 Nov** | **Hard experimental freeze. No new results after this date.** |
| Sun 2 – Tue 4 | Literature review compression. Abstract. Cross-references, bibliography. |
| Wed 5 | Full read-through, page count against the 50-page limit, submit. |

The freeze exists because the most common way a good honours thesis becomes a
mediocre one is a student still generating results in the final week and writing them
up badly. A clean negative result described well scores better than a good result
described in a rush.

## What has been cut, and stays cut

| Cut | Replacement |
|---|---|
| Three-stage public-corpus retrain | Collapsed to the initialisation ablation |
| DINOv2 few-shot comparison | One paragraph of future work |
| XGBoost UT comparison | Random Forest only; future work |
| Exhaustive destructive sectioning | Sampled only, to quantify label noise |
| Radiography (if not already booked) | Removed from the objective |

Do not reintroduce any of these without an explicit decision recorded in
`DECISIONS.md`.

## Results that are already secure

Worth knowing, because they mean the thesis is not at risk even if fusion
underperforms:

1. The sequence-leakage audit — a genuine methodological finding.
2. The two-modality ceiling proposition — analytical, no data can contradict it.
3. The predicted-versus-measured uncertainty comparison — a day's work once the step
   block exists.

Protect these first.
