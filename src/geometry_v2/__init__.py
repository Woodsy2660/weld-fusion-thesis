"""
Geometry V2: Asymmetry-based BPA construction workaround.

This module implements an alternative geometry BPA construction that uses
plate asymmetry as a discriminative feature for LOF detection, addressing
the limitation that centerline depth has SNR=0.23 (too low).

Key insight: Plate asymmetry (SNR=1.75) discriminates between LOF and non-LOF
conditions, but cannot distinguish LOF from misalignment. Therefore, high
asymmetry maps to the compound focal element {L,U}.

New focal element structure:
    G1: {U}     ISO 5817 exceedance (reinforcement/misalignment)
    G2: {N,P,L} Profile conforms
    G3: {L,U}   High plate asymmetry (LOF or misalignment)

The original geometry module (src/geometry/rules.py) is preserved unchanged.
"""

__version__ = "2.0.0"
