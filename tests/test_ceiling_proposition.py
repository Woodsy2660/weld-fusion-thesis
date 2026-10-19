"""
Tests for the ceiling proposition: V+G cannot isolate {L}.

The proposition:
    In the 4-class frame (Theta = {N, U, P, L}), the fusion of Vision and
    Geometry modalities cannot isolate {L} (lack of fusion). This is because
    every focal element containing L also contains N:

        Vision:   V4 = {N, U, L}   (L ⟹ N)
        Geometry: G2 = {N, P, L}   (L ⟹ N)

    Therefore:
        Bel({L}) ≤ Bel({N})
        Pl({L}) ≤ Pl({N})
        BetP(L) ≤ BetP(N)

    This means L can never be the unique pignistic winner from V+G alone.
    The Ultrasonic modality is REQUIRED to isolate L because only UT has
    U2 = {L} as a singleton focal element.

This limitation is fundamental to the sensing modalities, not a classifier
flaw. It justifies the three-modality fusion architecture.
"""
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.utils.frame import load_frame  # noqa: E402


class TestCeilingProposition:
    """Test the V+G ceiling proposition algebraically."""

    @pytest.fixture
    def frame(self):
        """Load the 4-class experimental frame."""
        return load_frame()

    def test_every_vision_element_containing_L_also_contains_N(self, frame):
        """Vision: L ⟹ N. This is the core structural constraint."""
        vision_heads = frame.heads_for("vision")
        for head, elements in vision_heads.items():
            els = set(elements)
            if "L" in els:
                assert "N" in els, (
                    f"Vision head {head} = {els} contains L without N. "
                    f"This would allow vision to isolate L, violating the ceiling."
                )

    def test_every_geometry_element_containing_L_also_contains_N(self, frame):
        """Geometry: L ⟹ N. This is the core structural constraint."""
        geometry_heads = frame.heads_for("geometry")
        for head, elements in geometry_heads.items():
            els = set(elements)
            if "L" in els:
                assert "N" in els, (
                    f"Geometry head {head} = {els} contains L without N. "
                    f"This would allow geometry to isolate L, violating the ceiling."
                )

    def test_ut_has_singleton_L(self, frame):
        """UT provides U2 = {L}, which is required to isolate L."""
        ut_heads = frame.heads_for("ut")
        singleton_L_found = False
        for head, elements in ut_heads.items():
            els = set(elements)
            if els == {"L"}:
                singleton_L_found = True
                break
        assert singleton_L_found, (
            "UT must have a singleton {L} head to isolate L. "
            f"Found heads: {ut_heads}"
        )

    def test_L_implication_holds_for_all_VG_focal_elements(self, frame):
        """
        For every focal element A in V ∪ G:
            L ∈ A ⟹ N ∈ A

        This is the necessary and sufficient condition for V+G not being
        able to isolate L via Dempster combination.
        """
        vision_heads = frame.heads_for("vision")
        geometry_heads = frame.heads_for("geometry")

        all_focals = list(vision_heads.values()) + list(geometry_heads.values())

        for focal in all_focals:
            els = set(focal)
            if "L" in els:
                assert "N" in els, (
                    f"Focal element {els} contains L but not N. "
                    f"This would violate the ceiling proposition."
                )

    def test_belief_L_cannot_exceed_belief_N(self, frame):
        """
        Algebraic proof: Bel({L}) ≤ Bel({N}) for any V+G BPA.

        By definition: Bel(A) = Σ_{B⊆A} m(B)

        Since every focal element containing L also contains N:
            {B : B ⊆ {L}} ⊆ {B : B ⊆ {N}}

        Therefore the mass contributing to Bel({L}) is a subset of the mass
        contributing to Bel({N}), so Bel({L}) ≤ Bel({N}).
        """
        # This is a structural proof, not a numerical test.
        # We verify the premise: every focal element that is a subset of {L}
        # must also be a subset of {N} (which is impossible since L ≠ N).
        # Actually, the only subset of {L} is {L} itself and ∅.
        # So Bel({L}) = m({L}) + m(∅) = m({L}).
        # Since no V or G head is exactly {L}, m({L}) = 0 for V+G.

        vision_heads = frame.heads_for("vision")
        geometry_heads = frame.heads_for("geometry")

        all_focals = [frozenset(v) for v in vision_heads.values()]
        all_focals += [frozenset(v) for v in geometry_heads.values()]

        # No V or G focal element is exactly {L}
        assert frozenset({"L"}) not in all_focals, (
            "V or G has a singleton {L} focal element, "
            "which would allow Bel({L}) > 0"
        )

    def test_plausibility_L_cannot_exceed_plausibility_N(self, frame):
        """
        Algebraic proof: Pl({L}) ≤ Pl({N}) for any V+G BPA.

        By definition: Pl(A) = Σ_{B∩A≠∅} m(B)

        Since every focal element containing L also contains N:
            {B : B ∩ {L} ≠ ∅} ⊆ {B : B ∩ {N} ≠ ∅}

        Any focal element intersecting {L} also intersects {N}.
        Therefore Pl({L}) ≤ Pl({N}).
        """
        vision_heads = frame.heads_for("vision")
        geometry_heads = frame.heads_for("geometry")

        all_focals = [frozenset(v) for v in vision_heads.values()]
        all_focals += [frozenset(v) for v in geometry_heads.values()]

        for focal in all_focals:
            # If focal intersects {L}, it must also intersect {N}
            if "L" in focal:
                assert "N" in focal, (
                    f"Focal element {set(focal)} contains L but not N. "
                    f"Pl({{L}}) could exceed Pl({{N}})."
                )


class TestPignisticTransform:
    """Test pignistic transform properties for the ceiling proposition."""

    @pytest.fixture
    def frame(self):
        return load_frame()

    def test_pignistic_L_cannot_exceed_pignistic_N(self, frame):
        """
        BetP(L) ≤ BetP(N) for any V+G BPA.

        The pignistic transform is:
            BetP(θ) = Σ_{A∋θ} m(A) / |A|

        Since every focal element A containing L also contains N:
            - Every term m(A)/|A| that contributes to BetP(L) also
              contributes to BetP(N)
            - Additional terms may contribute to BetP(N) but not BetP(L)
              (e.g., G1={U} contributes 0 to both, but V3={N,U} contributes
              to BetP(N) but not BetP(L))

        Therefore BetP(L) ≤ BetP(N).
        """
        vision_heads = frame.heads_for("vision")
        geometry_heads = frame.heads_for("geometry")

        all_focals = [frozenset(v) for v in vision_heads.values()]
        all_focals += [frozenset(v) for v in geometry_heads.values()]

        # Every focal element containing L also contains N
        for focal in all_focals:
            if "L" in focal:
                assert "N" in focal, (
                    f"Focal {set(focal)} violates L ⟹ N"
                )

        # Additionally, some focal elements contain N but not L
        # (e.g., V1={P}, V3={N,U}, G1={U})
        # This means BetP(N) can be strictly greater than BetP(L)
        focals_with_N_not_L = [f for f in all_focals if "N" in f and "L" not in f]
        assert len(focals_with_N_not_L) > 0, (
            "Expected at least one focal element containing N but not L"
        )


class TestNumericalExample:
    """Numerical test that BetP(L) < BetP(N) for a sample V+G BPA."""

    def test_betp_example(self):
        """
        Compute BetP for a sample V+G fusion and verify BetP(L) < BetP(N).

        Example masses (post-Dempster combination, simplified):
            m({P}) = 0.1        [from V1]
            m({N,U}) = 0.2      [from V3]
            m({N,U,L}) = 0.3    [from V4]
            m({U}) = 0.1        [from G1]
            m({N,P,L}) = 0.2    [from G2]
            m(Θ) = 0.1          [ignorance floor]

        BetP calculation (Θ = {N,U,P,L}, |Θ|=4):
            BetP(N) = m({N,U})/2 + m({N,U,L})/3 + m({N,P,L})/3 + m(Θ)/4
                    = 0.2/2 + 0.3/3 + 0.2/3 + 0.1/4
                    = 0.1 + 0.1 + 0.0667 + 0.025
                    = 0.2917
            BetP(L) = m({N,U,L})/3 + m({N,P,L})/3 + m(Θ)/4
                    = 0.3/3 + 0.2/3 + 0.1/4
                    = 0.1 + 0.0667 + 0.025
                    = 0.1917
            BetP(U) = m({N,U})/2 + m({N,U,L})/3 + m({U})/1 + m(Θ)/4
                    = 0.2/2 + 0.3/3 + 0.1/1 + 0.1/4
                    = 0.1 + 0.1 + 0.1 + 0.025
                    = 0.325
            BetP(P) = m({P})/1 + m({N,P,L})/3 + m(Θ)/4
                    = 0.1/1 + 0.2/3 + 0.1/4
                    = 0.1 + 0.0667 + 0.025
                    = 0.1917

        Sum: 0.2917 + 0.1917 + 0.325 + 0.1917 = 1.0001 ≈ 1.0 ✓

        Key result: BetP(L) = 0.1917 < BetP(N) = 0.2917 ✓
        """
        # Focal elements (as frozensets)
        theta = frozenset({"N", "U", "P", "L"})

        # Sample BPA (these are valid V+G focal elements)
        bpa = {
            frozenset({"P"}): 0.1,           # V1
            frozenset({"N", "U"}): 0.2,      # V3
            frozenset({"N", "U", "L"}): 0.3, # V4
            frozenset({"U"}): 0.1,           # G1
            frozenset({"N", "P", "L"}): 0.2, # G2
            theta: 0.1,                       # ignorance
        }

        # Verify masses sum to 1
        assert abs(sum(bpa.values()) - 1.0) < 1e-9

        # Compute pignistic transform
        betp = {cls: 0.0 for cls in theta}
        for focal, mass in bpa.items():
            card = len(focal)
            contribution = mass / card
            for cls in focal:
                betp[cls] += contribution

        # Verify BetP sums to 1
        assert abs(sum(betp.values()) - 1.0) < 1e-9

        # The key assertion: BetP(L) < BetP(N)
        assert betp["L"] < betp["N"], (
            f"BetP(L)={betp['L']:.4f} should be less than BetP(N)={betp['N']:.4f}"
        )

        # In fact, BetP(L) cannot exceed BetP(N) for ANY V+G BPA
        # because the structural constraint L ⟹ N holds for all focal elements.

        # Print results for documentation
        print(f"\nPignistic probabilities:")
        for cls in sorted(betp):
            print(f"  BetP({cls}) = {betp[cls]:.4f}")
        print(f"\nBetP(L) = {betp['L']:.4f} < BetP(N) = {betp['N']:.4f} ✓")


class TestBetPCardinality:
    """
    BetP cardinality regression: mass on {N,U,L} must split three ways.

    This is the specific test requested by Task 0 item 5: confirm that a
    vision BPA with mass on {N,U,L} correctly assigns 1/3 to each of N, U, L.
    """

    def test_mass_on_NUL_splits_three_ways(self):
        """
        A vision BPA with mass only on V4 = {N,U,L} must split equally.

        BetP(θ) = Σ_{A∋θ} m(A) / |A|

        For m({N,U,L}) = 0.9, m(Θ) = 0.1:
            BetP(N) = 0.9/3 + 0.1/4 = 0.325
            BetP(U) = 0.9/3 + 0.1/4 = 0.325
            BetP(L) = 0.9/3 + 0.1/4 = 0.325
            BetP(P) = 0.1/4 = 0.025

        The key assertion: mass on {N,U,L} gives equal BetP to N, U, and L.
        """
        theta = frozenset({"N", "U", "P", "L"})
        nul = frozenset({"N", "U", "L"})

        # Vision BPA with mass only on V4
        bpa = {
            nul: 0.9,
            theta: 0.1,
        }

        # Compute BetP
        betp = {cls: 0.0 for cls in theta}
        for focal, mass in bpa.items():
            for cls in focal:
                betp[cls] += mass / len(focal)

        # Verify sum to 1
        assert abs(sum(betp.values()) - 1.0) < 1e-9

        # The three-way split assertion
        expected_nul = 0.9 / 3 + 0.1 / 4  # = 0.325
        expected_p = 0.1 / 4              # = 0.025

        assert abs(betp["N"] - expected_nul) < 1e-9, f"BetP(N) = {betp['N']}"
        assert abs(betp["U"] - expected_nul) < 1e-9, f"BetP(U) = {betp['U']}"
        assert abs(betp["L"] - expected_nul) < 1e-9, f"BetP(L) = {betp['L']}"
        assert abs(betp["P"] - expected_p) < 1e-9, f"BetP(P) = {betp['P']}"

        # The ceiling still holds
        assert betp["L"] <= betp["N"], "BetP(L) should not exceed BetP(N)"

        print(f"\nThree-way split verified:")
        print(f"  BetP(N) = BetP(U) = BetP(L) = {betp['N']:.4f}")
        print(f"  BetP(P) = {betp['P']:.4f}")


class TestCeilingUnderOperations:
    """
    Verify that BetP(L) ≤ BetP(N) is preserved by DS operations.

    Task 0 item 6: extend ceiling test coverage for:
    - Shafer discounting
    - Murphy averaging of discounted BPAs
    - Dempster self-combination

    Both operations preserve "L ∈ A ⇒ N ∈ A", but we test rather than assert.
    """

    @pytest.fixture
    def sample_vg_bpa(self):
        """Sample V+G BPA satisfying the L ⟹ N constraint."""
        theta = frozenset({"N", "U", "P", "L"})
        return {
            frozenset({"P"}): 0.1,           # V1
            frozenset({"N", "U"}): 0.2,      # V3
            frozenset({"N", "U", "L"}): 0.3, # V4
            frozenset({"U"}): 0.1,           # G1
            frozenset({"N", "P", "L"}): 0.2, # G2
            theta: 0.1,                       # Θ
        }

    def _compute_betp(self, bpa):
        """Compute pignistic transform."""
        theta = frozenset({"N", "U", "P", "L"})
        betp = {cls: 0.0 for cls in theta}
        for focal, mass in bpa.items():
            for cls in focal:
                betp[cls] += mass / len(focal)
        return betp

    def _shafer_discount(self, bpa, alpha):
        """
        Shafer discounting: m'(A) = α·m(A) for A ≠ Θ, m'(Θ) += (1-α)·(1-m(Θ)).

        Discounting moves mass toward Θ but preserves the focal element structure.
        """
        theta = frozenset({"N", "U", "P", "L"})
        discounted = {}
        moved_to_theta = 0.0
        for focal, mass in bpa.items():
            if focal == theta:
                discounted[theta] = mass
            else:
                discounted[focal] = alpha * mass
                moved_to_theta += (1 - alpha) * mass
        discounted[theta] = discounted.get(theta, 0.0) + moved_to_theta
        return discounted

    def _average_bpas(self, bpas):
        """Arithmetic average of multiple BPAs."""
        n = len(bpas)
        all_focals = set()
        for bpa in bpas:
            all_focals.update(bpa.keys())
        avg = {}
        for focal in all_focals:
            avg[focal] = sum(bpa.get(focal, 0.0) for bpa in bpas) / n
        return avg

    def _dempster_combine(self, bpa1, bpa2):
        """Dempster's rule of combination."""
        theta = frozenset({"N", "U", "P", "L"})
        combined = {}
        conflict = 0.0
        for f1, m1 in bpa1.items():
            for f2, m2 in bpa2.items():
                inter = f1 & f2
                if not inter:
                    conflict += m1 * m2
                else:
                    combined[inter] = combined.get(inter, 0.0) + m1 * m2
        if conflict >= 1.0:
            raise ValueError("Total conflict")
        norm = 1.0 / (1.0 - conflict)
        return {k: v * norm for k, v in combined.items()}

    def test_ceiling_holds_after_shafer_discounting(self, sample_vg_bpa):
        """BetP(L) ≤ BetP(N) after Shafer discounting with any α ∈ (0, 1)."""
        for alpha in [0.1, 0.5, 0.8, 0.95]:
            discounted = self._shafer_discount(sample_vg_bpa, alpha)
            betp = self._compute_betp(discounted)

            # Verify sum to 1
            assert abs(sum(betp.values()) - 1.0) < 1e-9

            # Ceiling preserved
            assert betp["L"] <= betp["N"] + 1e-9, (
                f"α={alpha}: BetP(L)={betp['L']:.4f} > BetP(N)={betp['N']:.4f}"
            )

    def test_ceiling_holds_after_murphy_averaging(self, sample_vg_bpa):
        """BetP(L) ≤ BetP(N) after Murphy averaging of two discounted BPAs."""
        theta = frozenset({"N", "U", "P", "L"})

        # Two different V+G BPAs
        bpa1 = self._shafer_discount(sample_vg_bpa, 0.8)
        bpa2 = self._shafer_discount({
            frozenset({"P"}): 0.05,
            frozenset({"N", "U"}): 0.4,
            frozenset({"N", "U", "L"}): 0.2,
            frozenset({"U"}): 0.2,
            frozenset({"N", "P", "L"}): 0.1,
            theta: 0.05,
        }, 0.9)

        # Murphy averaging: average then self-combine (n-1) times
        avg = self._average_bpas([bpa1, bpa2])
        result = avg.copy()
        result = self._dempster_combine(result, avg)  # One self-combination

        betp = self._compute_betp(result)

        # Verify sum to 1
        assert abs(sum(betp.values()) - 1.0) < 1e-9

        # Ceiling preserved
        assert betp["L"] <= betp["N"] + 1e-9, (
            f"After Murphy: BetP(L)={betp['L']:.4f} > BetP(N)={betp['N']:.4f}"
        )

    def test_ceiling_holds_after_dempster_self_combination(self, sample_vg_bpa):
        """BetP(L) ≤ BetP(N) after Dempster self-combination."""
        # Combine BPA with itself
        combined = self._dempster_combine(sample_vg_bpa, sample_vg_bpa)

        betp = self._compute_betp(combined)

        # Verify sum to 1
        assert abs(sum(betp.values()) - 1.0) < 1e-9

        # Ceiling preserved
        assert betp["L"] <= betp["N"] + 1e-9, (
            f"After self-combination: BetP(L)={betp['L']:.4f} > BetP(N)={betp['N']:.4f}"
        )

        # Additional check: focal element structure preserved
        # All resulting focal elements containing L should also contain N
        for focal in combined.keys():
            if "L" in focal:
                assert "N" in focal, (
                    f"Post-combination focal {set(focal)} has L without N"
                )


class Test5ClassFrameForCeiling:
    """Test the 5-class analytical frame for ceiling derivation."""

    def test_5class_frame_exists(self):
        """The 5-class frame must exist for analytical ceiling derivation."""
        frame_path = REPO_ROOT / "configs" / "frame_5class.yaml"
        assert frame_path.exists(), (
            f"5-class frame not found at {frame_path}. "
            "This is required for analytical ceiling derivation."
        )

    def test_5class_frame_has_C(self):
        """The 5-class frame must include crack (C) for ceiling proposition."""
        from src.utils.frame import load_frame
        frame = load_frame(REPO_ROOT / "configs" / "frame_5class.yaml")
        assert "C" in frame.classes, "5-class frame must include C"

    def test_5class_ceiling_still_holds(self):
        """
        Even with C, V+G cannot isolate L.

        In 5-class frame:
            V4 = {N, U, L, C}  - contains L and N
            G2 = {N, P, L, C}  - contains L and N

        Still: L ⟹ N for all V and G focal elements.
        """
        from src.utils.frame import load_frame
        frame = load_frame(REPO_ROOT / "configs" / "frame_5class.yaml")

        vision_heads = frame.heads_for("vision")
        geometry_heads = frame.heads_for("geometry")

        all_focals = list(vision_heads.values()) + list(geometry_heads.values())

        for focal in all_focals:
            els = set(focal)
            if "L" in els:
                assert "N" in els, (
                    f"5-class frame: focal {els} contains L but not N. "
                    f"This would violate the ceiling proposition."
                )
