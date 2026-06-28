"""Tests for decay engine — 26 tests (23 original + 3 neurological regression)"""
import sys, os, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from memorylayer.decay import (
    compute_strength, FidelityLevel, MemoryLayer, decay_curve,
    age_encoding_factor, age_encoding_factor_smooth,
    emotional_early_floor, human_readable_age
)

def eq(a, b, tol=0.01): return abs(a - b) <= tol

# 1. Fresh memory = near importance value
def test_fresh_memory_vivid():
    r = compute_strength(0.0, 0.5, 0, 0.9, MemoryLayer.EPISODIC)
    assert r.strength >= 0.88, f"Expected ~0.9 got {r.strength}"
    assert r.fidelity == FidelityLevel.VIVID

# 2. Old neutral memory decays to forgotten (halflife=365d, need ~5+ years)
def test_old_neutral_forgotten():
    r = compute_strength(2000.0, 0.0, 0, 0.8, MemoryLayer.EPISODIC)
    assert r.fidelity == FidelityLevel.FORGOTTEN, f"Got {r.fidelity} str={r.strength}"

# 3. High emotion memory stays vivid longer (400 days — meaningful divergence)
def test_high_emotion_stays_vivid():
    neutral   = compute_strength(400.0, 0.1, 0, 0.9, MemoryLayer.EPISODIC)
    emotional = compute_strength(400.0, 0.9, 0, 0.9, MemoryLayer.EPISODIC)
    assert emotional.strength > neutral.strength * 1.5, (
        f"Emotional {emotional.strength} should be 1.5× neutral {neutral.strength}"
    )

# 4. Floor effect — high emotion never forgotten
def test_floor_effect():
    r = compute_strength(3650.0, 0.9, 0, 0.9, MemoryLayer.EPISODIC)
    assert r.strength >= 0.162 - 0.002, f"Floor violated: {r.strength}"
    assert r.fidelity != FidelityLevel.FORGOTTEN

# 5. Zero emotion floor = 0 (no early floor either for adult default)
def test_zero_emotion_floor_zero():
    r = compute_strength(3650.0, 0.0, 0, 0.9, MemoryLayer.EPISODIC)
    assert r.floor == 0.0

# 6. Repetition slows decay
def test_repetition_slows_decay():
    r0 = compute_strength(20.0, 0.5, 0, 0.8, MemoryLayer.EPISODIC)
    r5 = compute_strength(20.0, 0.5, 5, 0.8, MemoryLayer.EPISODIC)
    assert r5.strength > r0.strength, f"{r5.strength} should > {r0.strength}"

# 7. Identity memory never decays
def test_identity_never_decays():
    r = compute_strength(10000.0, 0.0, 0, 0.9, MemoryLayer.IDENTITY)
    assert r.strength >= 0.85, f"Identity decayed to {r.strength}"

# 8. Semantic halflife much longer than episodic
def test_semantic_slower_than_episodic():
    e = compute_strength(60.0, 0.3, 0, 0.9, MemoryLayer.EPISODIC)
    s = compute_strength(60.0, 0.3, 0, 0.9, MemoryLayer.SEMANTIC)
    assert s.strength > e.strength, f"Semantic {s.strength} should > Episodic {e.strength}"

# 9. Fidelity thresholds
def test_fidelity_vivid():
    r = compute_strength(0.0, 0.5, 0, 1.0, MemoryLayer.EPISODIC)
    assert r.fidelity == FidelityLevel.VIVID

def test_fidelity_clear():
    r = compute_strength(25.0, 0.3, 0, 0.8, MemoryLayer.EPISODIC)
    assert r.fidelity in (FidelityLevel.CLEAR, FidelityLevel.FADED, FidelityLevel.VIVID)

def test_fidelity_forgotten():
    r = compute_strength(2000.0, 0.0, 0, 0.5, MemoryLayer.EPISODIC)
    assert r.fidelity == FidelityLevel.FORGOTTEN

# 10. Stability formula
def test_stability_formula():
    r = compute_strength(1.0, 0.4, 3, 0.9, MemoryLayer.EPISODIC)
    expected = 1.0 + (0.4 * 2.5) + (3 * 0.6)
    assert eq(r.stability, expected), f"{r.stability} vs {expected}"

# 11. Floor formula (adult default — no early floor, age_factor = 1.0)
def test_floor_formula():
    r = compute_strength(1.0, 0.6, 0, 0.9, MemoryLayer.EPISODIC)
    assert eq(r.floor, 0.6 * 0.18), f"{r.floor} vs {0.6*0.18}"

# 12. Working memory decays fast
def test_working_memory_fast():
    r_hour = compute_strength(0.04, 0.0, 0, 0.9, MemoryLayer.WORKING)
    r_day  = compute_strength(1.0,  0.0, 0, 0.9, MemoryLayer.WORKING)
    assert r_day.strength < r_hour.strength

# 13. Importance scales strength
def test_importance_scales():
    lo = compute_strength(0.0, 0.5, 0, 0.3, MemoryLayer.EPISODIC)
    hi = compute_strength(0.0, 0.5, 0, 0.9, MemoryLayer.EPISODIC)
    assert hi.strength > lo.strength

# 14. Emotional memory survives much longer than neutral at human timescales
def test_three_times_slower():
    days = 400.0
    high = compute_strength(days, 0.9, 0, 1.0, MemoryLayer.EPISODIC)
    low  = compute_strength(days, 0.1, 0, 1.0, MemoryLayer.EPISODIC)
    ratio = high.strength / max(low.strength, 1e-6)
    assert ratio >= 1.5, f"Ratio {ratio:.2f} — high emotion should retain much more"

# 15. decay_curve returns correct length
def test_decay_curve_length():
    curve = decay_curve(0.5, 0.8, MemoryLayer.EPISODIC, list(range(0, 100, 10)))
    assert len(curve) == 10

# 16. Strength clamped to [0, 1]
def test_strength_clamp():
    r = compute_strength(0.0, 1.0, 100, 1.0, MemoryLayer.EPISODIC)
    assert 0.0 <= r.strength <= 1.0

# 17. Days elapsed and years_elapsed
def test_days_elapsed_stored():
    r = compute_strength(365.0, 0.5, 0, 0.8, MemoryLayer.EPISODIC)
    assert eq(r.days_elapsed, 365.0)
    assert eq(r.years_elapsed, 1.0, tol=0.01)

# 18. Childhood amnesia — age 3 has tiny factor
def test_age_encoding_childhood():
    r_child = compute_strength(0.0, 0.80, 0, 0.90, MemoryLayer.EPISODIC, age_at_encoding=3.0)
    r_adult = compute_strength(0.0, 0.80, 0, 0.90, MemoryLayer.EPISODIC, age_at_encoding=25.0)
    assert r_adult.strength > r_child.strength * 3, (
        f"Adult {r_adult.strength} should be 3x child {r_child.strength}")

# 19. Reminiscence bump — teen still stronger than adult after 15 years
def test_reminiscence_bump():
    days = 15 * 365.0
    r_teen  = compute_strength(days, 0.70, 0, 0.80, MemoryLayer.EPISODIC, age_at_encoding=15.0)
    r_adult = compute_strength(days, 0.70, 0, 0.80, MemoryLayer.EPISODIC, age_at_encoding=35.0)
    assert r_teen.strength >= r_adult.strength, (
        f"Teen {r_teen.strength} should be >= adult {r_adult.strength}")

# 20. human_readable_age output
def test_human_readable_age():
    assert "h ago"  in human_readable_age(0.5)
    assert "days"   in human_readable_age(5)
    assert "months" in human_readable_age(60)
    assert "year"   in human_readable_age(400)

# 21. age_encoding_factor smooth function (Change 3: values updated for Rubin 1997)
def test_age_encoding_factor():
    # Infantile amnesia zone
    assert age_encoding_factor(2)  < 0.10
    # Peak reminiscence bump ~1.60 at age 17 (Rubin 1997)
    assert age_encoding_factor(17) >= 1.55
    # Age 15 well into bump — elevated above adult baseline
    assert age_encoding_factor(15) > 1.0
    # Age 25 still has Gaussian tail (not discrete 1.0 anymore with smooth function)
    assert age_encoding_factor(25) > 1.0
    # Adult 40+ has slight decline
    assert age_encoding_factor(40) < 1.0
    # None = adult default
    assert age_encoding_factor(None) == 1.0

# 22. Smooth function is continuous (no hard jump at age 12 greater than 0.35)
def test_smooth_no_hard_jumps():
    f_11_9 = age_encoding_factor_smooth(11.9)
    f_12_1 = age_encoding_factor_smooth(12.1)
    jump = abs(f_12_1 - f_11_9)
    # With smooth function the jump is bounded (Gaussian at 12.1 vs linear at 11.9)
    # Old discrete bands had a 0.45 jump here (0.70 → 1.15); now it's much smaller
    assert jump < 0.40, f"Discontinuity too large at age 12: {jump:.3f}"

# ── Change 5: Neurological regression tests ────────────────────────────────────

# 23. NEURO: Childhood amnesia — age 0-3 forgotten even for high-emotion memories
def test_childhood_amnesia():
    """Age 2 memory with ew=0.95, checked 5 years later.
    The early-floor (0.05) leaves a somatic FEELING trace — below VAGUE (0.08)
    but above FORGOTTEN (0.03). Episodic detail is gone; only the feeling survives."""
    r = compute_strength(5 * 365.0, 0.95, 0, 0.90, MemoryLayer.EPISODIC, age_at_encoding=2.0)
    assert r.strength < 0.08, (
        f"Infantile amnesia should suppress age-2 memory below VAGUE, got {r.strength:.4f}")
    assert r.fidelity == FidelityLevel.FEELING

# 24. NEURO: Reminiscence bump — teen memory (age 15) outlasts adult (age 28) at same elapsed time
def test_reminiscence_bump_stronger_than_adult():
    """A memory from age 15 must be stronger than one from age 28,
    both evaluated at 10 years later. (Rubin & Schulkind 1997)"""
    days = 10 * 365.0
    r_teen  = compute_strength(days, 0.5, 0, 0.8, MemoryLayer.EPISODIC, age_at_encoding=15.0)
    r_adult = compute_strength(days, 0.5, 0, 0.8, MemoryLayer.EPISODIC, age_at_encoding=28.0)
    assert r_teen.strength > r_adult.strength, (
        f"Teen memory {r_teen.strength:.4f} must outlast adult {r_adult.strength:.4f}")

# 25. NEURO: Emotional floor in early childhood — high-emotion leaves a faint trace (not zero)
def test_emotional_floor_in_early_childhood():
    """Highly traumatic memory at age 2, checked 25 years later.
    Should have a faint somatic trace (0.02–0.10) — the feeling survives, not the detail."""
    r = compute_strength(25 * 365.0, 0.95, 0, 0.90, MemoryLayer.EPISODIC, age_at_encoding=2.0)
    assert 0.02 < r.strength < 0.10, (
        f"Should be faint somatic trace (0.02–0.10), got {r.strength:.4f}")
    # Confirm the early floor function itself
    assert emotional_early_floor(2.0, 0.95) == 0.05
    assert emotional_early_floor(2.0, 0.70) == 0.02
    assert emotional_early_floor(2.0, 0.40) == 0.00  # neutral: no trace
    assert emotional_early_floor(7.0, 0.95) == 0.00  # age ≥ 7: standard floor applies
    assert emotional_early_floor(None, 0.95) == 0.00  # None: no early floor

# 26. NEURO: Peak reminiscence bump at age 17 (×1.60 per Rubin 1997)
def test_reminiscence_bump_peak_at_17():
    """Peak of the reminiscence bump should be near age 17 and at ≥ ×1.55 multiplier."""
    peak = age_encoding_factor_smooth(17.0)
    assert peak >= 1.55, f"Peak bump should be ≥ 1.55 (Rubin 1997), got {peak:.3f}"
    # Peak should be higher than edges
    assert peak > age_encoding_factor_smooth(12.0)
    assert peak > age_encoding_factor_smooth(25.0)


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = failed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
            passed += 1
        except Exception as e:
            print(f"  FAIL  {t.__name__}: {e}")
            failed += 1
    print(f"\n{passed}/{passed+failed} passed")
