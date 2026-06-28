
# MemoryLayer — Human-like AI Memory System
# Copyright 2026 Goriparti Mounica
# 
# Licensed under the Apache License, Version 2.0
#
# PATENT NOTICE: This software implements methods 
# covered by provisional patent application 
# IN/PA/2026/202641071906 filed with the Indian 
# Patent Office on June 10, 2026. The following 
# mechanisms are patent-pending:
# 1. Emotionally-modulated temporal decay rate 
#    with floor effect
# 2. Fidelity-degraded content retrieval with 
#    confidence-calibrated natural language expression
#
# Commercial use of these patent-pending mechanisms 
# requires a separate commercial license.
# Contact: mounica.goriparti@email.com
"""
MemoryLayer — Decay Engine  (Human-Calibrated v2.0)

Models memory of any person looking back across their life.
The age-at-encoding factor implements three neuroscience phenomena:

  CHILDHOOD AMNESIA (age 0-7):
    Near-zero encoding — most memories lost regardless of emotion.
    Exception: extreme emotional events leave a faint somatic trace (floor 0.05).

  LATENCY PERIOD (age 7-12):
    Linear recovery — selective survival of vivid events.

  REMINISCENCE BUMP (age 12-25, peak at 17):
    Gaussian elevation — adolescent and early-adult memories are recalled
    at 1.5-1.6× the rate of mid-adult memories. (Rubin & Schulkind 1997)

  ADULT BASELINE (age 25-35): multiplier ~1.0
  GENTLE DECLINE  (age 35+):  slight encoding weakness with age

Formula (from patent):
  strength(t) = max(floor, exp(-t / (stability × halflife)) × importance)
  stability   = 1 + (emotional_weight × 2.5) + (repetition_count × 0.6)
  floor       = emotional_weight × 0.18 × age_factor

Halflives:
  Working  : ~1 hour
  Episodic : 365 days (1 year — vivid for ~2-3 years)
  Semantic : 1825 days (5 years)
  Identity : infinite
"""

import math
from dataclasses import dataclass
from enum import Enum
from typing import Optional


class MemoryLayer(Enum):
    WORKING      = "working"
    EPISODIC     = "episodic"
    SEMANTIC     = "semantic"
    IDENTITY     = "identity"
    PROSPECTIVE  = "prospective"   # intentions that decay fast if unacted on


class FidelityLevel(Enum):
    VIVID     = "VIVID"      # strength > 0.70  — "I clearly remember…"
    CLEAR     = "CLEAR"      # 0.45 – 0.70      — "I remember…"
    FADED     = "FADED"      # 0.22 – 0.45      — "I vaguely recall…"
    VAGUE     = "VAGUE"      # 0.08 – 0.22      — "I have a faint sense that…"
    FEELING   = "FEELING"    # 0.03 – 0.08      — "I don't remember what happened, but I remember how it felt"
    FORGOTTEN = "FORGOTTEN"  # < 0.03            — not returned at all


HALFLIFE = {
    MemoryLayer.WORKING:     0.042,   # ~1 hour
    MemoryLayer.EPISODIC:    365.0,   # 1 year
    MemoryLayer.SEMANTIC:    1825.0,  # 5 years
    MemoryLayer.IDENTITY:    1e9,     # permanent
    MemoryLayer.PROSPECTIVE: 3.0,     # 3 days — intentions fade fast if unacted on
}

FIDELITY_THRESHOLDS = [
    (0.70, FidelityLevel.VIVID),
    (0.45, FidelityLevel.CLEAR),
    (0.22, FidelityLevel.FADED),
    (0.08, FidelityLevel.VAGUE),
    (0.03, FidelityLevel.FEELING),    # emotion survives, episodic detail gone
    (0.00, FidelityLevel.FORGOTTEN),
]


# ── Change 3: Smooth continuous age-encoding function ──────────────────────────
# Replaces the old discrete band dict. No hard jumps between ages.
# Three regions stitched together:
#   0-7  : quadratic recovery from near-zero (childhood amnesia)
#   7-12 : linear climb through latency period
#   12-30: Gaussian bell curve peaking at age 17 (reminiscence bump — Rubin 1997)
#   30+  : gentle decline from adult baseline

def age_encoding_factor_smooth(age: float) -> float:
    """
    Continuous age-encoding multiplier applied to initial importance.
    Implements childhood amnesia, latency period, reminiscence bump, and aging decline.
    """
    # Childhood amnesia: steep recovery from near-zero (quadratic)
    if age < 7:
        return 0.04 + (0.51 * (age / 7) ** 2)

    # Latency period: linear climb
    if age < 12:
        return 0.55 + (0.65 * (age - 7) / 5)

    # Reminiscence bump: Gaussian bell curve peaking at age 17
    # (Rubin & Schulkind 1997: adolescent memories recalled at 1.5-2× adult rate)
    if age < 30:
        bump_peak   = 17.0
        bump_width  = 7.0
        bump_height = 0.60
        bump = bump_height * math.exp(-((age - bump_peak) ** 2) / (2 * bump_width ** 2))
        return 1.0 + bump

    # Adult baseline with gradual encoding decline after 35
    return max(0.75, 1.0 - ((age - 30) * 0.004))


def age_encoding_factor(age_at_encoding: Optional[float]) -> float:
    """Public API: returns smooth age-encoding multiplier. None → adult baseline (1.0)."""
    if age_at_encoding is None:
        return 1.0
    return age_encoding_factor_smooth(age_at_encoding)


# ── Change 2: Emotional early floor for pre-7 memories ────────────────────────
# Even though episodic details are lost in childhood amnesia, highly emotional
# events leave a somatic/emotional imprint — "I remember being terrified"
# not "I remember what happened." This floor is deliberately below VAGUE
# threshold (0.08): it represents a felt trace, not a retrievable memory.

def emotional_early_floor(age_at_encoding: Optional[float], emotional_weight: float) -> float:
    """
    Additional floor for high-emotion memories formed before age 7.
    Returns a value in [0, 0.05] — below retrieval threshold but mathematically present.
    """
    if age_at_encoding is None or age_at_encoding >= 7:
        return 0.0
    if emotional_weight > 0.85:
        return 0.05   # "I remember being scared" — feeling survives, not detail
    if emotional_weight > 0.65:
        return 0.02   # very faint emotional trace
    return 0.0        # neutral early memories: nothing survives


@dataclass
class DecayResult:
    strength: float
    fidelity: FidelityLevel
    stability: float
    floor: float          # effective floor (standard × age_factor, or early_floor if higher)
    days_elapsed: float
    years_elapsed: float
    layer: MemoryLayer
    emotional_weight: float
    repetition_count: int
    age_factor: float     # smooth age-encoding multiplier applied


def compute_strength(
    days_elapsed: float,
    emotional_weight: float,
    repetition_count: int,
    importance: float,
    layer: MemoryLayer,
    age_at_encoding: Optional[float] = None,
) -> DecayResult:
    """
    Compute current memory strength using the MemoryLayer decay formula.

    Args:
        days_elapsed:      time since memory was stored (fractional days)
        emotional_weight:  0.0–1.0 emotional significance
        repetition_count:  times this memory has been recalled
        importance:        base importance at write time (0.0–1.0)
        layer:             which memory layer
        age_at_encoding:   person's age (years) when memory was formed.
                           None → adult default (1.0 factor, no early floor).
    """
    halflife   = HALFLIFE[layer]
    stability  = 1.0 + (emotional_weight * 2.5) + (repetition_count * 0.6)
    age_factor = age_encoding_factor(age_at_encoding)

    # Raw floor (Claim 1 of patent: emotionally-modulated floor effect)
    floor_base = emotional_weight * 0.18
    standard_floor = floor_base * age_factor

    # Change 2: additional somatic trace for intense early-childhood memories
    early_floor = emotional_early_floor(age_at_encoding, emotional_weight)
    effective_floor = max(standard_floor, early_floor)

    # Effective starting importance adjusted for age-at-encoding
    effective_importance = min(1.0, importance * age_factor)

    if layer == MemoryLayer.IDENTITY:
        raw = effective_importance
    else:
        raw = math.exp(-days_elapsed / (stability * halflife)) * effective_importance

    strength = max(effective_floor, min(1.0, raw))
    fidelity = _classify_fidelity(strength)

    return DecayResult(
        strength=round(strength, 4),
        fidelity=fidelity,
        stability=round(stability, 3),
        floor=round(effective_floor, 4),
        days_elapsed=round(days_elapsed, 3),
        years_elapsed=round(days_elapsed / 365.25, 2),
        layer=layer,
        emotional_weight=emotional_weight,
        repetition_count=repetition_count,
        age_factor=round(age_factor, 4),
    )


def _classify_fidelity(strength: float) -> FidelityLevel:
    for threshold, level in FIDELITY_THRESHOLDS:
        if strength >= threshold:
            return level
    return FidelityLevel.FORGOTTEN


def decay_curve(
    emotional_weight: float,
    importance: float,
    layer: MemoryLayer,
    days: list,
    age_at_encoding: Optional[float] = None,
) -> list:
    """Return list of (day, strength, fidelity) tuples — for display/plotting."""
    return [
        (d, *((r := compute_strength(d, emotional_weight, 0, importance,
                                      layer, age_at_encoding)).strength,
              r.fidelity.value))
        for d in days
    ]


def human_readable_age(days: float) -> str:
    """Convert days elapsed to a human-readable time string."""
    if days < 1:
        hours = days * 24
        return f"{hours:.0f}h ago" if hours >= 1 else "just now"
    if days < 14:
        return f"{days:.0f} days ago"
    if days < 60:
        return f"{days/7:.0f} weeks ago"
    if days < 365:
        return f"{days/30.4:.0f} months ago"
    years = days / 365.25
    return f"{years:.1f} years ago" if years < 2 else f"{years:.0f} years ago"
