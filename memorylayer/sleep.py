"""
MemoryLayer — Sleep Simulation

Sleep quality modulates memory consolidation.
  NREM (slow-wave): rehearses and strengthens recent episodic memories.
  REM:              integrates emotional memories, prunes noise.
  Poor sleep:       weak consolidation, faster subsequent decay.

Mechanism:
  - Good sleep  (effectiveness ≥ 0.65) → adds replay repetitions (boosts stability)
  - Great sleep (effectiveness ≥ 0.65) → tags CLEAR+ episodic as 'sleep_consolidated'
  - Poor sleep  (effectiveness < 0.40) → no boost applied

Research:
  Walker (2017)      — Why We Sleep
  Diekelmann (2010)  — The memory function of sleep (Nature Rev Neuroscience)
  Stickgold (2005)   — Sleep-dependent memory consolidation (Nature)
"""

import time
from dataclasses import dataclass
from typing import Optional

from .memory_store import MemoryStore
from .decay import MemoryLayer, FidelityLevel


@dataclass
class SleepSession:
    quality: float           # 0.0 (fragmented / no sleep) to 1.0 (perfect)
    duration_hours: float    # total hours of sleep
    rem_fraction: float = 0.25   # fraction spent in REM (healthy adult ≈ 0.25)


@dataclass
class SleepConsolidationResult:
    memories_strengthened: int    # memories that received replay repetitions
    memories_consolidated: int    # episodic memories tagged as sleep-consolidated
    sleep_quality: float
    effectiveness: float          # computed overall effectiveness (0–1)


def _sleep_effectiveness(session: SleepSession) -> float:
    """
    Compute overall sleep effectiveness from quality, duration, and REM fraction.
      quality        → 50% of score
      duration       → 30%  (optimal = 8h; capped at 1.0)
      REM fraction   → 20%  (optimal = 0.25; penalty for deviation)
    """
    duration_score = min(1.0, session.duration_hours / 8.0)
    rem_deviation = abs(session.rem_fraction - 0.25) * 4.0
    rem_score = max(0.0, 1.0 - rem_deviation)
    return round(
        session.quality * 0.50
        + duration_score * 0.30
        + rem_score * 0.20,
        3,
    )


def simulate_sleep(
    store: MemoryStore,
    session: SleepSession,
    now: Optional[float] = None,
) -> SleepConsolidationResult:
    """
    Apply one sleep session's consolidation effects to the memory store in-place.

    Effects:
      Strengthening  — NREM replay: adds up to 2 extra repetitions on active memories.
      Consolidation  — Tags CLEAR+ episodic memories as 'sleep_consolidated'
                       (marks them for the episodic→semantic pipeline).
      Weak sleep     — effectiveness < 0.40 provides no benefit.
    """
    ts = now or time.time()
    effectiveness = _sleep_effectiveness(session)
    strengthened = 0
    consolidated = 0

    for rm in store.all_with_decay(now=ts):
        mem = rm.memory
        fid = rm.decay.fidelity

        if fid == FidelityLevel.FORGOTTEN:
            continue
        if mem.layer == MemoryLayer.IDENTITY.value:
            continue

        # NREM replay: add repetition boost for accessible memories
        if (effectiveness >= 0.40
                and fid in (FidelityLevel.VIVID, FidelityLevel.CLEAR, FidelityLevel.FADED)):
            replay = round(effectiveness * 2)   # 1–2 extra repetitions
            if replay > 0:
                mem.repetition_count += replay
                strengthened += 1

        # Sleep consolidation tag: CLEAR+ episodic on good sleep
        if (effectiveness >= 0.65
                and mem.layer == MemoryLayer.EPISODIC.value
                and fid in (FidelityLevel.VIVID, FidelityLevel.CLEAR)
                and "sleep_consolidated" not in mem.tags):
            mem.tags.append("sleep_consolidated")
            consolidated += 1

    return SleepConsolidationResult(
        memories_strengthened=strengthened,
        memories_consolidated=consolidated,
        sleep_quality=session.quality,
        effectiveness=effectiveness,
    )
