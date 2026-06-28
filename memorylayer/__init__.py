"""
MemoryLayer — AI That Remembers The Way You Do

Primary import:
    from memorylayer import MemoryStore, MemoryLayer
"""

# ── Primary API ───────────────────────────────────────────────────────────────
from .memory_store import MemoryStore, Memory, RetrievedMemory, spreading_activation
from .decay import (
    MemoryLayer, FidelityLevel, DecayResult,
    compute_strength,
    age_encoding_factor, age_encoding_factor_smooth, emotional_early_floor,
)

# ── Emotion + Fidelity ────────────────────────────────────────────────────────
from .emotion import classify, neural_emotional_weight, NEURAL_AVAILABLE
from .fidelity import degrade_content, build_prompt_context

# ── Consolidation + Reconsolidation ──────────────────────────────────────────
from .consolidation import run_consolidation
from .reconsolidation import (
    reconsolidate_sync, reconsolidate_async,
    RECONSOLIDATION_THRESHOLD, MAX_RECONSOLIDATIONS,
)

# ── Extensions ────────────────────────────────────────────────────────────────
from .embedding import embed, get_embedder, EMBED_AVAILABLE
from .prospective import (
    write_intention, complete_intention,
    get_pending_intentions, get_overdue_intentions, is_overdue,
)
from .collective import CollectiveMemoryStore, CollectiveMemory, EntityRecall
from .sleep import simulate_sleep, SleepSession, SleepConsolidationResult

__version__ = "3.0.0"
__author__  = "Mounica Goriparti"
__license__ = "Proprietary"
__all__ = [
    # Core
    "MemoryStore", "Memory", "RetrievedMemory", "spreading_activation",
    "MemoryLayer", "FidelityLevel", "DecayResult",
    # Emotion + Fidelity
    "classify", "neural_emotional_weight", "NEURAL_AVAILABLE",
    "degrade_content", "build_prompt_context",
    # Consolidation
    "run_consolidation",
    "reconsolidate_sync", "reconsolidate_async",
    "RECONSOLIDATION_THRESHOLD", "MAX_RECONSOLIDATIONS",
    # Embeddings
    "embed", "get_embedder", "EMBED_AVAILABLE",
    # Prospective
    "write_intention", "complete_intention",
    "get_pending_intentions", "get_overdue_intentions", "is_overdue",
    # Collective
    "CollectiveMemoryStore", "CollectiveMemory", "EntityRecall",
    # Sleep
    "simulate_sleep", "SleepSession", "SleepConsolidationResult",
]
