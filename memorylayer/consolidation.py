"""
MemoryLayer — Consolidation Pipeline

Time-triggered episodic → semantic compression (mirrors human sleep consolidation).
Episodic memories that are old enough and strong enough get abstracted into
compressed semantic facts that decay much more slowly.

Trigger: memories older than consolidation_threshold_days and still CLEAR or above.
"""

import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

from .decay import MemoryLayer, FidelityLevel
from .memory_store import MemoryStore, Memory, RetrievedMemory


@dataclass
class ConsolidationResult:
    consolidated: int        # number of episodes compressed
    skipped: int             # too old/weak or already semantic
    new_semantic_ids: List[str]
    summary: str


def _compress_content(content: str) -> str:
    """
    Produce a compressed semantic fact from an episodic memory.
    Strategy: keep the core factual claim, drop emotional/temporal qualifiers.
    Heuristic: take the main clause (strip subordinate clauses starting with
    'because', 'when', 'after', etc.) and trim to ≤ 80 chars.
    """
    import re
    # Strip subordinate clause markers
    text = re.sub(
        r"\b(because|when|after|before|although|while|since|as|if|unless)\b.*$",
        "", content, flags=re.IGNORECASE
    ).strip().rstrip(".,;")

    # Remove first-person hedging
    text = re.sub(r"^(I (felt|was|became|got|realized|noticed|felt)\s+)", "", text, flags=re.IGNORECASE)

    # Truncate
    if len(text) > 120:
        words = text.split()
        text = " ".join(words[:15]) + "…"

    return text.strip() or content[:80]


def run_consolidation(
    store: MemoryStore,
    consolidation_threshold_days: float = 7.0,
    min_fidelity: FidelityLevel = FidelityLevel.CLEAR,
    now: Optional[float] = None,
) -> ConsolidationResult:
    """
    Scan episodic memories older than threshold.
    Compress CLEAR+ memories to semantic layer.
    Returns summary of what was consolidated.
    """
    ts = now or time.time()
    threshold_sec = consolidation_threshold_days * 86400

    candidates = [
        r for r in store.all_with_decay(now=ts)
        if r.memory.layer == MemoryLayer.EPISODIC.value
        and (ts - r.memory.created_at) >= threshold_sec
        and r.decay.fidelity not in (FidelityLevel.FORGOTTEN, FidelityLevel.VAGUE)
    ]

    consolidated = 0
    new_ids = []

    for rm in candidates:
        compressed = _compress_content(rm.memory.content)
        new_mem = store.consolidate_to_semantic(
            memory_id=rm.memory.id,
            compressed_content=compressed,
            now=ts,
        )
        new_ids.append(new_mem.id)
        consolidated += 1

    all_episodic = sum(
        1 for r in store.all_with_decay(now=ts)
        if r.memory.layer == MemoryLayer.EPISODIC.value
    )

    summary = (
        f"Consolidated {consolidated} episodic → semantic. "
        f"Remaining episodic: {all_episodic}."
    )

    return ConsolidationResult(
        consolidated=consolidated,
        skipped=len(candidates) - consolidated,
        new_semantic_ids=new_ids,
        summary=summary,
    )
