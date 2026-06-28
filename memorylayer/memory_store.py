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
MemoryLayer — Memory Store

Core storage layer. Handles write, retrieve, tag/layer filtering,
spreading activation, and automatic emotional classification.
"""

import math
import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .decay import (
    compute_strength, FidelityLevel, MemoryLayer, DecayResult,
    age_encoding_factor,
)
from .emotion import classify
from .fidelity import degrade_content, FidelityResult


@dataclass
class Memory:
    id: str
    content: str
    layer: str
    tags: List[str]
    importance: float
    emotional_weight: float
    dominant_emotion: str
    repetition_count: int
    created_at: float
    last_accessed: float
    age_at_encoding: Optional[float]
    metadata: dict
    last_reconsolidated: float
    reconsolidation_count: int
    original_content: str


@dataclass
class RetrievedMemory:
    memory: Memory
    decay: DecayResult
    display_content: str
    fidelity_result: FidelityResult
    activation_boost: float = 0.0


# ── Spreading activation ──────────────────────────────────────────────────────

def _cosine_similarity(a: list, b: list) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(x * x for x in b))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


def spreading_activation(
    source: Memory,
    all_memories: List[Memory],
    decay_fn,
    now: float,
    boost: float = 0.3,
    sim_threshold: float = 0.5,
    emotion_tolerance: float = 0.5,
) -> List[Tuple[str, float]]:
    """
    Find memories semantically similar to `source` and return
    (memory_id, activation_boost) pairs. Excludes:
      - memories without embeddings
      - emotionally incompatible memories (outside emotion_tolerance)
      - memories at or below their floor (somatic remnants, not retrievable)
    """
    src_emb = source.metadata.get("embedding") if source.metadata else None
    if src_emb is None:
        return []

    activated = []
    for mem in all_memories:
        if mem.id == source.id:
            continue
        mem_emb = mem.metadata.get("embedding") if mem.metadata else None
        if mem_emb is None:
            continue

        sim = _cosine_similarity(src_emb, mem_emb)
        if sim < sim_threshold:
            continue

        if abs(mem.emotional_weight - source.emotional_weight) > emotion_tolerance:
            continue

        dr = decay_fn(mem, now)
        # Exclude if truly forgotten OR only alive due to emotional floor (somatic trace)
        if dr.fidelity == FidelityLevel.FORGOTTEN or dr.strength <= dr.floor:
            continue

        activated.append((mem.id, boost * sim))

    return activated


# ── MemoryStore ───────────────────────────────────────────────────────────────

class MemoryStore:
    def __init__(self, auto_embed: bool = False):
        self._memories: Dict[str, Memory] = {}
        self._decay = self._compute_decay
        self._auto_embed = auto_embed

    def _compute_decay(self, memory: Memory, now: float) -> DecayResult:
        days = (now - memory.created_at) / 86400.0
        layer = MemoryLayer(memory.layer)
        return compute_strength(
            days_elapsed=days,
            emotional_weight=memory.emotional_weight,
            repetition_count=memory.repetition_count,
            importance=memory.importance,
            layer=layer,
            age_at_encoding=memory.age_at_encoding,
        )

    def write(
        self,
        content: str,
        layer: MemoryLayer = MemoryLayer.EPISODIC,
        tags: Optional[List[str]] = None,
        emotional_weight: Optional[float] = None,
        importance: float = 0.7,
        now: Optional[float] = None,
        age_at_encoding: Optional[float] = None,
        metadata: Optional[dict] = None,
    ) -> Memory:
        ts = now or time.time()

        # Auto-generate embedding if enabled and not already provided
        if self._auto_embed and "embedding" not in (metadata or {}):
            try:
                from .embedding import embed as _embed
                emb = _embed(content)
                if emb is not None:
                    metadata = {**(metadata or {}), "embedding": emb}
            except Exception:
                pass

        if emotional_weight is None:
            result = classify(content)
            ew = result.weight
            dominant = result.dominant_emotion
        else:
            ew = emotional_weight
            dominant = "manual"

        mem = Memory(
            id=str(uuid.uuid4()),
            content=content,
            layer=layer.value,
            tags=tags or [],
            importance=importance,
            emotional_weight=ew,
            dominant_emotion=dominant,
            repetition_count=0,
            created_at=ts,
            last_accessed=ts,
            age_at_encoding=age_at_encoding,
            metadata=metadata or {},
            last_reconsolidated=0.0,
            reconsolidation_count=0,
            original_content="",
        )
        self._memories[mem.id] = mem
        return mem

    def _make_retrieved(
        self,
        memory: Memory,
        now: float,
        activation_boost: float = 0.0,
        increment_recall: bool = False,
    ) -> RetrievedMemory:
        if increment_recall:
            memory.repetition_count += 1
            memory.last_accessed = now
        dr = self._compute_decay(memory, now)
        fr = degrade_content(memory.content, dr.fidelity)
        return RetrievedMemory(
            memory=memory,
            decay=dr,
            display_content=fr.displayed_content,
            fidelity_result=fr,
            activation_boost=activation_boost,
        )

    def retrieve(
        self,
        query_tags: Optional[List[str]] = None,
        layer: Optional[MemoryLayer] = None,
        now: Optional[float] = None,
        spreading: bool = True,
        increment_recall: bool = True,
    ) -> List[RetrievedMemory]:
        ts = now or time.time()
        candidates = list(self._memories.values())

        if layer is not None:
            candidates = [m for m in candidates if m.layer == layer.value]
        if query_tags:
            candidates = [m for m in candidates if any(t in m.tags for t in query_tags)]

        results = []
        for mem in candidates:
            rm = self._make_retrieved(mem, ts, increment_recall=increment_recall)
            if rm.decay.fidelity != FidelityLevel.FORGOTTEN:
                results.append(rm)

        if spreading and results:
            top = max(results, key=lambda r: r.decay.strength)
            all_mems = list(self._memories.values())
            boost_map = {mid: b for mid, b in
                         spreading_activation(top.memory, all_mems, self._decay, ts)}
            for rm in results:
                if rm.memory.id in boost_map:
                    rm.activation_boost = boost_map[rm.memory.id]

        results.sort(key=lambda r: r.decay.strength + r.activation_boost, reverse=True)
        return results

    def retrieve_one(
        self,
        memory_id: str,
        increment_recall: bool = True,
        now: Optional[float] = None,
    ) -> Optional[RetrievedMemory]:
        mem = self._memories.get(memory_id)
        if mem is None:
            return None
        ts = now or time.time()
        return self._make_retrieved(mem, ts, increment_recall=increment_recall)

    def all_with_decay(self, now: Optional[float] = None) -> List[RetrievedMemory]:
        ts = now or time.time()
        return [self._make_retrieved(m, ts) for m in self._memories.values()]

    def consolidate_to_semantic(
        self,
        memory_id: str,
        compressed_content: str,
        now: Optional[float] = None,
    ) -> Memory:
        ts = now or time.time()
        original = self._memories.get(memory_id)
        new_mem = self.write(
            content=compressed_content,
            layer=MemoryLayer.SEMANTIC,
            tags=original.tags if original else [],
            emotional_weight=original.emotional_weight if original else 0.0,
            importance=original.importance if original else 0.7,
            now=ts,
            age_at_encoding=original.age_at_encoding if original else None,
        )
        if original:
            del self._memories[memory_id]
        return new_mem

    def stats(self) -> dict:
        ts = time.time()
        all_rm = self.all_with_decay(now=ts)
        active = sum(1 for rm in all_rm if rm.decay.fidelity != FidelityLevel.FORGOTTEN)
        by_layer: Dict[str, int] = {}
        for rm in all_rm:
            by_layer[rm.memory.layer] = by_layer.get(rm.memory.layer, 0) + 1
        return {
            "total": len(self._memories),
            "active": active,
            "forgotten": len(self._memories) - active,
            "by_layer": by_layer,
        }
