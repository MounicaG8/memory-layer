"""
MemoryLayer — Collective Memory

Shared events with per-entity emotional weights.
Models how different people remember the same event differently:
Alice scored the winning goal — her memory stays vivid for decades.
Bob watched from the stands — his fades to a faint impression in a few years.

Research: Halbwachs (1992)       — On Collective Memory
          Welzer et al. (1995)    — Collective and individual memory
          Roediger & Abel (2015)  — Collective Memory: A New Arena of Cognitive Study
"""

import math
import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .decay import MemoryLayer, FidelityLevel, compute_strength, DecayResult


@dataclass
class CollectiveMemory:
    id: str
    content: str
    created_at: float
    tags: List[str]
    entity_weights: Dict[str, float]   # entity_id → emotional_weight
    metadata: dict = field(default_factory=dict)


@dataclass
class EntityRecall:
    event: CollectiveMemory
    entity_id: str
    emotional_weight: float
    decay: DecayResult


class CollectiveMemoryStore:
    """
    Stores events shared between multiple entities.
    Each entity recalls the same event through their own emotional weight —
    same moment, different memory strength.
    """

    def __init__(self):
        self._events: Dict[str, CollectiveMemory] = {}

    def write_event(
        self,
        content: str,
        participants: Dict[str, float],
        tags: Optional[List[str]] = None,
        now: Optional[float] = None,
        metadata: Optional[dict] = None,
    ) -> CollectiveMemory:
        """
        Record a shared event.

        Args:
            content:      Description of the shared event.
            participants: Mapping of entity_id → emotional_weight (0–1).
        """
        ts = now or time.time()
        event = CollectiveMemory(
            id=str(uuid.uuid4()),
            content=content,
            created_at=ts,
            tags=tags or [],
            entity_weights=dict(participants),
            metadata=metadata or {},
        )
        self._events[event.id] = event
        return event

    def recall_for(
        self,
        entity_id: str,
        now: Optional[float] = None,
        importance: float = 0.70,
    ) -> List[EntityRecall]:
        """
        Return all events accessible to `entity_id`, ordered by strength.
        Non-participants and forgotten memories are excluded.
        """
        ts = now or time.time()
        results = []
        for event in self._events.values():
            ew = event.entity_weights.get(entity_id)
            if ew is None:
                continue
            days = (ts - event.created_at) / 86400.0
            decay = compute_strength(
                days_elapsed=days,
                emotional_weight=ew,
                repetition_count=0,
                importance=importance,
                layer=MemoryLayer.EPISODIC,
            )
            if decay.fidelity != FidelityLevel.FORGOTTEN:
                results.append(EntityRecall(
                    event=event,
                    entity_id=entity_id,
                    emotional_weight=ew,
                    decay=decay,
                ))
        results.sort(key=lambda r: r.decay.strength, reverse=True)
        return results

    def divergence(self, event_id: str) -> Optional[float]:
        """
        Emotional divergence across participants for one event.
        Returns the standard deviation of emotional weights,
        or None if the event has fewer than 2 participants.

        0.0 = everyone remembers it the same way.
        ~0.45 = maximum disagreement (one person at 0.95, another at 0.05).
        """
        event = self._events.get(event_id)
        if event is None or len(event.entity_weights) < 2:
            return None
        weights = list(event.entity_weights.values())
        mean = sum(weights) / len(weights)
        variance = sum((w - mean) ** 2 for w in weights) / len(weights)
        return round(math.sqrt(variance), 4)

    def stats(self) -> dict:
        return {
            "total_events": len(self._events),
            "total_participant_slots": sum(
                len(e.entity_weights) for e in self._events.values()
            ),
        }
