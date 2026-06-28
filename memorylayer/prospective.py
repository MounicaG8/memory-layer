"""
MemoryLayer — Prospective Memory

Intentions and plans that decay quickly if not acted upon.
Halflife: 3 days — an unfulfilled intention fades within a week.
Acting on the intention converts it to an episodic memory.

Research: McDaniel & Einstein (2007) — Prospective Memory: An Overview and Synthesis
          Kliegel et al. (2008)      — Prospective Memory: Cognitive, Neuroscience, and
                                       Applied Perspectives
"""

import time
from typing import List, Optional

from .memory_store import MemoryStore, Memory, RetrievedMemory
from .decay import MemoryLayer


def write_intention(
    store: MemoryStore,
    content: str,
    due_in_days: float = 7.0,
    importance: float = 0.80,
    tags: Optional[List[str]] = None,
    now: Optional[float] = None,
) -> Memory:
    """
    Store a prospective memory (intention / plan).
    Due date is recorded in metadata["due_at"].
    """
    ts = now or time.time()
    return store.write(
        content=content,
        layer=MemoryLayer.PROSPECTIVE,
        importance=importance,
        emotional_weight=0.30,
        tags=tags or [],
        metadata={"due_at": ts + due_in_days * 86400},
        now=ts,
    )


def is_overdue(memory: Memory, now: Optional[float] = None) -> bool:
    """Return True if the intention's due date has passed without completion."""
    ts = now or time.time()
    return ts > memory.metadata.get("due_at", 0)


def complete_intention(
    store: MemoryStore,
    memory_id: str,
    now: Optional[float] = None,
) -> Memory:
    """
    Mark a prospective memory as completed.
    Removes the prospective entry and writes a new episodic memory
    capturing the act of completion.
    """
    ts = now or time.time()
    mem = store._memories.get(memory_id)
    if mem is None:
        raise KeyError(f"Memory {memory_id!r} not found")

    new_mem = store.write(
        content=f"Completed: {mem.content}",
        layer=MemoryLayer.EPISODIC,
        importance=mem.importance,
        emotional_weight=0.50,   # mild satisfaction of getting it done
        tags=list(mem.tags),
        now=ts,
    )
    del store._memories[memory_id]
    return new_mem


def get_pending_intentions(
    store: MemoryStore,
    now: Optional[float] = None,
) -> List[RetrievedMemory]:
    """Return all prospective memories that have not yet been forgotten."""
    return store.retrieve(
        layer=MemoryLayer.PROSPECTIVE,
        now=now,
        increment_recall=False,
    )


def get_overdue_intentions(
    store: MemoryStore,
    now: Optional[float] = None,
) -> List[RetrievedMemory]:
    """Return prospective memories that are past their due date."""
    ts = now or time.time()
    return [r for r in get_pending_intentions(store, now=ts)
            if is_overdue(r.memory, now=ts)]
