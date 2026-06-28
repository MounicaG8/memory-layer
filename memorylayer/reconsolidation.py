"""
MemoryLayer — Memory Reconsolidation

Every time a weak memory is recalled, it is slightly modified by current context.
Strong memories (≥ 0.45) resist modification — they are stable and reliable.
Weak memories fill gaps from the current context — they drift over time.

Research basis:
  Nader & Hardt (2009)  — reconsolidation as labile post-retrieval window
  Dudai (2012)           — reconsolidation: a circuit-level analysis
  Hupbach et al. (2008) — contextual cueing drives memory modification

Patent relevance:
  Third novel mechanism beyond Claim 1 (decay+floor) and Claim 2 (fidelity).
  No other AI memory system implements post-retrieval content drift.

Usage:
    # Synchronous (for scripts, tests)
    new_content = reconsolidate_sync(memory, query, strength, llm_fn)

    # Async (for FastAPI, async AI pipelines)
    new_content = await reconsolidate_async(memory, query, strength, async_llm_fn)
"""

import time
from typing import Callable, Optional, Awaitable

RECONSOLIDATION_THRESHOLD = 0.45   # memories weaker than this are reconsolidated on recall
RECONSOLIDATION_FLOOR     = 0.03   # FEELING-level memories are too faint to reconsolidate
MAX_RECONSOLIDATIONS      = 3      # after this many drifts the trace stabilises


def _build_reconsolidation_prompt(
    original_content: str,
    current_query: str,
    current_strength: float,
) -> str:
    influence = round(1.0 - current_strength, 2)
    return (
        f"Original memory (confidence {current_strength:.0%}):\n"
        f"{original_content}\n\n"
        f"Current context the person is thinking about:\n"
        f"{current_query}\n\n"
        f"Given that this memory is old and {influence:.0%} uncertain, "
        f"what would the person likely remember now? "
        f"They may unconsciously fill gaps with current context. "
        f"One sentence only."
    )


def reconsolidate_sync(
    memory,                          # Memory dataclass
    current_query: str,
    current_strength: float,
    llm_fn: Callable[[str], str],
    now: Optional[float] = None,
) -> str:
    """
    Synchronous reconsolidation.

    Strong memories (≥ RECONSOLIDATION_THRESHOLD) are returned unchanged.
    FEELING-level memories (≤ RECONSOLIDATION_FLOOR) are too faint to modify.
    All others are blended with current_query via llm_fn.

    The memory.content is updated in-place — the drift is permanent.

    Args:
        memory:           Memory object being recalled
        current_query:    The query/context that triggered recall
        current_strength: Current decay strength (0.0–1.0)
        llm_fn:           Callable(prompt: str) -> str
        now:              Timestamp (default: time.time())

    Returns:
        The reconsolidated content string.
    """
    if current_strength >= RECONSOLIDATION_THRESHOLD:
        return memory.content    # strong — resists modification

    if current_strength <= RECONSOLIDATION_FLOOR:
        return memory.content    # FEELING-level — too faint to reshape

    if getattr(memory, "reconsolidation_count", 0) >= MAX_RECONSOLIDATIONS:
        return memory.content    # trace has stabilised — further drift capped

    # Preserve the original on first drift
    if not getattr(memory, "original_content", ""):
        memory.original_content = memory.content

    prompt = _build_reconsolidation_prompt(
        memory.content, current_query, current_strength
    )
    reconsolidated = llm_fn(prompt)

    memory.content              = reconsolidated
    memory.last_reconsolidated  = now or time.time()
    memory.reconsolidation_count = getattr(memory, "reconsolidation_count", 0) + 1
    return reconsolidated


async def reconsolidate_async(
    memory,
    current_query: str,
    current_strength: float,
    llm_fn: Callable[[str], Awaitable[str]],
    now: Optional[float] = None,
) -> str:
    """
    Async reconsolidation — llm_fn is an async callable.
    Same logic as reconsolidate_sync but awaits the LLM call.
    """
    if current_strength >= RECONSOLIDATION_THRESHOLD:
        return memory.content

    if current_strength <= RECONSOLIDATION_FLOOR:
        return memory.content

    if getattr(memory, "reconsolidation_count", 0) >= MAX_RECONSOLIDATIONS:
        return memory.content

    if not getattr(memory, "original_content", ""):
        memory.original_content = memory.content

    prompt = _build_reconsolidation_prompt(
        memory.content, current_query, current_strength
    )
    reconsolidated = await llm_fn(prompt)

    memory.content               = reconsolidated
    memory.last_reconsolidated   = now or time.time()
    memory.reconsolidation_count = getattr(memory, "reconsolidation_count", 0) + 1
    return reconsolidated
