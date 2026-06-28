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
MemoryLayer — Fidelity-Degraded Content Retrieval  (Patent Claim 2)

Retrieved memory content is dynamically compressed in specificity
proportional to computed temporal strength.

  VIVID   (>0.70) : "I clearly remember: [full verbatim content]"
  CLEAR   (0.45–) : "I remember: [full content]"
  FADED   (0.22–) : "I vaguely recall: [first sentence only]"
  VAGUE   (0.08–) : "I have a faint sense that: [one-line gist]"
  FEELING (0.03–) : "I don't remember what happened, but I remember how it felt — [emotion descriptor]"
  FORGOTTEN(<.03) : Not returned to AI at all
"""

import re
from dataclasses import dataclass
from .decay import FidelityLevel


@dataclass
class FidelityResult:
    fidelity: FidelityLevel
    prefix_phrase: str          # "I clearly remember:", "I vaguely recall:", …
    displayed_content: str      # what gets injected into the LLM prompt
    original_length: int
    displayed_length: int
    compression_ratio: float


PREFIX = {
    FidelityLevel.VIVID:     "I clearly remember",
    FidelityLevel.CLEAR:     "I remember",
    FidelityLevel.FADED:     "I vaguely recall",
    FidelityLevel.VAGUE:     "I have a faint sense that",
    FidelityLevel.FEELING:   "I don't remember what happened, but I remember how it felt —",
    FidelityLevel.FORGOTTEN: "",
}


def degrade_content(content: str, fidelity: FidelityLevel) -> FidelityResult:
    """
    Apply fidelity degradation to content string.
    Returns a FidelityResult with appropriately compressed text and prefix phrase.
    """
    original_len = len(content)
    prefix = PREFIX[fidelity]

    if fidelity == FidelityLevel.FORGOTTEN:
        return FidelityResult(
            fidelity=fidelity,
            prefix_phrase="",
            displayed_content="",
            original_length=original_len,
            displayed_length=0,
            compression_ratio=0.0,
        )

    if fidelity in (FidelityLevel.VIVID, FidelityLevel.CLEAR):
        shown = content
    elif fidelity == FidelityLevel.FADED:
        shown = _first_sentence(content)
    elif fidelity == FidelityLevel.VAGUE:
        shown = _gist(content)
    elif fidelity == FidelityLevel.FEELING:
        # Emotion survives — details do not. Return only the emotional descriptor.
        shown = _emotional_feeling(content)
    else:
        shown = ""

    displayed = f"{prefix}: {shown}" if shown else ""

    return FidelityResult(
        fidelity=fidelity,
        prefix_phrase=prefix,
        displayed_content=displayed,
        original_length=original_len,
        displayed_length=len(shown),
        compression_ratio=round(len(shown) / max(original_len, 1), 3),
    )


# ── Content extractors ────────────────────────────────────────────────────────

def _first_sentence(text: str) -> str:
    """Extract just the first sentence."""
    text = text.strip()
    m = re.search(r"[.!?](?:\s|$)", text)
    if m:
        return text[:m.end()].strip()
    return text[:100].strip() + ("…" if len(text) > 100 else "")


def _gist(text: str) -> str:
    """Very short gist: subject + verb core (first clause or first 8 words)."""
    text = text.strip()
    comma = text.find(",")
    if 10 < comma < 80:
        return text[:comma].strip()
    words = text.split()
    short = " ".join(words[:8])
    if len(words) > 8:
        short += "…"
    return short


def _emotional_feeling(content: str) -> str:
    """
    Return a 2-3 word emotional descriptor.
    The specific episodic detail is lost — only the emotional quality survives.
    This models the somatic trace of early childhood memories.
    """
    c = content.lower()

    # Death / grief
    if any(w in c for w in ["died", "death", "dead", "passed away", "funeral",
                              "grief", "mourning", "loss", "murder", "killed"]):
        return "profound grief and loss"
    # Fear / trauma
    if any(w in c for w in ["terrified", "scared", "panic", "abuse", "attacked",
                              "helpless", "trapped", "hurt", "trauma", "nightmare"]):
        return "fear and helplessness"
    # Rage / betrayal
    if any(w in c for w in ["furious", "rage", "angry", "betrayed", "humiliated",
                              "lied", "cheated", "hate"]):
        return "anger and hurt"
    # Joy / love
    if any(w in c for w in ["love", "joy", "happy", "ecstatic", "overjoyed",
                              "celebrated", "won", "born", "birth", "wedding"]):
        return "warmth and joy"
    # Pride / achievement
    if any(w in c for w in ["proud", "achieved", "graduated", "promotion", "won",
                              "victory", "success"]):
        return "pride and excitement"
    # Sadness / longing
    if any(w in c for w in ["miss", "sad", "heartbroken", "devastated", "lonely",
                              "regret", "cried", "cry"]):
        return "sadness and longing"
    # Surprise / shock
    if any(w in c for w in ["shocked", "surprised", "unexpected", "suddenly",
                              "accident"]):
        return "surprise and disorientation"
    # Generic fallback — keep first 3 words stripped of punctuation
    words = re.sub(r"[^\w\s]", "", content).split()
    return " ".join(words[:3]).lower() if words else "something significant"


def build_prompt_context(retrieved_memories: list) -> str:
    """
    Build the memory context block for injection into an LLM system prompt.
    Each memory is labelled with its fidelity so the LLM calibrates confidence.
    FEELING memories are included but labelled as somatic traces.
    """
    if not retrieved_memories:
        return ""

    lines = ["[MEMORY CONTEXT — respond with appropriate certainty]\n"]
    for i, rm in enumerate(retrieved_memories, 1):
        f = rm.decay.fidelity.value
        lines.append(f"Memory {i} [{f}]: {rm.display_content}")

    lines.append(
        "\n[If a memory is FADED or VAGUE, express uncertainty naturally. "
        "If FEELING, acknowledge only the emotion — never assert what happened. "
        "Never assert vague memories as facts. FORGOTTEN = not mentioned.]"
    )
    return "\n".join(lines)
