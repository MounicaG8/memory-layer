"""
MemoryLayer — Reconsolidation Demo

Shows how a weak memory drifts through successive recalls in different contexts.
Each recall slightly rewrites the memory, coloured by the current emotional state.

This visualises Patent Claim 3: post-retrieval content drift.
Run:  python demo_reconsolidation.py
"""

import time
from memorylayer.memory_store import MemoryStore
from memorylayer.decay import MemoryLayer
from memorylayer.reconsolidation import reconsolidate_sync, MAX_RECONSOLIDATIONS

DAY = 86400

# ── Context scenarios ─────────────────────────────────────────────────────────

CONTEXTS = [
    ("Work stress",   "I've been exhausted by deadlines and back-to-back meetings all week."),
    ("Nostalgia",     "I've been thinking about old friends and warm memories from the past."),
    ("Recent loss",   "I just lost a close friend and everything feels heavy and lonely."),
    ("New romance",   "I've just fallen in love — everything feels light and wonderful."),
]


def mock_llm(prompt: str) -> str:
    """Context-sensitive mock that drifts the memory toward the current mood."""
    p = prompt.lower()
    if "exhausted" in p or "deadlines" in p or "meetings" in p:
        return "She looked tired and distracted — or maybe I'm just projecting my own exhaustion."
    if "old friends" in p or "nostalgia" in p or "warm memories" in p:
        return "She was smiling broadly, the warmest smile I can picture from that whole year."
    if "lost" in p or "lonely" in p or "heavy" in p:
        return "Looking back, maybe her smile had a sadness behind it I chose not to see."
    if "love" in p or "wonderful" in p or "fallen" in p:
        return "Her smile was warm, full of something unspoken — like she already knew."
    return "I remember something about a smile, but the details keep shifting."


# ── Demo runner ───────────────────────────────────────────────────────────────

def run_demo():
    store = MemoryStore()

    # Plant a weak episodic memory 600 days in the past
    enc_ts = time.time() - 600 * DAY
    m = store.write(
        "I remember seeing her smile at the coffee shop on a rainy afternoon.",
        layer=MemoryLayer.EPISODIC,
        emotional_weight=0.35,
        importance=0.45,
        now=enc_ts,
    )

    print("=" * 72)
    print("  MEMORY RECONSOLIDATION DEMO")
    print("=" * 72)
    print(f"\n  Original (600 days ago):\n    \"{m.content}\"\n")
    print(f"  {'Context':<22} {'Strength':>8}  {'Fidelity':<10}  Reconsolidated content")
    print("  " + "-" * 68)

    for label, ctx_text in CONTEXTS:
        dr = store._compute_decay(m, time.time())
        if m.reconsolidation_count >= MAX_RECONSOLIDATIONS:
            verdict = "(capped — memory trace has stabilised)"
        else:
            verdict = reconsolidate_sync(m, ctx_text, dr.strength, mock_llm)
        print(f"  {label:<22} {dr.strength:>7.3f}  {dr.fidelity.value:<10}  {verdict[:60]}")

    print("  " + "-" * 68)
    print(f"\n  Drifted content:\n    \"{m.content}\"")
    print(f"  Original preserved:\n    \"{m.original_content}\"")
    print(f"\n  Reconsolidation count: {m.reconsolidation_count}/{MAX_RECONSOLIDATIONS}")
    print("=" * 72)


if __name__ == "__main__":
    run_demo()
