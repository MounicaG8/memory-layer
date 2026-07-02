License: Apache 2.0 | Patent pending: IN/PA/2026/202641071906
# MemoryLayer

AI memory that decays, degrades, and drifts the way human memory does.

## Install

```bash
pip install memorydrift
pip install "memorydrift[embed]"
pip install "memorydrift[full]"
```

## Quickstart

```python
from memorylayer import MemoryStore, MemoryLayer

store = MemoryStore(auto_embed=True)

store.write("I saved the blacksmith's son", emotional_weight=0.9, age_at_encoding=28)
store.write("The dragon was defeated in the northern tower", emotional_weight=0.7)

for r in store.retrieve():
    print(r.display_content)   # "I clearly remember: I saved the blacksmith's son"
    print(r.decay.fidelity)    # VIVID / CLEAR / FADED / VAGUE / FEELING
```

Spreading activation fires automatically — semantically similar memories surface together.

## What makes it different

Every other AI memory system is a key-value store. MemoryLayer implements the neuroscience:

| Human memory property | MemoryLayer |
|---|---|
| Emotional memories last longer | `emotional_weight` stretches halflife up to 3× |
| Details fade before the gist | Fidelity degrades: VIVID → CLEAR → FADED → VAGUE → FEELING |
| Rehearsal keeps memories strong | `repetition_count` boosts stability (spaced repetition) |
| Related memories surface together | Spreading activation via semantic embeddings |
| Weak memories drift on recall | Reconsolidation: content rewrites toward current context |
| Childhood memories are hazy | Age-at-encoding: childhood amnesia + reminiscence bump (Rubin 1997) |
| Sleep consolidates the day | `simulate_sleep()` adds replay repetitions, tags CLEAR+ for promotion |

## Memory layers

| Layer | Halflife | Use for |
|---|---|---|
| `WORKING` | ~1 hour | Active conversation context |
| `EPISODIC` | 1 year | Events and experiences |
| `SEMANTIC` | 5 years | Facts and knowledge |
| `PROSPECTIVE` | 3 days | Intentions that fade if unacted on |
| `IDENTITY` | Forever | Core beliefs and self-concept |

## Advanced API

```python
from memorylayer import (
    # Prospective memory — intentions
    write_intention, complete_intention, get_overdue_intentions,

    # Collective memory — shared events, per-entity emotional weight
    CollectiveMemoryStore,

    # Sleep consolidation
    simulate_sleep, SleepSession,

    # Reconsolidation — memory drift on recall
    reconsolidate_sync,
)

# Prospective memory
store = MemoryStore()
intention = write_intention(store, "Call the lawyer", due_in_days=3)
complete_intention(store, intention.id)   # converts to episodic on completion

# Collective memory
shared = CollectiveMemoryStore()
event = shared.write_event(
    "The team won the championship",
    participants={"alice": 0.95, "bob": 0.40},   # alice scored; bob watched
)
alice_memories = shared.recall_for("alice")       # stronger than bob's

# Sleep consolidation
result = simulate_sleep(store, SleepSession(quality=0.85, duration_hours=7.5))
print(f"Strengthened: {result.memories_strengthened}")
```

## Fidelity levels

When a memory is retrieved, its content is automatically degraded to match its strength:

```
VIVID    (> 0.70)  "I clearly remember: she smiled at the coffee shop."
CLEAR    (0.45–)   "I remember: something about a coffee shop."
FADED    (0.22–)   "I vaguely recall: she smiled."
VAGUE    (0.08–)   "I have a faint sense that: she smiled…"
FEELING  (0.03–)   "I don't remember what happened, but I remember warmth and joy."
FORGOTTEN(< 0.03)  [not returned — the memory is gone]
```

## Patent

MemoryLayer implements three novel mechanisms filed with the Indian Patent Office (June 2026):

1. **Emotionally-modulated decay floor** — high-emotion memories asymptotically approach a floor rather than decaying to zero.
2. **Fidelity-degraded retrieval** — content specificity decreases proportionally to computed temporal strength.
3. **Post-retrieval reconsolidation** — weak memories are rewritten toward current context on each recall.

---

*Author: Mounica Goriparti · mounica.goriparti@gmail.com*
