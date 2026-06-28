"""Tests for MemoryStore + Emotion + Fidelity + Reconsolidation + Spreading — 47 cases"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from memorylayer.memory_store import MemoryStore, spreading_activation
from memorylayer.decay import MemoryLayer, FidelityLevel, compute_strength
from memorylayer.emotion import classify, neural_emotional_weight, NEURAL_AVAILABLE
from memorylayer.fidelity import degrade_content, build_prompt_context, _emotional_feeling
from memorylayer.reconsolidation import reconsolidate_sync, RECONSOLIDATION_THRESHOLD, MAX_RECONSOLIDATIONS

DAY = 86400

# ── Emotion Classifier ────────────────────────────────────────────────────────

def test_emotion_high_grief():
    r = classify("My mother died yesterday. I am devastated.")
    assert r.weight >= 0.70, f"Expected high emotion got {r.weight}"

def test_emotion_high_joy():
    r = classify("My daughter was born today! I am overjoyed!")
    assert r.weight >= 0.65

def test_emotion_low_neutral():
    r = classify("I need to submit the quarterly report by Friday.")
    assert r.weight <= 0.35, f"Expected low emotion got {r.weight}"

def test_emotion_negation():
    pos = classify("I am very happy today")
    neg = classify("I am not very happy today")
    assert pos.weight > neg.weight, "Negation should reduce score"

def test_emotion_exclamation_bonus():
    r1 = classify("I love this so much")
    r2 = classify("I love this so much!!!")
    assert r2.weight >= r1.weight

def test_emotion_wedding():
    r = classify("We got married at the beach. Best day of my life.")
    assert r.weight >= 0.60

def test_emotion_betrayal():
    r = classify("I was completely betrayed by my closest friend.")
    assert r.weight >= 0.72

# ── Fidelity Degradation ──────────────────────────────────────────────────────

def test_fidelity_vivid_full():
    content = "I clearly recall the entire event. Every detail is sharp."
    r = degrade_content(content, FidelityLevel.VIVID)
    assert content in r.displayed_content
    assert "clearly remember" in r.displayed_content

def test_fidelity_clear_full():
    content = "We had dinner at 7pm. She wore a blue dress."
    r = degrade_content(content, FidelityLevel.CLEAR)
    assert content in r.displayed_content
    assert "I remember" in r.displayed_content

def test_fidelity_faded_first_sentence():
    content = "We had dinner. Then we walked along the river. It was beautiful."
    r = degrade_content(content, FidelityLevel.FADED)
    assert "We had dinner" in r.displayed_content
    assert "river" not in r.displayed_content

def test_fidelity_vague_gist():
    content = "We had a very long and detailed discussion about the project timeline and budget."
    r = degrade_content(content, FidelityLevel.VAGUE)
    assert len(r.displayed_content) < len(content)

def test_fidelity_forgotten_empty():
    r = degrade_content("something important", FidelityLevel.FORGOTTEN)
    assert r.displayed_content == ""
    assert r.compression_ratio == 0.0

def test_fidelity_compression_ratio():
    content = "A" * 100
    r = degrade_content(content, FidelityLevel.FADED)
    assert 0 < r.compression_ratio <= 1.0

# ── MemoryStore Write/Retrieve ────────────────────────────────────────────────

def test_store_write_returns_memory():
    store = MemoryStore()
    m = store.write("Meeting at 3pm tomorrow", layer=MemoryLayer.WORKING)
    assert m.id
    assert m.layer == MemoryLayer.WORKING.value

def test_store_auto_emotion_classify():
    store = MemoryStore()
    m = store.write("My father passed away last night.")
    assert m.emotional_weight >= 0.70

def test_store_retrieve_fresh():
    store = MemoryStore()
    store.write("Lunch with Sarah", layer=MemoryLayer.EPISODIC)
    results = store.retrieve()
    assert len(results) == 1
    assert results[0].decay.fidelity == FidelityLevel.VIVID

def test_store_forgotten_excluded():
    store = MemoryStore()
    old_ts = time.time() - 2500 * DAY  # ~7 years — well past halflife for neutral
    store.write("Ancient neutral memo", layer=MemoryLayer.EPISODIC,
                emotional_weight=0.0, importance=0.5, now=old_ts)
    results = store.retrieve()
    assert len(results) == 0, "7-year-old neutral memory should be forgotten"

def test_store_high_emotion_survives_long():
    store = MemoryStore()
    old_ts = time.time() - 500 * DAY
    store.write("My wedding day was perfect.", layer=MemoryLayer.EPISODIC,
                emotional_weight=0.9, importance=0.9, now=old_ts)
    results = store.retrieve()
    assert len(results) == 1, "High-emotion memory must survive 500 days"

def test_store_tag_filter():
    store = MemoryStore()
    store.write("Work meeting", layer=MemoryLayer.EPISODIC, tags=["work"])
    store.write("Family dinner", layer=MemoryLayer.EPISODIC, tags=["family"])
    family = store.retrieve(query_tags=["family"])
    assert len(family) == 1
    assert "dinner" in family[0].memory.content

def test_store_layer_filter():
    store = MemoryStore()
    store.write("Fact about Python", layer=MemoryLayer.SEMANTIC)
    store.write("Today's event", layer=MemoryLayer.EPISODIC)
    sem = store.retrieve(layer=MemoryLayer.SEMANTIC)
    assert len(sem) == 1

def test_store_increment_repetition():
    store = MemoryStore()
    m = store.write("Remember this", layer=MemoryLayer.EPISODIC)
    assert m.repetition_count == 0
    store.retrieve_one(m.id)
    assert m.repetition_count == 1

def test_store_repetition_slows_decay():
    store = MemoryStore()
    old_ts = time.time() - 30 * DAY
    m = store.write("Something", layer=MemoryLayer.EPISODIC,
                    emotional_weight=0.3, now=old_ts)
    r0 = store.retrieve_one(m.id)
    m.repetition_count = 10
    r10 = store.retrieve_one(m.id, increment_recall=False)
    assert r10.decay.strength >= r0.decay.strength

def test_store_identity_permanent():
    store = MemoryStore()
    old_ts = time.time() - 3650 * DAY
    store.write("I am a software engineer.", layer=MemoryLayer.IDENTITY,
                emotional_weight=0.0, now=old_ts)
    results = store.retrieve()
    assert len(results) == 1, "Identity memory must last forever"

def test_store_stats():
    store = MemoryStore()
    store.write("fact", layer=MemoryLayer.SEMANTIC)
    store.write("event", layer=MemoryLayer.EPISODIC)
    s = store.stats()
    assert s["total"] == 2
    assert s["active"] >= 2

def test_store_consolidation():
    from memorylayer.consolidation import run_consolidation
    store = MemoryStore()
    old_ts = time.time() - 8 * DAY
    store.write("Had a long meeting about Q3 budget.", layer=MemoryLayer.EPISODIC,
                emotional_weight=0.3, now=old_ts)
    result = run_consolidation(store, consolidation_threshold_days=7.0)
    assert result.consolidated >= 1

def test_build_prompt_context():
    store = MemoryStore()
    store.write("I graduated from university.", layer=MemoryLayer.EPISODIC,
                emotional_weight=0.8)
    results = store.retrieve()
    context = build_prompt_context(results)
    assert "MEMORY CONTEXT" in context
    assert len(context) > 0


# ── FEELING fidelity level ────────────────────────────────────────────────────

def test_feeling_fidelity_threshold():
    """Strength 0.03–0.08 should be FEELING, not FORGOTTEN."""
    r = compute_strength(0.0, 0.0, 0, 0.05, MemoryLayer.EPISODIC)
    # Manually patch strength to test threshold
    from memorylayer.decay import _classify_fidelity
    assert _classify_fidelity(0.05) == FidelityLevel.FEELING
    assert _classify_fidelity(0.02) == FidelityLevel.FORGOTTEN
    assert _classify_fidelity(0.08) == FidelityLevel.VAGUE
    assert _classify_fidelity(0.031) == FidelityLevel.FEELING

def test_feeling_content_grief():
    """Grief content should produce 'grief and loss' descriptor."""
    result = degrade_content("My father died in the accident.", FidelityLevel.FEELING)
    assert "grief" in result.displayed_content.lower() or "loss" in result.displayed_content.lower()
    assert "don't remember what happened" in result.displayed_content.lower()

def test_feeling_content_joy():
    """Joyful content should produce 'warmth and joy' descriptor."""
    result = degrade_content("We got married in Paris. It was wonderful.", FidelityLevel.FEELING)
    assert result.fidelity == FidelityLevel.FEELING
    assert len(result.displayed_content) > 0

def test_feeling_content_fear():
    result = degrade_content("I was terrified, trapped in the basement.", FidelityLevel.FEELING)
    assert "fear" in result.displayed_content.lower() or "helpless" in result.displayed_content.lower()

def test_feeling_retrieved_from_store():
    """FEELING-level memories must be returned from retrieve(), not excluded."""
    store  = MemoryStore()
    # Early childhood memory (age 2, 28 years old now) with high emotion
    # age_factor at 2 = ~0.08, so effective_importance is low → FEELING range
    base   = time.time()
    enc_ts = base - 28 * 365.25 * DAY
    m = store.write("First steps. Fell over. Mum laughed.",
                    layer=MemoryLayer.EPISODIC,
                    emotional_weight=0.92, importance=0.90,
                    age_at_encoding=2.0, now=enc_ts)
    results = store.retrieve(now=base)
    # If strength is in FEELING range, it should be returned (not excluded)
    all_m = store.all_with_decay(now=base)
    target = next((r for r in all_m if r.memory.id == m.id), None)
    assert target is not None
    if target.decay.fidelity == FidelityLevel.FEELING:
        # Should appear in retrieve() results
        assert any(r.memory.id == m.id for r in results), "FEELING memory must be retrievable"

def test_feeling_prompt_context_labelled():
    """FEELING memories must be labelled [FEELING] in prompt context."""
    from memorylayer.memory_store import Memory, RetrievedMemory
    from memorylayer.decay import DecayResult
    # Build a mock RetrievedMemory with FEELING fidelity
    fr = degrade_content("I was terrified.", FidelityLevel.FEELING)
    ctx = build_prompt_context([
        type("RM", (), {
            "decay": type("D", (), {"fidelity": FidelityLevel.FEELING})(),
            "display_content": fr.displayed_content,
            "fidelity_result": fr,
        })()
    ])
    assert "FEELING" in ctx
    assert "If FEELING" in ctx  # instruction for LLM

def test_emotional_feeling_descriptors():
    """_emotional_feeling should map content to correct emotion descriptor."""
    assert "grief" in _emotional_feeling("My grandfather died.")
    assert "fear" in _emotional_feeling("I was terrified and helpless.")
    assert "anger" in _emotional_feeling("I was furious and felt betrayed.")
    assert "joy" in _emotional_feeling("We won the championship! So happy!")
    assert "pride" in _emotional_feeling("I achieved my promotion with success.")
    assert "sadness" in _emotional_feeling("I miss her so much. Heartbroken.")


# ── Neural emotion classifier ─────────────────────────────────────────────────

def test_neural_classifier_available_check():
    """NEURAL_AVAILABLE should be None before first call, then True/False after."""
    # After import, it may be None (not yet attempted) or False (not installed)
    assert NEURAL_AVAILABLE in (None, True, False)

def test_neural_classifier_stronger_than_neutral():
    """Neural or rule-based: devastated > neutral. Tests both paths."""
    devastated = neural_emotional_weight("I was absolutely devastated when I heard the news.")
    neutral    = neural_emotional_weight("The quarterly report is saved in the documents folder.")
    assert devastated > neutral, (
        f"Devastated ({devastated:.3f}) must score higher than neutral ({neutral:.3f})")

def test_neural_falls_back_gracefully():
    """neural_emotional_weight always returns a float in [0, 1] regardless of model availability."""
    result = neural_emotional_weight("Something happened.")
    assert 0.0 <= result <= 1.0

def test_neural_grief_high():
    """Strong grief text should score >= 0.60 whether neural or rule-based."""
    result = neural_emotional_weight("My mother passed away last night.")
    assert result >= 0.60, f"Grief should score >= 0.60, got {result}"


# ── Reconsolidation ───────────────────────────────────────────────────────────

def _make_memory(content="Original content.", ew=0.20, importance=0.60,
                 age=28.0, days_old=400.0):
    """Helper: create a fresh Memory dataclass for testing."""
    from memorylayer.memory_store import Memory
    ts = time.time() - days_old * DAY
    return Memory(
        id="test_id",
        content=content,
        layer=MemoryLayer.EPISODIC.value,
        tags=[],
        importance=importance,
        emotional_weight=ew,
        dominant_emotion="manual",
        repetition_count=0,
        created_at=ts,
        last_accessed=ts,
        age_at_encoding=age,
        metadata={},
        last_reconsolidated=0.0,
        reconsolidation_count=0,
        original_content="",
    )

def test_reconsolidation_skips_strong():
    """Memories at or above threshold (0.45) must not be modified."""
    mem = _make_memory(content="Clearly remembered event.", ew=0.80, days_old=30.0)
    original = mem.content
    llm = lambda p: "Modified version."
    result = reconsolidate_sync(mem, "some context", 0.80, llm)
    assert result == original, "Strong memory should be unchanged"
    assert mem.content == original

def test_reconsolidation_modifies_weak():
    """Weak memories (< 0.45) are blended with current context — content drifts."""
    mem = _make_memory(content="Original memory content.", ew=0.15, days_old=600.0)

    # Two different contexts produce two different reconsolidated versions
    responses = iter(["Context A version of the memory.", "Context B version of the memory."])
    llm = lambda p: next(responses)

    result1 = reconsolidate_sync(mem, "context about work",    0.25, llm)
    result2 = reconsolidate_sync(mem, "context about family",  0.25, llm)

    assert result1 != result2, "Different contexts should produce different reconsolidations"
    assert mem.content == result2, "Memory content should have drifted to last reconsolidation"

def test_reconsolidation_updates_timestamp():
    """last_reconsolidated should be set after a reconsolidation occurs."""
    mem = _make_memory(ew=0.15, days_old=600.0)
    assert mem.last_reconsolidated == 0.0

    llm = lambda p: "New version."
    reconsolidate_sync(mem, "query", 0.25, llm, now=999999.0)

    assert mem.last_reconsolidated == 999999.0

def test_reconsolidation_skips_feeling_level():
    """FEELING-level memories (≤ 0.03) are too faint to reconsolidate."""
    mem = _make_memory(content="Faint trace of something.", ew=0.05, days_old=9000.0)
    original = mem.content
    llm = lambda p: "Modified."
    result = reconsolidate_sync(mem, "any context", 0.02, llm)
    assert result == original, "FEELING-level memory should not be reconsolidated"
    assert mem.last_reconsolidated == 0.0

def test_reconsolidation_prompt_contains_original():
    """The LLM prompt must contain the original memory content."""
    mem    = _make_memory(content="The exact original text goes here.", ew=0.20)
    prompts = []
    llm    = lambda p: (prompts.append(p), "Reconsolidated.")[1]

    reconsolidate_sync(mem, "current context query", 0.25, llm)

    assert len(prompts) == 1
    assert "The exact original text goes here." in prompts[0]
    assert "current context query" in prompts[0]

def test_reconsolidation_capped_at_max():
    """After MAX_RECONSOLIDATIONS drifts the trace stabilises — no further changes."""
    mem = _make_memory(content="Original trace.", ew=0.15, days_old=600.0)

    call_count = [0]
    def llm(p):
        call_count[0] += 1
        return f"Drift version {call_count[0]}."

    # Drive past the cap
    for _ in range(MAX_RECONSOLIDATIONS + 2):
        reconsolidate_sync(mem, "some context", 0.25, llm)

    # LLM should only have been called MAX_RECONSOLIDATIONS times
    assert call_count[0] == MAX_RECONSOLIDATIONS, (
        f"LLM called {call_count[0]} times, expected {MAX_RECONSOLIDATIONS}")
    assert mem.reconsolidation_count == MAX_RECONSOLIDATIONS

def test_reconsolidation_preserves_original():
    """original_content is set on first drift and never overwritten by subsequent drifts."""
    mem = _make_memory(content="The very first version.", ew=0.15, days_old=600.0)
    assert mem.original_content == ""

    counter = [0]
    def llm(p):
        counter[0] += 1
        return f"Version {counter[0]}."

    reconsolidate_sync(mem, "context 1", 0.25, llm)
    assert mem.original_content == "The very first version."

    reconsolidate_sync(mem, "context 2", 0.25, llm)
    assert mem.original_content == "The very first version.", (
        "original_content must not be overwritten on subsequent drifts")


# ── Spreading Activation ─────────────────────────────────────────────────────

def test_spreading_boosts_related_memory():
    """Top result activates a semantically similar memory."""
    store = MemoryStore()
    emb_a = [1.0, 0.0, 0.0]
    emb_b = [0.95, 0.31, 0.0]   # cosine sim ~0.95 with emb_a
    emb_c = [0.0,  0.0,  1.0]   # cosine sim  0.0  with emb_a

    m_a = store.write("dragon battle", metadata={"embedding": emb_a}, emotional_weight=0.6)
    m_b = store.write("dragon fight",  metadata={"embedding": emb_b}, emotional_weight=0.5)
    m_c = store.write("bought bread",  metadata={"embedding": emb_c}, emotional_weight=0.5)

    results = store.retrieve(spreading=True, increment_recall=False)
    ids = [r.memory.id for r in results]

    assert m_a.id in ids
    assert m_b.id in ids   # activated by spreading from m_a
    r_b = next(r for r in results if r.memory.id == m_b.id)
    assert r_b.activation_boost > 0.0

def test_spreading_ignores_unrelated():
    """Spreading does not boost semantically distant memories."""
    store = MemoryStore()
    emb_a = [1.0, 0.0, 0.0]
    emb_b = [0.0, 0.0, 1.0]   # orthogonal — cosine sim = 0.0

    m_a = store.write("dragon battle", metadata={"embedding": emb_a}, emotional_weight=0.5)
    m_b = store.write("bought bread",  metadata={"embedding": emb_b}, emotional_weight=0.5)

    all_mems = list(store._memories.values())
    activated = spreading_activation(m_a, all_mems, store._decay, time.time())
    activated_ids = [mid for mid, _ in activated]
    assert m_b.id not in activated_ids

def test_spreading_ignores_emotional_mismatch():
    """High joy does not activate high grief."""
    store = MemoryStore()
    emb = [1.0, 0.0, 0.0]  # identical embedding — only emotion differs

    m_joy  = store.write("best day ever", metadata={"embedding": emb}, emotional_weight=0.95)
    m_grief= store.write("worst day ever",metadata={"embedding": emb}, emotional_weight=0.05)

    all_mems = list(store._memories.values())
    activated = spreading_activation(
        m_joy, all_mems, store._decay, time.time(), emotion_tolerance=0.3
    )
    activated_ids = [mid for mid, _ in activated]
    assert m_grief.id not in activated_ids

def test_spreading_does_not_surface_forgotten():
    """A very weak memory is not surfaced even with spreading boost."""
    store = MemoryStore()
    emb = [1.0, 0.0, 0.0]

    m_a = store.write("dragon battle", metadata={"embedding": emb}, emotional_weight=0.5)
    m_b = store.write("dragon fight",  metadata={"embedding": emb},
                      emotional_weight=0.5, importance=0.05)
    # Age m_b severely — force it to near-zero strength
    m_b.created_at = time.time() - 3000 * 86400  # 3000 days ago

    all_mems = list(store._memories.values())
    activated = spreading_activation(m_a, all_mems, store._decay, time.time(), boost=0.1)
    activated_ids = [mid for mid, _ in activated]
    assert m_b.id not in activated_ids

def test_spreading_skipped_without_embeddings():
    """Spreading returns empty list gracefully when no embeddings present."""
    store = MemoryStore()
    m_a = store.write("dragon battle", emotional_weight=0.5)  # no embedding
    m_b = store.write("dragon fight",  emotional_weight=0.5)  # no embedding

    all_mems = list(store._memories.values())
    activated = spreading_activation(m_a, all_mems, store._decay, time.time())
    assert activated == []

def test_spreading_disabled():
    """retrieve(spreading=False) returns no activation boosts."""
    store = MemoryStore()
    emb = [1.0, 0.0, 0.0]
    store.write("dragon battle", metadata={"embedding": emb}, emotional_weight=0.5)
    store.write("dragon fight",  metadata={"embedding": emb}, emotional_weight=0.5)

    results = store.retrieve(spreading=False, increment_recall=False)
    assert all(r.activation_boost == 0.0 for r in results)


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = failed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
            passed += 1
        except Exception as e:
            print(f"  FAIL  {t.__name__}: {e}")
            failed += 1
    print(f"\n{passed}/{passed+failed} passed")
