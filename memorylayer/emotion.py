"""
MemoryLayer — Emotion Classifier

Two classifiers available:

  classify(text)               — rule-based, zero dependencies (always available)
  neural_emotional_weight(text)— j-hartmann/emotion-english-distilroberta-base via
                                 HuggingFace transformers (~10ms on CPU, lazy-loaded)

NEURAL_AVAILABLE is set to True/False on first call to neural_emotional_weight().
Falls back gracefully if transformers is not installed.
"""

import os
import re
from dataclasses import dataclass
from typing import Optional

# ── Load .env without requiring python-dotenv ─────────────────────────────────
_env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")
if os.path.exists(_env_path):
    with open(_env_path) as _ef:
        for _line in _ef:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _v = _line.split("=", 1)
                os.environ.setdefault(_k.strip(), _v.strip())

# ── Neural classifier — lazy-loaded on first use ─────────────────────────────
_neural_pipeline = None
NEURAL_AVAILABLE: Optional[bool] = None  # None = not yet attempted


def _load_neural():
    """Lazy-load the HuggingFace emotion pipeline. Safe to call multiple times."""
    global _neural_pipeline, NEURAL_AVAILABLE
    if NEURAL_AVAILABLE is not None:
        return _neural_pipeline
    try:
        hf_token = os.getenv("HF_TOKEN")
        if hf_token:
            from huggingface_hub import login as hf_login
            hf_login(token=hf_token, add_to_git_credential=False)
        from transformers import pipeline as hf_pipeline
        _neural_pipeline = hf_pipeline(
            "text-classification",
            model="j-hartmann/emotion-english-distilroberta-base",
            top_k=None,   # return all labels (replaces deprecated return_all_scores=True)
            device=-1,    # CPU
        )
        NEURAL_AVAILABLE = True
    except Exception:
        _neural_pipeline = None
        NEURAL_AVAILABLE = False
    return _neural_pipeline


def neural_emotional_weight(text: str) -> float:
    """
    Score emotional weight using distilroberta-base emotion model.
    Returns 0.0–1.0. Falls back to rule-based classify() if model unavailable.

    High-emotion labels: fear, anger, sadness, surprise, joy
    Neutral label: neutral
    """
    pipe = _load_neural()
    if pipe is None:
        return classify(text).weight

    output = pipe(text[:512])
    # top_k=None → [[{label, score}, ...]]  (list-of-lists for single input)
    # Older API fallback: [{label, score}, ...]  (flat list of dicts)
    scores = output[0] if isinstance(output[0], list) else output
    high_labels = {"fear", "anger", "sadness", "surprise", "joy"}
    raw = sum(s["score"] for s in scores if s["label"].lower() in high_labels)
    return round(min(1.0, raw), 3)


@dataclass
class EmotionResult:
    weight: float           # 0.0–1.0
    dominant_emotion: str   # e.g. "grief", "joy", "neutral"
    confidence: float       # how certain the classifier is
    signals: list           # list of matched keywords


# High-emotion keywords grouped by category, each with a base score
HIGH_EMOTION = {
    # Grief / loss (strong negative)
    "died": 0.92, "death": 0.92, "dead": 0.90, "killed": 0.90,
    "murdered": 0.92, "suicide": 0.92, "funeral": 0.85, "grief": 0.88,
    "mourning": 0.85, "loss": 0.78, "lost": 0.72, "miss": 0.70,
    "heartbroken": 0.88, "devastated": 0.90, "destroyed": 0.85,
    "shattered": 0.85, "crushed": 0.82,

    # Fear / trauma
    "terrified": 0.88, "scared": 0.80, "afraid": 0.78, "fear": 0.80,
    "panic": 0.85, "trauma": 0.88, "nightmare": 0.82, "abuse": 0.90,
    "violence": 0.88, "attacked": 0.88, "threatened": 0.82,
    "helpless": 0.82, "trapped": 0.80,

    # Rage / anger
    "furious": 0.85, "rage": 0.88, "angry": 0.75, "hate": 0.82,
    "despise": 0.82, "betrayed": 0.90, "betrayal": 0.90,
    "lied": 0.78, "cheated": 0.82, "humiliated": 0.85,

    # Joy / love (strong positive)
    "love": 0.82, "loved": 0.82, "wedding": 0.85, "married": 0.82,
    "born": 0.88, "birth": 0.88, "baby": 0.82, "daughter": 0.72,
    "son": 0.72, "mother": 0.70, "father": 0.70, "family": 0.65,
    "proud": 0.75, "achieved": 0.72, "graduated": 0.80,
    "promotion": 0.75, "won": 0.72, "victory": 0.78, "success": 0.68,
    "celebrated": 0.75, "happy": 0.70, "joy": 0.78, "ecstatic": 0.88,
    "overjoyed": 0.88, "elated": 0.85,

    # Shock / surprise
    "shocked": 0.80, "surprised": 0.65, "unexpected": 0.65,
    "accident": 0.82, "suddenly": 0.70, "overnight": 0.55,

    # Passing / death euphemisms
    "passed away": 0.92, "passed on": 0.90, "gone forever": 0.92,
    "no longer with us": 0.90, "taken from us": 0.90,

    # Regret / guilt
    "regret": 0.78, "guilty": 0.78, "ashamed": 0.80, "sorry": 0.65,
    "mistake": 0.65, "forgive": 0.72, "forgiveness": 0.75,

    # Hope / longing
    "hope": 0.62, "dream": 0.60, "wish": 0.58, "pray": 0.65,
    "finally": 0.60, "waited": 0.58,
}

# Medium-emotion — contextual weight
MED_EMOTION = {
    "important": 0.40, "remember": 0.38, "never forget": 0.72,
    "best day": 0.70, "worst day": 0.78, "best moment": 0.70,
    "first time": 0.55, "last time": 0.60, "always": 0.40,
    "never": 0.45, "changed": 0.50, "realized": 0.48,
    "decided": 0.42, "promised": 0.65, "swore": 0.68,
}

# Low / neutral
LOW_EMOTION = {
    "meeting": 0.15, "report": 0.10, "email": 0.10,
    "task": 0.10, "work": 0.15, "project": 0.15,
    "reminder": 0.12, "note": 0.10, "update": 0.10,
    "schedule": 0.10, "plan": 0.12,
}

INTENSIFIERS = {
    "extremely": 1.4, "absolutely": 1.35, "incredibly": 1.35,
    "deeply": 1.3, "completely": 1.25, "totally": 1.2,
    "very": 1.15, "really": 1.10, "quite": 1.05,
    "so": 1.08, "such": 1.05,
}

DIMINISHERS = {
    "slightly": 0.6, "a bit": 0.65, "somewhat": 0.70,
    "barely": 0.5, "hardly": 0.5, "not very": 0.55,
    "kind of": 0.72, "sort of": 0.72,
}

NEGATIONS = {"not", "never", "no", "don't", "didn't", "wasn't",
             "isn't", "aren't", "won't", "can't", "couldn't", "wouldn't"}


def classify(text: str) -> EmotionResult:
    """Classify the emotional weight of a text string."""
    if not text or not text.strip():
        return EmotionResult(weight=0.0, dominant_emotion="neutral",
                             confidence=1.0, signals=[])

    text_lower = text.lower()
    words = re.findall(r"\b\w+\b", text_lower)
    word_set = set(words)

    scores = []
    signals = []
    dominant = "neutral"
    dominant_score = 0.0

    # Check negation context: words within 3 positions before each keyword
    word_positions = {w: [i for i, x in enumerate(words) if x == w] for w in word_set}

    def _is_negated(keyword_pos: int) -> bool:
        window = max(0, keyword_pos - 3)
        return any(words[i] in NEGATIONS for i in range(window, keyword_pos))

    # Score high-emotion keywords
    for phrase, score in {**HIGH_EMOTION, **MED_EMOTION}.items():
        phrase_words = phrase.split()
        if len(phrase_words) == 1:
            kw = phrase_words[0]
            if kw in word_set:
                positions = word_positions.get(kw, [])
                for pos in positions:
                    effective = score * (0.5 if _is_negated(pos) else 1.0)
                    # Apply intensifier/diminisher from previous word
                    if pos > 0:
                        prev = words[pos - 1]
                        effective *= INTENSIFIERS.get(prev, 1.0)
                        effective *= DIMINISHERS.get(prev, 1.0)
                    scores.append(effective)
                    signals.append(f"{kw}({effective:.2f})")
                    if effective > dominant_score:
                        dominant_score = effective
                        dominant = kw
        else:
            # Multi-word phrase
            if phrase in text_lower:
                scores.append(score)
                signals.append(f'"{phrase}"({score:.2f})')
                if score > dominant_score:
                    dominant_score = score
                    dominant = phrase

    # Punctuation bonus
    exclamations = len(re.findall(r"!", text))
    questions    = len(re.findall(r"\?", text))
    all_caps     = len(re.findall(r"\b[A-Z]{3,}\b", text))
    punc_bonus   = min(0.15, exclamations * 0.04 + questions * 0.02 + all_caps * 0.03)
    if punc_bonus > 0:
        scores.append(punc_bonus)
        signals.append(f"punctuation(+{punc_bonus:.2f})")

    # Combine: take max of top scores + weighted average of rest
    if not scores:
        return EmotionResult(weight=0.10, dominant_emotion="neutral",
                             confidence=0.9, signals=[])

    scores_sorted = sorted(scores, reverse=True)
    top           = scores_sorted[0]
    rest_avg      = sum(scores_sorted[1:]) / max(len(scores_sorted) - 1, 1)
    weight        = min(1.0, top * 0.7 + rest_avg * 0.3)

    confidence = min(1.0, 0.5 + len(signals) * 0.1)

    return EmotionResult(
        weight=round(weight, 3),
        dominant_emotion=dominant,
        confidence=round(confidence, 2),
        signals=signals[:6],  # top 6 for display
    )
