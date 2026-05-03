"""
NLG Safe Vocabulary Guide — MultiSync v2.2

Forbidden: causes/drives/determines/controls/leads to/results in
Allowed:  precedes temporally / predicts / anticipates / covaries with
"""

TEMPORAL_VERBS = {
    "precedes": ["precedes temporally", "temporally precedes", "comes before"],
    "predicts": ["predicts", "is a leading indicator for", "provides early signal for", "anticipates"],
    "aligns": ["aligns with", "covaries with", "is synchronized with"],
    "follows": ["follows", "lags behind", "is temporally subsequent to"],
}

STRENGTH_WORDS = {"strong": "strongly", "moderate": "moderately", "weak": "weakly"}
DIRECTION_WORDS = {
    "positive": {"verb": "covaries positively with", "adjective": "positive"},
    "negative": {"verb": "covaries negatively with", "adjective": "negative"},
}

FORBIDDEN_WORDS = [
    "causes", "cause", "drives", "drive", "makes", "made",
    "determines", "determine", "controls", "lead to", "result in",
]


def describe_edge(source, target, lag_sec, polarity="positive", strength="moderate", p_value=None):
    dir_word = DIRECTION_WORDS[polarity]
    strength_adv = STRENGTH_WORDS.get(strength, "moderately")
    sig = " (highly significant)" if p_value and p_value < 0.001 else ""
    if lag_sec < 0.5:
        return f"{source} and {target} are nearly synchronous{sig}. Their {dir_word['adjective']} covariance suggests tight temporal coupling."
    elif lag_sec < 3.0:
        verb = TEMPORAL_VERBS["precedes"][0]
        return f"{source} {strength_adv} {verb} {target} by {lag_sec:.1f}s{sig}. {source} provides early signal for {target}."
    else:
        verb = TEMPORAL_VERBS["precedes"][0]
        return f"{source} {strength_adv} {verb} {target} by {lag_sec:.1f}s{sig}. This is a slow anticipatory trend, not an instantaneous coupling."


def describe_driver_score(modality, driver_score, out_degree, in_degree):
    if driver_score > 1:
        return f"{modality} acts as a driver: it temporally precedes {out_degree} other modalities (out-degree={out_degree}, in-degree={in_degree}). This is a network hub, not necessarily a causal source."
    elif driver_score < -1:
        return f"{modality} acts as a follower: it is preceded by {in_degree} other modalities (in-degree={in_degree}, out-degree={out_degree})."
    else:
        return f"{modality} has balanced influence: out-degree={out_degree}, in-degree={in_degree}. No clear temporal leader/follower pattern."


def validate_nlg_text(text):
    """Raise ValueError if text contains forbidden causal language."""
    text_lower = text.lower()
    for word in FORBIDDEN_WORDS:
        if word in text_lower:
            raise ValueError(
                f"NLG text contains FORBIDDEN causal word: '{word}'. "
                f"Use temporal language instead (precedes, predicts, anticipates)."
            )
    return True
