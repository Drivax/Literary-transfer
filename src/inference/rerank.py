from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from utils.text_stats import bow_cosine


def style_strength(text, style):
    lexicon = {
        "dostoevsky": ["soul", "conscience", "shame", "sentence", "suffering", "confession"],
        "proust": ["memory", "evening", "fragrance", "silence", "time", "carriage"],
    }
    text_l = text.lower()
    return sum(1 for w in lexicon.get(style, []) if w in text_l) / max(1, len(lexicon.get(style, [])))


def rank_candidates(source, candidates, style, alpha=0.55, beta=0.45):
    scored = []
    for cand in candidates:
        s_style = style_strength(cand, style)
        s_sem = bow_cosine(source, cand)
        score = alpha * s_style + beta * s_sem
        scored.append((score, cand, s_style, s_sem))
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored
