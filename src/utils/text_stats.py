import math
import re
from collections import Counter

_WORD_RE = re.compile(r"[A-Za-z']+")


def tokenize(text):
    return [t.lower() for t in _WORD_RE.findall(text)]


def sentence_split(text):
    chunks = re.split(r"(?<=[.!?])\s+", text.strip())
    return [c for c in chunks if c]


def avg_sentence_length(text):
    sents = sentence_split(text)
    if not sents:
        return 0.0
    return sum(len(tokenize(s)) for s in sents) / len(sents)


def cosine_similarity_counter(a, b):
    if not a or not b:
        return 0.0
    keys = set(a.keys()) | set(b.keys())
    dot = sum(a.get(k, 0) * b.get(k, 0) for k in keys)
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def bow_cosine(text_a, text_b):
    return cosine_similarity_counter(Counter(tokenize(text_a)), Counter(tokenize(text_b)))


def distinct_ngrams(tokens, n=2):
    if len(tokens) < n:
        return 0.0
    grams = [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
    return len(set(grams)) / len(grams)


def pseudo_perplexity(text, style):
    tokens = tokenize(text)
    if not tokens:
        return 999.0
    style_lexicon = {
        "dostoevsky": {"soul", "shame", "conscience", "suffering", "sin", "wretched", "confession"},
        "proust": {"memory", "remembrance", "odor", "evening", "silence", "time", "carriage", "chamber"},
    }
    matches = sum(1 for t in tokens if t in style_lexicon.get(style, set()))
    ratio = matches / len(tokens)
    return round(60.0 / (1.0 + 12.0 * ratio), 3)
