import math
import re
from collections import Counter


TOKEN_RE = re.compile(r"[A-Za-z']+")


def tokenize(text):
    return [t.lower() for t in TOKEN_RE.findall(text)]


def nll_proxy(source, target):
    src = tokenize(source)
    tgt = tokenize(target)
    if not tgt:
        return 10.0
    overlap = len(set(src) & set(tgt))
    score = max(0.001, overlap / len(set(tgt)))
    return -math.log(score)


def sem_loss(source, target):
    src = Counter(tokenize(source))
    tgt = Counter(tokenize(target))
    keys = set(src) | set(tgt)
    dot = sum(src.get(k, 0) * tgt.get(k, 0) for k in keys)
    ns = math.sqrt(sum(v * v for v in src.values()))
    nt = math.sqrt(sum(v * v for v in tgt.values()))
    if ns == 0 or nt == 0:
        return 1.0
    cos = dot / (ns * nt)
    return 1 - max(0.0, min(1.0, cos))


def style_loss(target, style):
    lex = {
        "dostoevsky": {"soul", "shame", "conscience", "suffering", "sin"},
        "proust": {"memory", "time", "fragrance", "evening", "silence"},
    }
    toks = tokenize(target)
    if not toks:
        return 5.0
    hits = sum(1 for t in toks if t in lex.get(style, set()))
    prob = max(0.01, hits / len(toks))
    return -math.log(prob)


def total_loss(source, target, style, lam1=1.0, lam2=0.7, lam3=0.5):
    l_nll = nll_proxy(source, target)
    l_sem = sem_loss(source, target)
    l_style = style_loss(target, style)
    return {
        "nll": l_nll,
        "sem": l_sem,
        "style": l_style,
        "total": lam1 * l_nll + lam2 * l_sem + lam3 * l_style,
    }
