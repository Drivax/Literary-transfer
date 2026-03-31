import argparse
import hashlib
import json
from pathlib import Path
import random
import sys
import re

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer
except ImportError:
    torch = None
    AutoModelForCausalLM = None
    AutoModelForSeq2SeqLM = None
    AutoTokenizer = None

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

try:
    from .rerank import rank_candidates
except ImportError:
    from rerank import rank_candidates


_MODEL_CACHE = {}

MODEL_PROFILES = {
    "fast": {
        "model_name": "distilgpt2",
        "kind": "causal",
        "n_candidates": 4,
        "max_new_tokens": 96,
        "temp_cycle": [0.65, 0.75, 0.85, 0.95],
        "top_p_cycle": [0.88, 0.9, 0.93, 0.96],
    },
    "instruct": {
        "model_name": "google/flan-t5-small",
        "kind": "seq2seq",
        "n_candidates": 6,
        "max_new_tokens": 128,
        "temp_cycle": [0.62, 0.72, 0.82, 0.92],
        "top_p_cycle": [0.9, 0.92, 0.95, 0.97],
    },
}


def normalize_style(style):
    style_key = str(style).strip().lower()
    aliases = {
        "dosto": "dostoevsky",
        "dostoevsky": "dostoevsky",
        "proust": "proust",
    }
    if style_key not in aliases:
        raise ValueError("style must be one of: dosto, dostoevsky, proust")
    return aliases[style_key]


def _stable_seed(text):
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return int(digest[:12], 16)


def _sample_topk(ranked, top_k=3, temperature=0.35, seed=None):
    if not ranked:
        raise ValueError("No ranked candidates to sample from")
    k = max(1, min(top_k, len(ranked)))
    subset = ranked[:k]

    # Softmax sampling keeps quality from reranking while avoiding template lock-in.
    scores = [row[0] for row in subset]
    s_max = max(scores)
    weights = [pow(2.718281828, (s - s_max) / max(1e-6, temperature)) for s in scores]
    if seed is None:
        rng = random.Random()
    else:
        rng = random.Random(seed)
    pick = rng.choices(subset, weights=weights, k=1)[0]
    return pick


def _resolve_profile(profile, lm_model=None):
    key = str(profile or "instruct").strip().lower()
    if key not in MODEL_PROFILES:
        raise ValueError("profile must be one of: fast, instruct")
    cfg = dict(MODEL_PROFILES[key])
    if lm_model:
        cfg["model_name"] = lm_model
        cfg["kind"] = "auto"
    return key, cfg


def _load_lm(model_name, model_kind="auto"):
    if AutoTokenizer is None or torch is None:
        raise ImportError(
            "Missing dependencies for LM generation. Install 'transformers' and 'torch'."
        )
    cache_key = (model_name, model_kind)
    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    resolved_kind = model_kind
    if model_kind == "auto":
        if "t5" in model_name.lower() or "flan" in model_name.lower():
            resolved_kind = "seq2seq"
        else:
            resolved_kind = "causal"

    if resolved_kind == "seq2seq":
        if AutoModelForSeq2SeqLM is None:
            raise ImportError("Missing AutoModelForSeq2SeqLM from transformers")
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    else:
        if AutoModelForCausalLM is None:
            raise ImportError("Missing AutoModelForCausalLM from transformers")
        model = AutoModelForCausalLM.from_pretrained(model_name)

    model.eval()
    _MODEL_CACHE[cache_key] = (tokenizer, model, resolved_kind)
    return tokenizer, model, resolved_kind


def _style_instructions(style):
    if style == "dostoevsky":
        return (
            "Write in Dostoevsky-like prose: morally introspective, psychologically intense, "
            "serious, and dramatic. Preserve all factual meaning from the source text. "
            "Do not add new events."
        )
    return (
        "Write in Proust-like prose: long flowing cadence, reflective memory-driven tone, "
        "sensory and temporal nuance. Preserve all factual meaning from the source text. "
        "Do not add new events."
    )


def _build_prompt(text, style):
    author = "Fyodor Dostoevsky" if style == "dostoevsky" else "Marcel Proust"
    return (
        f"You are a literary style transfer model.\\n"
        f"Target author: {author}.\\n"
        f"Instruction: {_style_instructions(style)}\\n\\n"
        "Rewrite the source text in the target style.\\n"
        "Return only the rewritten text.\\n\\n"
        f"Source text:\\n{text}\\n\\n"
        "Rewritten text:\\n"
    )


def _extract_rewrite(prompt, decoded):
    if decoded.startswith(prompt):
        decoded = decoded[len(prompt) :]

    markers = ["Rewritten text:", "Output:", "Answer:"]
    for marker in markers:
        if marker in decoded:
            decoded = decoded.split(marker, 1)[1]

    candidate = decoded.strip().strip('"').strip()
    # Keep only the first block to avoid continuation artifacts.
    candidate = candidate.split("\n\n", 1)[0].strip()
    return candidate


def _tokenize(text):
    return re.findall(r"[a-zA-Z']+", text.lower())


def _semantic_overlap(source, candidate):
    src = set(_tokenize(source))
    if not src:
        return 0.0
    cand = set(_tokenize(candidate))
    return len(src & cand) / len(src)


def _style_markers(style):
    if style == "dostoevsky":
        return [
            "conscience",
            "soul",
            "shame",
            "torment",
            "confession",
            "guilt",
            "verdict",
            "suffering",
            "inner",
            "heart",
        ]
    return [
        "memory",
        "remembrance",
        "time",
        "scent",
        "silence",
        "moment",
        "past",
        "interval",
        "sensory",
        "recalled",
    ]


def _passes_quality_filter(source, candidate, style, strict_quality=True):
    src_len = max(1, len(_tokenize(source)))
    cand_len = max(1, len(_tokenize(candidate)))
    len_ratio = cand_len / src_len

    if strict_quality:
        if len_ratio < 0.5 or len_ratio > 2.4:
            return False
        if _semantic_overlap(source, candidate) < 0.35:
            return False
        if sum(1 for marker in _style_markers(style) if marker in candidate.lower()) < 1:
            return False
    else:
        if len_ratio < 0.35 or len_ratio > 3.0:
            return False
        if _semantic_overlap(source, candidate) < 0.2:
            return False

    return True


def _generate_once(tokenizer, model, model_kind, prompt, temperature, top_p, max_new_tokens):
    encoded = tokenizer(prompt, return_tensors="pt")

    with torch.no_grad():
        if model_kind == "seq2seq":
            output_ids = model.generate(
                **encoded,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=1.12,
                no_repeat_ngram_size=3,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
            )
            decoded = tokenizer.decode(output_ids[0], skip_special_tokens=True)
            return _extract_rewrite("", decoded)

        output_ids = model.generate(
            **encoded,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=1.12,
            no_repeat_ngram_size=3,
            max_new_tokens=max_new_tokens,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
    decoded = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return _extract_rewrite(prompt, decoded)


def candidates_lm(
    text,
    style,
    n=4,
    seed=None,
    model_name="distilgpt2",
    model_kind="auto",
    max_new_tokens=96,
    strict_quality=True,
):
    tokenizer, model, resolved_kind = _load_lm(model_name, model_kind=model_kind)
    prompt = _build_prompt(text, style)

    candidates = []
    seen = set()
    rejected = []
    temp_cycle = [0.65, 0.75, 0.85, 0.95]
    top_p_cycle = [0.88, 0.9, 0.93, 0.96]

    max_attempts = max(n + 2, n * 5)
    for i in range(max_attempts):
        if len(candidates) >= n:
            break
        run_seed = seed + i if seed is not None else _stable_seed(f"{text}|{style}|{i}")
        torch.manual_seed(run_seed)

        temperature = temp_cycle[i % len(temp_cycle)]
        top_p = top_p_cycle[i % len(top_p_cycle)]

        cand = _generate_once(
            tokenizer,
            model,
            resolved_kind,
            prompt,
            temperature,
            top_p,
            max_new_tokens,
        )
        if not cand:
            continue
        if cand.lower() == text.lower():
            continue
        if cand in seen:
            continue
        if not _passes_quality_filter(text, cand, style, strict_quality=strict_quality):
            rejected.append(cand)
            continue
        seen.add(cand)
        candidates.append(cand)

    if strict_quality and len(candidates) < n and rejected:
        for cand in rejected:
            if cand in seen:
                continue
            if _passes_quality_filter(text, cand, style, strict_quality=False):
                seen.add(cand)
                candidates.append(cand)
                if len(candidates) >= n:
                    break

    if not candidates:
        candidates = [text]
    return candidates


def generate_one(
    text,
    style,
    diversify=True,
    seed=None,
    n_candidates=4,
    lm_model=None,
    profile="instruct",
    max_new_tokens=96,
    strict_quality=True,
):
    style = normalize_style(style)
    resolved_profile, cfg = _resolve_profile(profile, lm_model=lm_model)
    n_candidates = max(1, n_candidates or cfg["n_candidates"])
    max_new_tokens = max(24, max_new_tokens or cfg["max_new_tokens"])

    model_name = cfg["model_name"]
    model_kind = cfg["kind"]

    cands = candidates_lm(
        text,
        style,
        n=n_candidates,
        seed=seed,
        model_name=model_name,
        model_kind=model_kind,
        max_new_tokens=max_new_tokens,
        strict_quality=strict_quality,
    )
    ranked = rank_candidates(text, cands, style)
    best = _sample_topk(ranked, top_k=3, temperature=0.35, seed=seed) if diversify else ranked[0]
    return {
        "output": best[1],
        "score": best[0],
        "style_score": best[2],
        "semantic_score": best[3],
        "model": model_name,
        "profile": resolved_profile,
        "strict_quality": strict_quality,
        "candidates": [{"score": r[0], "text": r[1]} for r in ranked],
    }


def pastiche(
    text,
    author,
    diversify=True,
    seed=None,
    n_candidates=4,
    lm_model=None,
    profile="instruct",
    max_new_tokens=96,
    strict_quality=True,
):
    """Rewrite input text in the requested author style.

    Args:
        text: Source text to transform.
        author: Target style. Accepted values: "proust", "dosto", "dostoevsky".

    Returns:
        The best style-transferred output string.
    """
    if not isinstance(text, str) or not text.strip():
        raise ValueError("text must be a non-empty string")
    result = generate_one(
        text.strip(),
        author,
        diversify=diversify,
        seed=seed,
        n_candidates=n_candidates,
        lm_model=lm_model,
        profile=profile,
        max_new_tokens=max_new_tokens,
        strict_quality=strict_quality,
    )
    return result["output"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="checkpoints/base")
    parser.add_argument("--adapter_path", required=True)
    parser.add_argument("--style", required=True, choices=["dosto", "dostoevsky", "proust"])
    parser.add_argument(
        "--profile",
        default="instruct",
        choices=["fast", "instruct"],
        help="Generation profile. 'instruct' uses an instruction-tuned model when available.",
    )
    parser.add_argument(
        "--lm_model",
        default=None,
        help="Optional Hugging Face model id override for text generation",
    )
    parser.add_argument("--n_candidates", type=int, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--deterministic", action="store_true", help="Always pick top-ranked candidate")
    parser.add_argument(
        "--strict_quality",
        dest="strict_quality",
        action="store_true",
        help="Reject weak generations and resample automatically",
    )
    parser.add_argument(
        "--allow_low_quality",
        dest="strict_quality",
        action="store_false",
        help="Disable strict filtering",
    )
    parser.set_defaults(strict_quality=True)
    parser.add_argument("--text", help="Single input text")
    parser.add_argument("--input_file", help="Optional JSONL file with {'id','text'} rows")
    parser.add_argument("--output_file", default="outputs/dev_predictions.jsonl")
    args = parser.parse_args()

    _ = Path(args.model_path)
    args.style = normalize_style(args.style)
    adapter_file = Path(args.adapter_path) / "adapter_artifact.json"
    if not adapter_file.exists():
        raise FileNotFoundError(f"Adapter artifact not found at {adapter_file}")

    rows = []
    if args.text:
        rows.append({"id": "single", "text": args.text})
    elif args.input_file:
        with open(args.input_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    else:
        raise ValueError("Provide either --text or --input_file")

    out = []
    for row in rows:
        gen = generate_one(
            row["text"],
            args.style,
            diversify=not args.deterministic,
            seed=args.seed,
            n_candidates=args.n_candidates,
            lm_model=args.lm_model,
            profile=args.profile,
            max_new_tokens=args.max_new_tokens,
            strict_quality=args.strict_quality,
        )
        out.append(
            {
                "id": row["id"],
                "style": args.style,
                "source": row["text"],
                "prediction": gen["output"],
                "score": gen["score"],
                "style_score": gen["style_score"],
                "semantic_score": gen["semantic_score"],
                "model": gen["model"],
                "profile": gen["profile"],
                "strict_quality": gen["strict_quality"],
            }
        )

    out_path = Path(args.output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for row in out:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Wrote {len(out)} generations to {out_path}")
    for row in out[:3]:
        print("---")
        print(f"id={row['id']} style={row['style']}")
        print(f"source: {row['source']}")
        print(f"prediction: {row['prediction']}")


if __name__ == "__main__":
    main()
