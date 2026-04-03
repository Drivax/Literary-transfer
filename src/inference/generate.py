import argparse
import hashlib
import json
from pathlib import Path
import random
import sys

try:
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer
except ImportError:
    torch = None
    AutoConfig = None
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

try:
    from ..utils.text_stats import distinct_ngrams, tokenize
except ImportError:
    from utils.text_stats import distinct_ngrams, tokenize


_MODEL_CACHE = {}


MODEL_PROFILES = {
    "legacy": {
        "model_name": "distilgpt2",
        "temperatures": [0.65, 0.75, 0.85, 0.95],
        "top_ps": [0.88, 0.9, 0.93, 0.96],
        "repetition_penalty": 1.12,
        "no_repeat_ngram_size": 3,
    },
    "instruct": {
        "model_name": "google/flan-t5-small",
        "temperatures": [0.55, 0.62, 0.7, 0.78],
        "top_ps": [0.82, 0.86, 0.9, 0.92],
        "repetition_penalty": 1.18,
        "no_repeat_ngram_size": 4,
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


def _resolve_profile(profile_name, model_name):
    if profile_name not in MODEL_PROFILES:
        raise ValueError(f"lm_profile must be one of: {', '.join(sorted(MODEL_PROFILES))}")
    profile = dict(MODEL_PROFILES[profile_name])
    if model_name:
        profile["model_name"] = model_name
    return profile


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


def _load_lm(model_name):
    if AutoConfig is None or AutoModelForCausalLM is None or AutoModelForSeq2SeqLM is None or AutoTokenizer is None or torch is None:
        raise ImportError(
            "Missing dependencies for LM generation. Install 'transformers' and 'torch'."
        )
    if model_name in _MODEL_CACHE:
        return _MODEL_CACHE[model_name]

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    config = AutoConfig.from_pretrained(model_name)
    if getattr(config, "is_encoder_decoder", False):
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        is_seq2seq = True
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name)
        is_seq2seq = False

    if tokenizer.pad_token_id is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    _MODEL_CACHE[model_name] = (tokenizer, model, is_seq2seq)
    return tokenizer, model, is_seq2seq


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


def _passes_quality_filter(source, candidate, style, min_semantic, min_style, min_length_ratio):
    source_tokens = tokenize(source)
    cand_tokens = tokenize(candidate)
    if len(cand_tokens) < 8:
        return False

    semantic = rank_candidates(source, [candidate], style)[0][3]
    style_score = rank_candidates(source, [candidate], style)[0][2]
    length_ratio = len(cand_tokens) / max(1, len(source_tokens))
    distinct2 = distinct_ngrams(cand_tokens, n=2)

    if semantic < min_semantic:
        return False
    if style_score < min_style:
        return False
    if length_ratio < min_length_ratio:
        return False
    if distinct2 < 0.55:
        return False
    return True


def candidates_lm(
    text,
    style,
    n=4,
    seed=None,
    model_name=None,
    max_new_tokens=96,
    lm_profile="legacy",
    strict_quality=False,
    min_semantic=0.45,
    min_style=0.10,
    min_length_ratio=0.5,
):
    profile = _resolve_profile(lm_profile, model_name)
    tokenizer, model, is_seq2seq = _load_lm(profile["model_name"])
    prompt = _build_prompt(text, style)
    model_inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
    model_inputs = {k: v.to(model.device) for k, v in model_inputs.items()}

    candidates = []
    rejected = []
    seen = set()
    temp_cycle = profile["temperatures"]
    top_p_cycle = profile["top_ps"]
    attempts = n + (12 if strict_quality else 4)

    for i in range(attempts):
        if len(candidates) >= n:
            break
        run_seed = seed + i if seed is not None else _stable_seed(f"{text}|{style}|{i}")
        generator = torch.Generator(device=model.device)
        generator.manual_seed(run_seed)

        temperature = temp_cycle[i % len(temp_cycle)]
        top_p = top_p_cycle[i % len(top_p_cycle)]

        with torch.no_grad():
            output_ids = model.generate(
                **model_inputs,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=profile["repetition_penalty"],
                no_repeat_ngram_size=profile["no_repeat_ngram_size"],
                max_new_tokens=max_new_tokens,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                generator=generator,
            )

        decoded = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        cand = decoded.strip() if is_seq2seq else _extract_rewrite(prompt, decoded)
        if not cand:
            continue
        if cand.lower() == text.lower():
            continue
        if cand in seen:
            continue
        if strict_quality and not _passes_quality_filter(
            text,
            cand,
            style,
            min_semantic=min_semantic,
            min_style=min_style,
            min_length_ratio=min_length_ratio,
        ):
            rejected.append(cand)
            continue
        seen.add(cand)
        candidates.append(cand)

    if not candidates:
        if rejected:
            candidates = rejected[:1]
        else:
            raise RuntimeError("No valid model generations were produced")
    return candidates


def generate_one(
    text,
    style,
    diversify=True,
    seed=None,
    n_candidates=4,
    lm_model=None,
    lm_profile="legacy",
    strict_quality=False,
    min_semantic=0.45,
    min_style=0.10,
    min_length_ratio=0.5,
    max_new_tokens=96,
):
    style = normalize_style(style)
    cands = candidates_lm(
        text,
        style,
        n=n_candidates,
        seed=seed,
        model_name=lm_model,
        lm_profile=lm_profile,
        strict_quality=strict_quality,
        min_semantic=min_semantic,
        min_style=min_style,
        min_length_ratio=min_length_ratio,
        max_new_tokens=max_new_tokens,
    )
    ranked = rank_candidates(text, cands, style)
    best = _sample_topk(ranked, top_k=3, temperature=0.35, seed=seed) if diversify else ranked[0]
    return {
        "output": best[1],
        "score": best[0],
        "style_score": best[2],
        "semantic_score": best[3],
        "candidates": [{"score": r[0], "text": r[1]} for r in ranked],
    }


def pastiche(
    text,
    author,
    diversify=True,
    seed=None,
    n_candidates=4,
    lm_model=None,
    lm_profile="legacy",
    strict_quality=False,
    min_semantic=0.45,
    min_style=0.10,
    min_length_ratio=0.5,
    max_new_tokens=96,
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
        lm_profile=lm_profile,
        strict_quality=strict_quality,
        min_semantic=min_semantic,
        min_style=min_style,
        min_length_ratio=min_length_ratio,
        max_new_tokens=max_new_tokens,
    )
    return result["output"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="checkpoints/base")
    parser.add_argument("--adapter_path", required=True)
    parser.add_argument("--style", required=True, choices=["dosto", "dostoevsky", "proust"])
    parser.add_argument(
        "--lm_model",
        default=None,
        help="Hugging Face model id for text generation",
    )
    parser.add_argument(
        "--lm_profile",
        default="legacy",
        choices=sorted(MODEL_PROFILES.keys()),
        help="Generation preset controlling default model and decoding strategy",
    )
    parser.add_argument("--n_candidates", type=int, default=4)
    parser.add_argument("--max_new_tokens", type=int, default=96)
    parser.add_argument("--strict_quality", action="store_true")
    parser.add_argument("--min_semantic", type=float, default=0.45)
    parser.add_argument("--min_style", type=float, default=0.10)
    parser.add_argument("--min_length_ratio", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--deterministic", action="store_true", help="Always pick top-ranked candidate")
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
            lm_profile=args.lm_profile,
            strict_quality=args.strict_quality,
            min_semantic=args.min_semantic,
            min_style=args.min_style,
            min_length_ratio=args.min_length_ratio,
            max_new_tokens=args.max_new_tokens,
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
