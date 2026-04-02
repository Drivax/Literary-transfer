import argparse
import hashlib
import json
from pathlib import Path
import random
import sys

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:
    torch = None
    AutoModelForCausalLM = None
    AutoTokenizer = None

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

try:
    from .rerank import rank_candidates
except ImportError:
    from rerank import rank_candidates


_MODEL_CACHE = {}


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


def _load_lm(model_name):
    if AutoModelForCausalLM is None or AutoTokenizer is None or torch is None:
        raise ImportError(
            "Missing dependencies for LM generation. Install 'transformers' and 'torch'."
        )
    if model_name in _MODEL_CACHE:
        return _MODEL_CACHE[model_name]

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.eval()
    _MODEL_CACHE[model_name] = (tokenizer, model)
    return tokenizer, model


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


def candidates_lm(text, style, n=4, seed=None, model_name="distilgpt2", max_new_tokens=96):
    tokenizer, model = _load_lm(model_name)
    prompt = _build_prompt(text, style)
    input_ids = tokenizer(prompt, return_tensors="pt")

    candidates = []
    seen = set()
    temp_cycle = [0.65, 0.75, 0.85, 0.95]
    top_p_cycle = [0.88, 0.9, 0.93, 0.96]

    for i in range(n + 2):
        if len(candidates) >= n:
            break
        run_seed = seed + i if seed is not None else _stable_seed(f"{text}|{style}|{i}")
        torch.manual_seed(run_seed)

        temperature = temp_cycle[i % len(temp_cycle)]
        top_p = top_p_cycle[i % len(top_p_cycle)]

        with torch.no_grad():
            output_ids = model.generate(
                **input_ids,
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
        cand = _extract_rewrite(prompt, decoded)
        if not cand:
            continue
        if cand.lower() == text.lower():
            continue
        if cand in seen:
            continue
        seen.add(cand)
        candidates.append(cand)

    if not candidates:
        candidates = [text]
    return candidates


def generate_one(
    text,
    style,
    diversify=True,
    seed=None,
    n_candidates=4,
    lm_model="distilgpt2",
    max_new_tokens=96,
):
    style = normalize_style(style)
    cands = candidates_lm(
        text,
        style,
        n=n_candidates,
        seed=seed,
        model_name=lm_model,
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
    lm_model="distilgpt2",
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
        default="distilgpt2",
        help="Hugging Face model id for text generation",
    )
    parser.add_argument("--n_candidates", type=int, default=4)
    parser.add_argument("--max_new_tokens", type=int, default=96)
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
