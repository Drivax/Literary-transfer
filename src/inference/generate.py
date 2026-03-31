import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from rerank import rank_candidates


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


def candidates_dostoevsky(text):
    return [
        (
            "Three times, it seemed, the hour appointed by duty passed me by while I stood pretending composure; "
            + text
            + " Yet beneath that calm there worked a private tribunal of conscience, and its verdict was shame."
        ),
        (
            "I spoke as though all were orderly, and even smiled when questioned; "
            + text
            + " but inwardly I felt the soul contract under a burden it had itself invited."
        ),
        (
            "What is worse than failure is the lie with which one softens it for others: "
            + text
            + " and this small deception, repeated, grows into suffering."
        ),
    ]


def candidates_proust(text):
    return [
        (
            "As evening gathered and the ordinary motion of the day withdrew a little from me, "
            + text
            + " and in that same instant memory, with its patient craftsmanship, reopened a long-sealed chamber of time."
        ),
        (
            "Before I had named what I felt, "
            + text
            + "; and the faint sensation, like a fragrance recovered from the lining of an old coat, arranged the past in silence."
        ),
        (
            "There are moments when the present, scarcely touched, yields to remembrance: "
            + text
            + ", and then one travels without moving, carried by time itself."
        ),
    ]


def generate_one(text, style):
    style = normalize_style(style)
    cands = candidates_dostoevsky(text) if style == "dostoevsky" else candidates_proust(text)
    ranked = rank_candidates(text, cands, style)
    best = ranked[0]
    return {
        "output": best[1],
        "score": best[0],
        "style_score": best[2],
        "semantic_score": best[3],
        "candidates": [{"score": r[0], "text": r[1]} for r in ranked],
    }


def pastiche(text, author):
    """Rewrite input text in the requested author style.

    Args:
        text: Source text to transform.
        author: Target style. Accepted values: "proust", "dosto", "dostoevsky".

    Returns:
        The best style-transferred output string.
    """
    if not isinstance(text, str) or not text.strip():
        raise ValueError("text must be a non-empty string")
    result = generate_one(text.strip(), author)
    return result["output"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="checkpoints/base")
    parser.add_argument("--adapter_path", required=True)
    parser.add_argument("--style", required=True, choices=["dosto", "dostoevsky", "proust"])
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
        gen = generate_one(row["text"], args.style)
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
