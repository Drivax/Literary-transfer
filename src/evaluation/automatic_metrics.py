import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from utils.text_stats import avg_sentence_length, bow_cosine, distinct_ngrams, pseudo_perplexity, tokenize


def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def compute_metrics(rows):
    if not rows:
        return {}
    sem_scores = [bow_cosine(r["source"], r["prediction"]) for r in rows]
    ppl_scores = [pseudo_perplexity(r["prediction"], r["style"]) for r in rows]
    sent_shift = [avg_sentence_length(r["prediction"]) - avg_sentence_length(r["source"]) for r in rows]
    distinct2 = [distinct_ngrams(tokenize(r["prediction"]), n=2) for r in rows]

    def avg(xs):
        return sum(xs) / len(xs) if xs else 0.0

    return {
        "n_samples": len(rows),
        "semantic_similarity_cosine": round(avg(sem_scores), 4),
        "stylometric_perplexity_proxy": round(avg(ppl_scores), 4),
        "avg_sentence_length_shift": round(avg(sent_shift), 4),
        "distinct_2": round(avg(distinct2), 4),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_file", required=True)
    parser.add_argument("--out_file", default="outputs/metrics_summary.json")
    args = parser.parse_args()

    rows = load_jsonl(Path(args.pred_file))
    by_style = {}
    for row in rows:
        by_style.setdefault(row["style"], []).append(row)

    report = {"overall": compute_metrics(rows), "by_style": {}}
    for style, style_rows in by_style.items():
        report["by_style"][style] = compute_metrics(style_rows)

    out_path = Path(args.out_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps(report, indent=2))
    print(f"Saved metrics report to {out_path}")


if __name__ == "__main__":
    main()
