import argparse
import json
from pathlib import Path


def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def style_transform(text, style):
    if style == "dostoevsky":
        return (
            "It seemed to me, in that uneasy hour, that "
            + text[0].lower()
            + text[1:]
            + " and this simple fact, once spoken inwardly, turned into a private accusation against my own conscience."
        )
    return (
        "As though recalled by a fragrance hidden in time, "
        + text[0].lower()
        + text[1:]
        + "; and before I could resist, memory had already arranged the scene with its old, patient precision."
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed_dir", required=True)
    parser.add_argument("--out_file", required=True)
    args = parser.parse_args()

    processed_dir = Path(args.processed_dir)
    modern_rows = load_jsonl(processed_dir / "modern_clean.jsonl")

    out_path = Path(args.out_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for row in modern_rows:
        for style in ["dostoevsky", "proust"]:
            rows.append(
                {
                    "id": f"{row['id']}_{style}",
                    "source": row["text"],
                    "target": style_transform(row["text"], style),
                    "style": style,
                }
            )

    with open(out_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Saved {len(rows)} training pairs to {out_path}")


if __name__ == "__main__":
    main()
