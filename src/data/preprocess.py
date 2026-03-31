import argparse
import json
import re
from pathlib import Path


def clean_text(text):
    text = text.replace("\t", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def load_lines_from_txt(path):
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = clean_text(line)
        if line:
            rows.append(line)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    style_records = []
    for style in ["dostoevsky", "proust"]:
        style_file = input_dir / style / "corpus.txt"
        if not style_file.exists():
            raise FileNotFoundError(f"Missing corpus file: {style_file}")
        for i, line in enumerate(load_lines_from_txt(style_file), start=1):
            style_records.append({"id": f"{style}_{i}", "style": style, "text": line})

    modern_file = input_dir / "modern" / "modern_inputs.jsonl"
    modern_rows = []
    with open(modern_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                row = json.loads(line)
                row["text"] = clean_text(row["text"])
                modern_rows.append(row)

    style_out = output_dir / "style_corpus.jsonl"
    with open(style_out, "w", encoding="utf-8") as f:
        for row in style_records:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    modern_out = output_dir / "modern_clean.jsonl"
    with open(modern_out, "w", encoding="utf-8") as f:
        for row in modern_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Saved {len(style_records)} style rows to {style_out}")
    print(f"Saved {len(modern_rows)} modern rows to {modern_out}")


if __name__ == "__main__":
    main()
