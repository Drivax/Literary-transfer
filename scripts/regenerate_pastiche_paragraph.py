from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.inference.generate import pastiche


def main():
    text = Path("data/raw/modern/paragraph_input.txt").read_text(encoding="utf-8").strip()
    seeds = [11, 22, 33]
    styles = [("dostoevsky", "DOSTO"), ("proust", "PROUST")]

    lines = []
    for style_key, label in styles:
        for idx, seed in enumerate(seeds, start=1):
            out = pastiche(
                text,
                style_key,
                diversify=True,
                seed=seed,
                n_candidates=3,
                lm_profile="instruct",
                strict_quality=True,
                min_semantic=0.45,
                min_style=0.10,
                min_length_ratio=0.45,
                max_new_tokens=90,
            )
            lines.append(f"{label} #{idx}: {out}")

    out_path = Path("outputs/pastiche_paragraph_outputs.txt")
    out_path.write_text("\n\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {out_path} with {len(lines)} model generations.")


if __name__ == "__main__":
    main()
