import argparse
import json
from pathlib import Path

from losses import total_loss


def parse_simple_yaml(path):
    cfg = {}
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if ":" not in line:
            continue
        k, v = line.split(":", 1)
        k = k.strip()
        v = v.strip()
        if v.lower() in {"true", "false"}:
            cfg[k] = v.lower() == "true"
            continue
        try:
            if "." in v:
                cfg[k] = float(v)
            else:
                cfg[k] = int(v)
            continue
        except ValueError:
            pass
        cfg[k] = v.strip('"').strip("'")
    return cfg


def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = parse_simple_yaml(args.config)
    style = cfg.get("style")
    train_file = Path(cfg.get("train_file", "data/processed/train.jsonl"))
    output_dir = Path(cfg.get("output_dir", f"checkpoints/{style}_lora"))
    rank = int(cfg.get("lora_rank", 16))
    alpha = float(cfg.get("lora_alpha", 32.0))

    rows = [r for r in load_jsonl(train_file) if r.get("style") == style]
    if not rows:
        raise ValueError(f"No rows found for style '{style}' in {train_file}")

    stats = {"nll": 0.0, "sem": 0.0, "style": 0.0, "total": 0.0}
    for row in rows:
        losses = total_loss(row["source"], row["target"], style)
        for k in stats:
            stats[k] += losses[k]

    for k in stats:
        stats[k] /= len(rows)

    output_dir.mkdir(parents=True, exist_ok=True)
    artifact = {
        "adapter_type": "lora",
        "style": style,
        "lora_rank": rank,
        "lora_alpha": alpha,
        "train_rows": len(rows),
        "avg_losses": stats,
    }

    artifact_path = output_dir / "adapter_artifact.json"
    artifact_path.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    print(f"Saved adapter artifact to {artifact_path}")
    print(json.dumps(artifact, indent=2))


if __name__ == "__main__":
    main()
