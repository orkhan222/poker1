from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dataset import build_qlora_splits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Phase 3 supervised event-normalization splits")
    parser.add_argument("--source", default=Path("evaluation/event_extraction_phase1.jsonl"), type=Path)
    parser.add_argument("--output-dir", default=Path("data"), type=Path)
    parser.add_argument("--manifest-out", default=Path("reports/qlora_dataset_manifest.json"), type=Path)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--no-shuffle-train", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = build_qlora_splits(
        source=args.source,
        output_dir=args.output_dir,
        seed=args.seed,
        shuffle_train=not args.no_shuffle_train,
    )
    args.manifest_out.parent.mkdir(parents=True, exist_ok=True)
    args.manifest_out.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(f"phase3_train_rows={manifest['counts'].get('train', 0)}")
    print(f"phase3_val_rows={manifest['counts'].get('val', 0)}")
    print(f"phase3_test_rows={manifest['counts'].get('test', 0)}")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    if manifest["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()

