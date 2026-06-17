from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from poker_agent.event_normalization.benchmark import main


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Phase 2 event-normalization baselines")
    parser.add_argument("--dataset", default=Path("evaluation/event_extraction_phase1.jsonl"), type=Path)
    parser.add_argument("--split", default="test", choices=("train", "valid", "test", "all"))
    parser.add_argument("--backend", default="heuristic", choices=("heuristic", "transformers"))
    parser.add_argument("--model-ids", default="heuristic_text")
    parser.add_argument("--few-shot-sizes", default="5,10")
    parser.add_argument("--out-csv", default=Path("reports/phase2_event_benchmark_results.csv"), type=Path)
    parser.add_argument("--out-json", default=Path("reports/phase2_event_benchmark_results.json"), type=Path)
    parser.add_argument("--predictions-out", default=Path("reports/phase2_event_benchmark_predictions.jsonl"), type=Path)
    parser.add_argument("--report-out", default=Path("reports/phase2_event_benchmark_report.md"), type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-new-tokens", default=128, type=int)
    parser.add_argument("--temperature", default=0.0, type=float)
    parser.add_argument("--max-examples", default=0, type=int)
    parser.add_argument("--seed", default=42, type=int)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
