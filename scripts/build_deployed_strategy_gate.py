from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from poker_agent.deployed_strategy_gate import write_deployed_strategy_gate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the deployed strategy-stack approval gate")
    parser.add_argument("--project-root", default=ROOT, type=Path)
    parser.add_argument("--out", default=ROOT / "reports" / "deployed_strategy_gate.json", type=Path)
    parser.add_argument("--markdown-out", default=ROOT / "reports" / "deployed_strategy_gate.md", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = write_deployed_strategy_gate(args.project_root, args.out, args.markdown_out)
    print(
        json.dumps(
            {
                "status": payload["status"],
                "strategy_policy_status": payload["strategy_policy_status"],
                "raw_supervised_model_status": payload["raw_supervised_model_status"],
                "blocking_items": len(payload["blocking_items"]),
                "component_risks": len(payload["component_risks"]),
                "out": str(args.out),
            },
            sort_keys=True,
        )
    )
    if payload["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
