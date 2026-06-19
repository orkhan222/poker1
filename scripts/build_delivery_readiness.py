from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from poker_agent.delivery_readiness import write_delivery_readiness


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the service and strategy readiness summary")
    parser.add_argument("--project-root", default=ROOT, type=Path)
    parser.add_argument("--out", default=ROOT / "reports" / "delivery_readiness.json", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = write_delivery_readiness(args.project_root, args.out)
    print(
        json.dumps(
            {
                "overall_status": payload["overall_status"],
                "service_delivery_status": payload["service_delivery_status"],
                "strategy_policy_status": payload["strategy_policy_status"],
                "out": str(args.out),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
