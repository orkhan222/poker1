from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from poker_agent.strategy_remediation import write_strategy_remediation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the strategy remediation report")
    parser.add_argument("--project-root", default=ROOT, type=Path)
    parser.add_argument("--out", default=ROOT / "reports" / "strategy_remediation.json", type=Path)
    parser.add_argument("--markdown-out", default=ROOT / "reports" / "strategy_remediation.md", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = write_strategy_remediation(args.project_root, args.out, args.markdown_out)
    print(
        json.dumps(
            {
                "strategy_policy_status": payload["strategy_policy_status"],
                "release_mode": payload["release_mode"],
                "blocking_items": len(payload["blocking_items"]),
                "out": str(args.out),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
