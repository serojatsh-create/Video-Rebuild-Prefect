from __future__ import annotations

import argparse
import json

from video_rebuild.flows import run_contract_only_pipeline
from video_rebuild.models import TimeSpan


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the no-media contract-only pipeline.")
    parser.add_argument("--segment-id", default="seg-001")
    parser.add_argument("--start", type=float, default=0.0)
    parser.add_argument("--end", type=float, default=2.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = run_contract_only_pipeline(
        segment_id=args.segment_id,
        span=TimeSpan(start_s=args.start, end_s=args.end),
    )
    print(json.dumps(result.model_dump(mode="json"), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

