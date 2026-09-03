from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from video_rebuild.config import load_config
from video_rebuild.runtime_pipeline import run_candidate_analysis


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run ffprobe, scene detection and person exclusion on one video."
    )
    parser.add_argument("video", type=Path)
    parser.add_argument("config", type=Path)
    parser.add_argument("--minimum-duration", type=float, default=0.5)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = run_candidate_analysis(
        args.video,
        load_config(args.config),
        minimum_duration_s=args.minimum_duration,
    )
    json.dump(result.model_dump(mode="json"), sys.stdout, ensure_ascii=False, indent=2)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
