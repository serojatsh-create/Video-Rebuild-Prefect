from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from video_rebuild.flows import analysis_packet_flow


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare representative frames and OCR packets.")
    parser.add_argument("video", type=Path)
    parser.add_argument("config", type=Path)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--minimum-duration", type=float, default=0.5)
    parser.add_argument("--coarse-fps", type=float, default=2.0)
    parser.add_argument("--maximum-coarse-frames", type=int, default=30)
    parser.add_argument("--context-seconds", type=float, default=0.5)
    parser.add_argument("--maximum-fine-frames", type=int, default=30)
    parser.add_argument("--parallel-workers", type=int, default=2)
    args = parser.parse_args()
    result = analysis_packet_flow(
        str(args.video.resolve()), str(args.config.resolve()),
        str(args.output_root.resolve()),
        args.minimum_duration,
        args.coarse_fps,
        args.maximum_coarse_frames,
        args.context_seconds,
        args.maximum_fine_frames,
        args.parallel_workers,
    )
    json.dump(result, sys.stdout, ensure_ascii=False, indent=2)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
