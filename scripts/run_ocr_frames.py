from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from video_rebuild.adapters.ocr import PaddleOCRAdapter
from video_rebuild.config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run local PP-OCRv6 on one or more existing image files."
    )
    parser.add_argument("config", type=Path)
    parser.add_argument("frames", type=Path, nargs="+")
    parser.add_argument("--minimum-score", type=float, default=0.5)
    args = parser.parse_args()
    for frame in args.frames:
        if not frame.is_file():
            parser.error(f"frame does not exist: {frame}")
    return args


def main() -> int:
    args = parse_args()
    tools = load_config(args.config)
    adapter = PaddleOCRAdapter(
        python_path=tools.ocr_python,
        worker_path=tools.ocr_worker,
        detection_model_dir=tools.ocr_detection_model,
        recognition_model_dir=tools.ocr_recognition_model,
    )
    result = adapter.recognize(
        args.frames,
        minimum_score=args.minimum_score,
    )
    json.dump(result, sys.stdout, ensure_ascii=False, indent=2)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
