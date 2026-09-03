from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from video_rebuild.config import load_config
from video_rebuild.tooling import ToolCapability, ToolSpec, probe_tool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Report local Video Rebuild capabilities.")
    parser.add_argument("config", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    specs = [
        ToolSpec(name="ffmpeg", path=config.ffmpeg),
        ToolSpec(name="ffprobe", path=config.ffprobe),
        ToolSpec(name="effect_analysis_python", path=config.effect_analysis_python),
        ToolSpec(name="effect_analysis_project", path=config.effect_analysis_project),
        ToolSpec(name="scene_python", path=config.scene_python),
        ToolSpec(name="scene_worker", path=config.scene_worker),
        ToolSpec(name="person_python", path=config.person_python),
        ToolSpec(name="person_worker", path=config.person_worker),
        ToolSpec(name="person_model", path=config.person_model),
        ToolSpec(name="ocr_python", path=config.ocr_python),
        ToolSpec(name="ocr_worker", path=config.ocr_worker),
        ToolSpec(name="ocr_detection_model", path=config.ocr_detection_model),
        ToolSpec(name="ocr_recognition_model", path=config.ocr_recognition_model),
    ]
    capabilities: list[ToolCapability] = []
    for spec in specs:
        if spec.name in {
            "effect_analysis_project",
            "ocr_detection_model",
            "ocr_recognition_model",
        }:
            capabilities.append(
                ToolCapability(
                    name=spec.name,
                    path=spec.path,
                    available=spec.path.is_dir(),
                    reason=None if spec.path.is_dir() else "path_not_found",
                )
            )
        else:
            capabilities.append(probe_tool(spec))
    json.dump(
        {"capabilities": [item.model_dump(mode="json") for item in capabilities]},
        sys.stdout,
        ensure_ascii=False,
        indent=2,
    )
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
