from __future__ import annotations

import argparse
import json
from pathlib import Path

from video_rebuild.models import (
    AnalysisPacket,
    ExecutionPlan,
    FullFlowResult,
    QCReport,
    SegmentAnalysis,
    VisualEvidenceSequence,
    VisualReconstructionSpec,
    VideoMeta,
    VideoCandidateResult,
)
from video_rebuild.visual_analysis import VisualAnalysisRunResult


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export Pydantic JSON schemas.")
    parser.add_argument("output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    models = [
        AnalysisPacket,
        VideoMeta,
        VideoCandidateResult,
        SegmentAnalysis,
        VisualEvidenceSequence,
        VisualReconstructionSpec,
        ExecutionPlan,
        QCReport,
        FullFlowResult,
        VisualAnalysisRunResult,
    ]
    for model in models:
        target = args.output / f"{model.__name__}.schema.json"
        target.write_text(
            json.dumps(model.model_json_schema(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
