from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .visual_analysis import CodexVisualAnalyzer, VisualAnalysisError


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze a prepared video evidence packet with Codex."
    )
    parser.add_argument("packet", type=Path, help="Path to analysis_packet.json")
    parser.add_argument("output", type=Path, help="Output SegmentAnalysis JSON path")
    parser.add_argument("--human-review", action="store_true")
    parser.add_argument("--codex", default="codex", help="Codex CLI executable")
    parser.add_argument("--schema", type=Path)
    parser.add_argument("--timeout-seconds", type=float, default=600.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        result = CodexVisualAnalyzer(
            codex_executable=args.codex,
            schema_path=args.schema,
            timeout_seconds=args.timeout_seconds,
        ).analyze(
            packet_path=args.packet,
            output_path=args.output,
            require_human_review=args.human_review,
        )
    except VisualAnalysisError as error:
        print(str(error), file=sys.stderr)
        return 1
    json.dump(result.model_dump(mode="json"), sys.stdout, ensure_ascii=False, indent=2)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
