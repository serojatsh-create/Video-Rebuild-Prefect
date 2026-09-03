from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .base import JsonSubprocessAdapter


@dataclass(frozen=True)
class MotionAnalysisAdapter:
    python_path: Path
    worker_path: Path

    def analyze(
        self,
        video_path: Path,
        start_frame: int,
        end_frame: int,
        *,
        resize_width: int = 320,
    ) -> dict[str, Any]:
        return JsonSubprocessAdapter(
            name="motion-analysis",
            python_path=self.python_path,
            worker_path=self.worker_path,
        ).run(
            {
                "action": "analyze",
                "video_path": str(video_path),
                "start_frame": start_frame,
                "end_frame": end_frame,
                "resize_width": resize_width,
            }
        )
