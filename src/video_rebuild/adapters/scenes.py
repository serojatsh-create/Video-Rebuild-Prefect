from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .base import JsonSubprocessAdapter


@dataclass(frozen=True)
class SceneDetectionAdapter:
    python_path: Path
    worker_path: Path

    def _runner(self) -> JsonSubprocessAdapter:
        return JsonSubprocessAdapter(
            name="scene-detection",
            python_path=self.python_path,
            worker_path=self.worker_path,
        )

    def probe(self) -> dict[str, Any]:
        return self._runner().run({"action": "probe"})

    def detect(
        self,
        video_path: Path,
        *,
        threshold: float = 27.0,
        minimum_scene_length_frames: int = 15,
    ) -> dict[str, Any]:
        return self._runner().run(
            {
                "action": "detect",
                "video_path": str(video_path),
                "threshold": threshold,
                "minimum_scene_length_frames": minimum_scene_length_frames,
            }
        )
