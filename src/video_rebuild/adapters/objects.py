from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .base import JsonSubprocessAdapter


@dataclass(frozen=True)
class YOLOObjectAdapter:
    python_path: Path
    worker_path: Path
    model_path: Path
    device: str = "auto"

    def detect(
        self,
        frame_paths: Iterable[Path],
        *,
        confidence: float = 0.25,
        image_size: int = 640,
    ) -> dict[str, Any]:
        return JsonSubprocessAdapter(
            name="object-detection",
            python_path=self.python_path,
            worker_path=self.worker_path,
        ).run(
            {
                "action": "detect",
                "frame_paths": [str(path) for path in frame_paths],
                "model_path": str(self.model_path),
                "device": self.device,
                "confidence": confidence,
                "image_size": image_size,
            }
        )
