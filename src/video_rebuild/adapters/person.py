from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..models import TimeSpan
from ..person_frames import frames_to_span_dicts
from .base import JsonSubprocessAdapter


@dataclass(frozen=True)
class UltralyticsPersonAdapter:
    python_path: Path
    worker_path: Path
    model_path: Path
    device: str = "auto"

    def _runner(self) -> JsonSubprocessAdapter:
        return JsonSubprocessAdapter(
            name="person",
            python_path=self.python_path,
            worker_path=self.worker_path,
        )

    def probe(self) -> dict[str, Any]:
        return self._runner().run(
            {
                "action": "probe",
                "model_path": str(self.model_path),
                "device": self.device,
            }
        )

    def detect(
        self,
        video_path: Path,
        *,
        confidence: float = 0.35,
        image_size: int = 640,
        gap_tolerance_frames: int = 2,
    ) -> dict[str, Any]:
        return self._runner().run(
            {
                "action": "detect",
                "video_path": str(video_path),
                "model_path": str(self.model_path),
                "confidence": confidence,
                "image_size": image_size,
                "gap_tolerance_frames": gap_tolerance_frames,
                "device": self.device,
            }
        )


def frames_to_spans(
    presence: list[bool],
    fps: float,
    gap_tolerance_frames: int = 0,
) -> list[TimeSpan]:
    payloads = frames_to_span_dicts(
        presence,
        fps=fps,
        gap_tolerance_frames=gap_tolerance_frames,
    )
    return [TimeSpan.model_validate(payload) for payload in payloads]
