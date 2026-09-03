from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .base import JsonSubprocessAdapter


@dataclass(frozen=True)
class PaddleOCRAdapter:
    python_path: Path
    worker_path: Path
    detection_model_dir: Path
    recognition_model_dir: Path
    detection_model_name: str = "PP-OCRv6_tiny_det"
    recognition_model_name: str = "PP-OCRv6_tiny_rec"
    device: str = "auto"

    def _runner(self) -> JsonSubprocessAdapter:
        return JsonSubprocessAdapter(
            name="ocr",
            python_path=self.python_path,
            worker_path=self.worker_path,
        )

    def _model_payload(self) -> dict[str, str]:
        return {
            "detection_model_name": self.detection_model_name,
            "detection_model_dir": str(self.detection_model_dir),
            "recognition_model_name": self.recognition_model_name,
            "recognition_model_dir": str(self.recognition_model_dir),
            "device": self.device,
        }

    def probe(self) -> dict[str, Any]:
        return self._runner().run(
            {
                "action": "probe",
                **self._model_payload(),
            }
        )

    def recognize(
        self,
        frame_paths: Iterable[Path],
        *,
        language: str = "ch",
        minimum_score: float = 0.5,
    ) -> dict[str, Any]:
        return self._runner().run(
            {
                "action": "recognize",
                **self._model_payload(),
                "frame_paths": [str(path) for path in frame_paths],
                "language": language,
                "minimum_score": minimum_score,
            }
        )
