from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, field_validator


class ToolPaths(BaseModel):
    ffmpeg: Path
    ffprobe: Path
    effect_analysis_python: Path
    effect_analysis_project: Path
    scene_python: Path
    scene_worker: Path
    person_python: Path
    person_worker: Path
    person_model: Path
    ocr_python: Path
    ocr_worker: Path
    ocr_detection_model: Path
    ocr_recognition_model: Path

    @field_validator("*", mode="after")
    @classmethod
    def require_absolute_path(cls, value: Path) -> Path:
        if not value.is_absolute():
            raise ValueError(f"tool paths must be absolute: {value}")
        return value


def load_config(path: Path) -> ToolPaths:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return ToolPaths.model_validate(payload)
