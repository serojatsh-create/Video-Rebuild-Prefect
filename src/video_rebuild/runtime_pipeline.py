from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol

from .adapters.person import UltralyticsPersonAdapter
from .adapters.scenes import SceneDetectionAdapter
from .candidates import build_candidate_result
from .config import ToolPaths
from .media import probe_video
from .models import TimeSpan, VideoCandidateResult, VideoMeta


class SceneDetector(Protocol):
    def detect(self, video_path: Path, **kwargs: object) -> dict[str, Any]: ...


class PersonDetector(Protocol):
    def detect(self, video_path: Path, **kwargs: object) -> dict[str, Any]: ...


def _parse_spans(payload: dict[str, Any], key: str) -> list[TimeSpan]:
    items = payload.get(key)
    if not isinstance(items, list):
        raise ValueError(f"adapter payload field {key!r} must be a list")
    return [TimeSpan.model_validate(item) for item in items]


def run_candidate_analysis(
    video_path: Path,
    tools: ToolPaths,
    *,
    minimum_duration_s: float = 0.5,
    scene_threshold: float = 27.0,
    minimum_scene_length_frames: int = 15,
    person_confidence: float = 0.35,
    person_image_size: int = 640,
    person_gap_tolerance_frames: int = 2,
    scene_detector: SceneDetector | None = None,
    person_detector: PersonDetector | None = None,
    media_probe: Callable[[Path, Path], VideoMeta] = probe_video,
) -> VideoCandidateResult:
    video = media_probe(video_path, tools.ffprobe)
    scenes = scene_detector or SceneDetectionAdapter(
        python_path=tools.scene_python,
        worker_path=tools.scene_worker,
    )
    people = person_detector or UltralyticsPersonAdapter(
        python_path=tools.person_python,
        worker_path=tools.person_worker,
        model_path=tools.person_model,
    )
    scene_payload = scenes.detect(
        video_path,
        threshold=scene_threshold,
        minimum_scene_length_frames=minimum_scene_length_frames,
    )
    person_payload = people.detect(
        video_path,
        confidence=person_confidence,
        image_size=person_image_size,
        gap_tolerance_frames=person_gap_tolerance_frames,
    )
    return build_candidate_result(
        video=video,
        shots=_parse_spans(scene_payload, "scenes"),
        person_segments=_parse_spans(person_payload, "person_segments"),
        minimum_duration_s=minimum_duration_s,
    )
