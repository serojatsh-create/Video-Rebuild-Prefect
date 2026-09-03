from __future__ import annotations

import unittest
from pathlib import Path

from video_rebuild.config import ToolPaths
from video_rebuild.models import VideoMeta
from video_rebuild.runtime_pipeline import run_candidate_analysis


class FakeSceneDetector:
    def detect(self, video_path: Path, **_: object) -> dict[str, object]:
        return {
            "scenes": [
                {"start_s": 0.0, "end_s": 5.0},
                {"start_s": 5.0, "end_s": 6.0},
            ]
        }


class FakePersonDetector:
    def detect(self, video_path: Path, **_: object) -> dict[str, object]:
        return {"person_segments": [{"start_s": 0.0, "end_s": 5.0}]}


class RuntimePipelineTests(unittest.TestCase):
    def test_composes_media_probe_scenes_people_and_candidates(self) -> None:
        tools = ToolPaths(
            ffmpeg=Path("E:/tools/ffmpeg.exe"),
            ffprobe=Path("E:/tools/ffprobe.exe"),
            effect_analysis_python=Path("E:/effect/python.exe"),
            effect_analysis_project=Path("E:/effect"),
            scene_python=Path("E:/effect/python.exe"),
            scene_worker=Path("E:/project/workers/scene_worker.py"),
            person_python=Path("E:/person/python.exe"),
            person_worker=Path("E:/project/workers/person_worker.py"),
            person_model=Path("E:/project/models/yolo11n.pt"),
            ocr_python=Path("E:/ocr/python.exe"),
            ocr_worker=Path("E:/project/workers/ocr_worker.py"),
            ocr_detection_model=Path("E:/project/models/PP-OCRv6_tiny_det"),
            ocr_recognition_model=Path("E:/project/models/PP-OCRv6_tiny_rec"),
        )

        result = run_candidate_analysis(
            Path("E:/reference.mp4"),
            tools,
            minimum_duration_s=0.5,
            scene_detector=FakeSceneDetector(),
            person_detector=FakePersonDetector(),
            media_probe=lambda *_: VideoMeta(
                path="E:/reference.mp4",
                duration_s=6.0,
                width=1920,
                height=1080,
                fps=30.0,
                frame_count=180,
            ),
        )

        self.assertEqual(len(result.candidates), 1)
        self.assertEqual(result.candidates[0].span.start_s, 5.0)
        self.assertEqual(result.candidates[0].span.end_s, 6.0)


if __name__ == "__main__":
    unittest.main()
