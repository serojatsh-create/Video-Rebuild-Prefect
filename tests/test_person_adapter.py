from __future__ import annotations

import unittest
import sys
import tempfile
from pathlib import Path

from video_rebuild.adapters.person import UltralyticsPersonAdapter, frames_to_spans


class PersonFrameAggregationTests(unittest.TestCase):
    def test_converts_presence_frames_to_half_open_time_spans(self) -> None:
        spans = frames_to_spans(
            presence=[False, True, True, False, True],
            fps=2.0,
        )

        self.assertEqual(
            [span.model_dump() for span in spans],
            [
                {"start_s": 0.5, "end_s": 1.5, "start_frame": 1, "end_frame": 3},
                {"start_s": 2.0, "end_s": 2.5, "start_frame": 4, "end_frame": 5},
            ],
        )

    def test_merges_short_absence_gap(self) -> None:
        spans = frames_to_spans(
            presence=[True, True, False, True, True],
            fps=5.0,
            gap_tolerance_frames=1,
        )

        self.assertEqual(
            [span.model_dump() for span in spans],
            [{"start_s": 0.0, "end_s": 1.0, "start_frame": 0, "end_frame": 5}],
        )

    def test_rejects_non_positive_fps(self) -> None:
        with self.assertRaisesRegex(ValueError, "fps"):
            frames_to_spans([True], fps=0)


class UltralyticsPersonAdapterTests(unittest.TestCase):
    def test_detect_passes_explicit_cpu_model_and_video_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            worker = root / "worker.py"
            worker.write_text(
                "import json, sys\n"
                "payload = json.loads(sys.stdin.read())\n"
                "print(json.dumps(payload))\n",
                encoding="utf-8",
            )
            adapter = UltralyticsPersonAdapter(
                python_path=Path(sys.executable),
                worker_path=worker,
                model_path=root / "yolo11n.pt",
            )

            result = adapter.detect(
                Path("E:/reference/video.mp4"),
                confidence=0.4,
                image_size=512,
                gap_tolerance_frames=3,
            )

            self.assertEqual(result["action"], "detect")
            self.assertEqual(result["video_path"], "E:\\reference\\video.mp4")
            self.assertEqual(result["model_path"], str(root / "yolo11n.pt"))
            self.assertEqual(result["confidence"], 0.4)
            self.assertEqual(result["image_size"], 512)
            self.assertEqual(result["gap_tolerance_frames"], 3)
            self.assertEqual(result["device"], "auto")


if __name__ == "__main__":
    unittest.main()
