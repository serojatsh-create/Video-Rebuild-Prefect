from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

from video_rebuild.adapters.scenes import SceneDetectionAdapter


class SceneDetectionAdapterTests(unittest.TestCase):
    def test_detect_sends_explicit_threshold_and_minimum_scene_length(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            worker = root / "worker.py"
            worker.write_text(
                "import json, sys\n"
                "payload = json.loads(sys.stdin.read())\n"
                "print(json.dumps(payload))\n",
                encoding="utf-8",
            )
            adapter = SceneDetectionAdapter(
                python_path=Path(sys.executable),
                worker_path=worker,
            )

            result = adapter.detect(
                Path("E:/reference/video.mp4"),
                threshold=31.0,
                minimum_scene_length_frames=12,
            )

            self.assertEqual(result["action"], "detect")
            self.assertEqual(result["video_path"], "E:\\reference\\video.mp4")
            self.assertEqual(result["threshold"], 31.0)
            self.assertEqual(result["minimum_scene_length_frames"], 12)


if __name__ == "__main__":
    unittest.main()
