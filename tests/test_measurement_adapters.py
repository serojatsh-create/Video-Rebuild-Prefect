from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

from video_rebuild.adapters.motion import MotionAnalysisAdapter
from video_rebuild.adapters.objects import YOLOObjectAdapter
from workers.motion_worker import _threshold
from workers.object_worker import prediction_options


class MeasurementAdapterTests(unittest.TestCase):
    def test_object_worker_streams_large_frame_lists(self) -> None:
        options = prediction_options({"confidence": 0.3, "image_size": 512}, "cuda:0")
        self.assertTrue(options["stream"])
        self.assertEqual(options["device"], "cuda:0")

    def test_motion_threshold_detects_change_above_low_video_noise(self) -> None:
        scores = [0.00012, 0.00014, 0.00016, 0.00015, 0.0029]
        threshold = _threshold(scores)
        self.assertLess(threshold, 0.0029)
        self.assertGreaterEqual(threshold, 0.0005)

    def test_motion_adapter_sends_frame_range(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            worker = root / "worker.py"
            worker.write_text(
                "import json, sys\n"
                "payload=json.loads(sys.stdin.read())\n"
                "print(json.dumps(payload))\n",
                encoding="utf-8",
            )
            adapter = MotionAnalysisAdapter(Path(sys.executable), worker)
            result = adapter.analyze(Path("E:/video.mp4"), 10, 20)
            self.assertEqual(result["start_frame"], 10)
            self.assertEqual(result["end_frame"], 20)

    def test_object_adapter_batches_all_frames(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            worker = root / "worker.py"
            worker.write_text(
                "import json, sys\n"
                "payload=json.loads(sys.stdin.read())\n"
                "print(json.dumps(payload))\n",
                encoding="utf-8",
            )
            adapter = YOLOObjectAdapter(
                Path(sys.executable), worker, root / "model.pt"
            )
            result = adapter.detect([root / "a.png", root / "b.png"])
            self.assertEqual(result["frame_paths"], [str(root / "a.png"), str(root / "b.png")])


if __name__ == "__main__":
    unittest.main()
