from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

from video_rebuild.adapters.ocr import PaddleOCRAdapter


class PaddleOCRAdapterTests(unittest.TestCase):
    def test_probe_passes_only_explicit_local_model_directories(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            worker = root / "worker.py"
            worker.write_text(
                "import json, sys\n"
                "payload = json.loads(sys.stdin.read())\n"
                "print(json.dumps(payload))\n",
                encoding="utf-8",
            )
            adapter = PaddleOCRAdapter(
                python_path=Path(sys.executable),
                worker_path=worker,
                detection_model_dir=root / "det",
                recognition_model_dir=root / "rec",
            )

            result = adapter.probe()

            self.assertEqual(result["action"], "probe")
            self.assertEqual(result["detection_model_dir"], str(root / "det"))
            self.assertEqual(result["recognition_model_dir"], str(root / "rec"))
            self.assertEqual(result["detection_model_name"], "PP-OCRv6_tiny_det")
            self.assertEqual(result["recognition_model_name"], "PP-OCRv6_tiny_rec")
            self.assertEqual(result["device"], "auto")

    def test_recognize_sends_frame_paths_and_threshold(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            worker = root / "worker.py"
            worker.write_text(
                "import json, sys\n"
                "payload = json.loads(sys.stdin.read())\n"
                "print(json.dumps(payload))\n",
                encoding="utf-8",
            )
            adapter = PaddleOCRAdapter(
                python_path=Path(sys.executable),
                worker_path=worker,
                detection_model_dir=root / "det",
                recognition_model_dir=root / "rec",
            )

            result = adapter.recognize(
                [root / "frame-001.png"],
                language="ch",
                minimum_score=0.75,
            )

            self.assertEqual(result["action"], "recognize")
            self.assertEqual(result["frame_paths"], [str(root / "frame-001.png")])
            self.assertEqual(result["language"], "ch")
            self.assertEqual(result["minimum_score"], 0.75)
            self.assertEqual(result["detection_model_name"], "PP-OCRv6_tiny_det")
            self.assertEqual(result["recognition_model_name"], "PP-OCRv6_tiny_rec")
            self.assertEqual(result["device"], "auto")


if __name__ == "__main__":
    unittest.main()
