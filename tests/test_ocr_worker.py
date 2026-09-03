from __future__ import annotations

import json
import subprocess
import unittest
from pathlib import Path

from workers.ocr_worker import normalize_prediction, resolve_device


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OCR_PYTHON = PROJECT_ROOT / "runtimes" / "ocr_cpu" / "Scripts" / "python.exe"
WORKER = PROJECT_ROOT / "workers" / "ocr_worker.py"


class OCRWorkerTests(unittest.TestCase):
    def test_auto_device_prefers_gpu_when_cuda_is_available(self) -> None:
        self.assertEqual(resolve_device("auto", cuda_available=True), "gpu:0")

    def test_auto_device_falls_back_to_cpu_when_cuda_is_unavailable(self) -> None:
        self.assertEqual(resolve_device("auto", cuda_available=False), "cpu")

    def test_normalizes_paddleocr_result_json_payload(self) -> None:
        class FakeResult:
            json = {
                "res": {
                    "input_path": "frame.png",
                    "rec_texts": ["文字"],
                    "rec_scores": [0.9],
                    "rec_polys": [[[0, 0], [1, 0], [1, 1], [0, 1]]],
                }
            }

        normalized = normalize_prediction(FakeResult())

        self.assertEqual(normalized["input_path"], "frame.png")
        self.assertEqual(normalized["rec_texts"], ["文字"])

    @unittest.skipUnless(OCR_PYTHON.is_file(), "OCR runtime download is incomplete")
    def test_probe_reports_ready_cpu_runtime_and_local_models(self) -> None:
        completed = subprocess.run(
            [str(OCR_PYTHON), str(WORKER)],
            input=json.dumps(
                {
                    "action": "probe",
                    "detection_model_dir": str(
                        PROJECT_ROOT / "downloads" / "models" / "ocr" / "PP-OCRv6_tiny_det"
                    ),
                    "recognition_model_dir": str(
                        PROJECT_ROOT / "downloads" / "models" / "ocr" / "PP-OCRv6_tiny_rec"
                    ),
                }
            ),
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=False,
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        payload = json.loads(completed.stdout)
        self.assertEqual(payload["adapter"], "paddleocr-v6")
        self.assertTrue(payload["packages"]["paddleocr"])
        self.assertTrue(payload["available"])
        self.assertEqual(payload["device"], "cpu")


if __name__ == "__main__":
    unittest.main()
