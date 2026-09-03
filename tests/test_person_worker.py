from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path

from workers.person_worker import resolve_device


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKER = PROJECT_ROOT / "workers" / "person_worker.py"


class PersonWorkerTests(unittest.TestCase):
    def test_auto_device_prefers_cuda_when_available(self) -> None:
        self.assertEqual(resolve_device("auto", cuda_available=True), "cuda:0")

    def test_auto_device_falls_back_to_cpu_when_cuda_is_unavailable(self) -> None:
        self.assertEqual(resolve_device("auto", cuda_available=False), "cpu")

    def test_explicit_cpu_remains_cpu_when_cuda_is_available(self) -> None:
        self.assertEqual(resolve_device("cpu", cuda_available=True), "cpu")

    def test_rejects_paddle_style_gpu_device(self) -> None:
        with self.assertRaisesRegex(ValueError, "device"):
            resolve_device("gpu:0", cuda_available=True)

    def test_probe_returns_structured_dependency_state(self) -> None:
        completed = subprocess.run(
            [sys.executable, str(WORKER)],
            input=json.dumps(
                {
                    "action": "probe",
                    "model_path": str(
                        PROJECT_ROOT / "downloads" / "models" / "person" / "yolo11n.pt"
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
        self.assertEqual(payload["adapter"], "person-ultralytics")
        self.assertIn("ultralytics", payload["packages"])
        self.assertIn("model_exists", payload)


if __name__ == "__main__":
    unittest.main()
