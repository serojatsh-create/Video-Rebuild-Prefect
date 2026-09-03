from __future__ import annotations

import os
import subprocess
import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "scripts" / "run_ocr_frames.py"
CONFIG = PROJECT_ROOT / "config" / "tools.local.json"


class OCRCommandTests(unittest.TestCase):
    def test_rejects_missing_frame_before_worker_launch(self) -> None:
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(PROJECT_ROOT / "src")
        missing_frame = PROJECT_ROOT / "runs" / "fixtures" / "missing.png"

        completed = subprocess.run(
            [sys.executable, str(SCRIPT), str(CONFIG), str(missing_frame)],
            capture_output=True,
            text=True,
            encoding="utf-8",
            env=environment,
            check=False,
        )

        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("frame does not exist", completed.stderr)


if __name__ == "__main__":
    unittest.main()
