from __future__ import annotations

import os
import subprocess
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PERSON_PYTHON = PROJECT_ROOT / "runtimes" / "person_cpu" / "Scripts" / "python.exe"


class PersonRuntimeContractTests(unittest.TestCase):
    @unittest.skipUnless(PERSON_PYTHON.is_file(), "person runtime download is incomplete")
    def test_standard_library_frame_helper_imports_without_pydantic(self) -> None:
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(PROJECT_ROOT / "src")
        completed = subprocess.run(
            [
                str(PERSON_PYTHON),
                "-c",
                "from video_rebuild.person_frames import frames_to_span_dicts; "
                "print(frames_to_span_dicts([True, False], 2.0))",
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            env=environment,
            check=False,
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("start_s", completed.stdout)


if __name__ == "__main__":
    unittest.main()

