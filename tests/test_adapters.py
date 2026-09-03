from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

from video_rebuild.adapters.base import (
    DependencyUnavailableError,
    JsonAdapterError,
    JsonSubprocessAdapter,
)


class JsonSubprocessAdapterTests(unittest.TestCase):
    def test_rejects_missing_runtime_before_launch(self) -> None:
        adapter = JsonSubprocessAdapter(
            name="person",
            python_path=Path("E:/missing/python.exe"),
            worker_path=Path("E:/missing/worker.py"),
        )

        with self.assertRaisesRegex(DependencyUnavailableError, "person.*python"):
            adapter.run({"video_path": "E:/video.mp4"})

    def test_rejects_non_json_worker_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            worker = Path(temporary_directory) / "worker.py"
            worker.write_text("print('not-json')\n", encoding="utf-8")
            adapter = JsonSubprocessAdapter(
                name="test",
                python_path=Path(sys.executable),
                worker_path=worker,
            )

            with self.assertRaisesRegex(JsonAdapterError, "invalid JSON"):
                adapter.run({"value": 1})

    def test_returns_json_object_from_worker(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            worker = Path(temporary_directory) / "worker.py"
            worker.write_text(
                "import json, sys\n"
                "payload = json.loads(sys.stdin.read())\n"
                "print(json.dumps({'received': payload['value']}))\n",
                encoding="utf-8",
            )
            adapter = JsonSubprocessAdapter(
                name="test",
                python_path=Path(sys.executable),
                worker_path=worker,
            )

            result = adapter.run({"value": 7})

            self.assertEqual(result, {"received": 7})

    def test_worker_stdout_is_forced_to_utf8(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            worker = Path(temporary_directory) / "worker.py"
            worker.write_text(
                "import json, os\n"
                "print(json.dumps({'text': '中文', 'encoding': os.environ.get('PYTHONIOENCODING')}, ensure_ascii=False))\n",
                encoding="utf-8",
            )
            adapter = JsonSubprocessAdapter(
                name="test", python_path=Path(sys.executable), worker_path=worker
            )
            result = adapter.run({"value": 1})
            self.assertEqual(result, {"text": "中文", "encoding": "utf-8"})


if __name__ == "__main__":
    unittest.main()
