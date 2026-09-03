from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

from video_rebuild.tooling import ToolSpec, probe_tool


class ToolProbeTests(unittest.TestCase):
    def test_reports_missing_tool_without_running_it(self) -> None:
        missing = Path(tempfile.gettempdir()) / "video-rebuild-definitely-missing.exe"

        result = probe_tool(ToolSpec(name="missing", path=missing))

        self.assertFalse(result.available)
        self.assertEqual(result.reason, "path_not_found")

    def test_reports_existing_python_interpreter(self) -> None:
        result = probe_tool(ToolSpec(name="python", path=Path(sys.executable)))

        self.assertTrue(result.available)
        self.assertIsNone(result.reason)


if __name__ == "__main__":
    unittest.main()

