from __future__ import annotations

import unittest

from video_rebuild.analysis import build_stub_analysis
from video_rebuild.models import AnalysisStatus, TimeSpan


class StubAnalysisTests(unittest.TestCase):
    def test_stub_is_explicitly_marked_needs_ai(self) -> None:
        result = build_stub_analysis(
            segment_id="seg-001",
            span=TimeSpan(start_s=2.0, end_s=4.0),
        )

        self.assertEqual(result.status, AnalysisStatus.NEEDS_AI)
        self.assertEqual(result.confidence, 0.0)
        self.assertIn("not run", result.visual_summary.lower())


if __name__ == "__main__":
    unittest.main()

