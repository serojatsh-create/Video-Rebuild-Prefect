from __future__ import annotations

import unittest

from pydantic import ValidationError

from video_rebuild.models import AnalysisStatus, SegmentAnalysis, TimeSpan


class TimeSpanTests(unittest.TestCase):
    def test_rejects_end_not_after_start(self) -> None:
        with self.assertRaises(ValidationError):
            TimeSpan(start_s=2.0, end_s=2.0)

    def test_serializes_segment_analysis_to_json_safe_data(self) -> None:
        analysis = SegmentAnalysis(
            segment_id="seg-001",
            span=TimeSpan(start_s=0.0, end_s=1.0),
            status=AnalysisStatus.NEEDS_AI,
            visual_summary="Semantic analysis has not run.",
            confidence=0.0,
            uncertainties=["visual adapter unavailable"],
        )

        payload = analysis.model_dump(mode="json")

        self.assertEqual(payload["status"], "needs_ai")
        self.assertEqual(payload["span"], {"start_s": 0.0, "end_s": 1.0})


if __name__ == "__main__":
    unittest.main()

