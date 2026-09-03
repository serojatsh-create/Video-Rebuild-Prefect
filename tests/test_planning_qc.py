from __future__ import annotations

import unittest

from video_rebuild.models import (
    AnalysisStatus,
    GateStatus,
    QCDecision,
    QCGateResult,
    SegmentAnalysis,
    TimeSpan,
)
from video_rebuild.planning import build_baseline_plan
from video_rebuild.qc import decide_qc


class PlanningTests(unittest.TestCase):
    def test_builds_duration_validation_for_segment(self) -> None:
        analysis = SegmentAnalysis(
            segment_id="seg-001",
            span=TimeSpan(start_s=5.0, end_s=8.5),
            status=AnalysisStatus.NEEDS_AI,
            visual_summary="stub",
            confidence=0.2,
        )

        plan = build_baseline_plan(analysis)

        self.assertEqual(plan.segment_id, "seg-001")
        self.assertEqual(plan.operations[0].start_s, 0.0)
        self.assertEqual(plan.operations[0].end_s, 3.5)
        self.assertEqual(plan.validation_targets[0].expected, 3.5)


class QCDecisionTests(unittest.TestCase):
    def test_low_confidence_requires_human_review(self) -> None:
        report = decide_qc(
            segment_id="seg-001",
            gates=[
                QCGateResult(
                    gate="render_exists",
                    status=GateStatus.PASS,
                    message="render exists",
                )
            ],
            analysis_confidence=0.2,
        )

        self.assertEqual(report.decision, QCDecision.HUMAN_REVIEW)

    def test_failed_gate_requests_retry(self) -> None:
        report = decide_qc(
            segment_id="seg-001",
            gates=[
                QCGateResult(
                    gate="duration_match",
                    status=GateStatus.FAIL,
                    message="duration mismatch",
                )
            ],
            analysis_confidence=0.9,
        )

        self.assertEqual(report.decision, QCDecision.RETRY)

    def test_high_confidence_passes_when_all_gates_pass(self) -> None:
        report = decide_qc(
            segment_id="seg-001",
            gates=[
                QCGateResult(
                    gate="render_exists",
                    status=GateStatus.PASS,
                    message="render exists",
                )
            ],
            analysis_confidence=0.9,
        )

        self.assertEqual(report.decision, QCDecision.PASS)


if __name__ == "__main__":
    unittest.main()

