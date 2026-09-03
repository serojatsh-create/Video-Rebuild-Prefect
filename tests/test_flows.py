from __future__ import annotations

import unittest

from prefect import Flow, Task

from video_rebuild.flows import (
    analysis_packet_flow,
    analysis_flow,
    build_stub_analysis_task,
    codex_visual_analysis_flow,
    codex_visual_analysis_task,
    full_rebuild_flow,
    prepare_analysis_packets_task,
    run_contract_only_pipeline,
)
from video_rebuild.models import QCDecision, TimeSpan


class FlowContractTests(unittest.TestCase):
    def test_prefect_objects_are_real_flows_and_tasks(self) -> None:
        self.assertIsInstance(analysis_flow, Flow)
        self.assertIsInstance(full_rebuild_flow, Flow)
        self.assertIsInstance(build_stub_analysis_task, Task)
        self.assertIsInstance(analysis_packet_flow, Flow)
        self.assertIsInstance(prepare_analysis_packets_task, Task)
        self.assertIsInstance(codex_visual_analysis_flow, Flow)
        self.assertIsInstance(codex_visual_analysis_task, Task)

    def test_contract_only_pipeline_stops_at_human_review(self) -> None:
        result = run_contract_only_pipeline(
            segment_id="seg-001",
            span=TimeSpan(start_s=0.0, end_s=2.0),
        )

        self.assertEqual(result.qc.decision, QCDecision.HUMAN_REVIEW)
        self.assertEqual(result.execution_plan.segment_id, "seg-001")


if __name__ == "__main__":
    unittest.main()
