from __future__ import annotations

from .models import (
    ExecutionOperation,
    ExecutionPlan,
    ImplementationStrategy,
    SegmentAnalysis,
    ValidationTarget,
)


def build_baseline_plan(analysis: SegmentAnalysis) -> ExecutionPlan:
    duration_s = analysis.span.end_s - analysis.span.start_s
    return ExecutionPlan(
        segment_id=analysis.segment_id,
        strategy=ImplementationStrategy.JIANYING,
        router_version="baseline-v0",
        operations=[
            ExecutionOperation(
                id="op-001",
                tool="jianying",
                action="hold",
                start_s=0.0,
                end_s=duration_s,
                parameters={"description": analysis.visual_summary},
            )
        ],
        validation_targets=[
            ValidationTarget(
                name="duration",
                metric="duration_s",
                expected=duration_s,
                tolerance=0.15,
            )
        ],
        fallbacks=[ImplementationStrategy.HYBRID, ImplementationStrategy.SVG],
        notes=[
            "Baseline plan only; visual semantics and editor execution require adapters."
        ],
    )

