from __future__ import annotations

from .models import AnalysisStatus, SegmentAnalysis, TimeSpan


def build_stub_analysis(segment_id: str, span: TimeSpan) -> SegmentAnalysis:
    return SegmentAnalysis(
        segment_id=segment_id,
        span=span,
        status=AnalysisStatus.NEEDS_AI,
        visual_summary="Visual semantic analysis has not run.",
        confidence=0.0,
        uncertainties=["visual AI adapter is unavailable or has not been invoked"],
        analyzer="stub-v0",
    )

