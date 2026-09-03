from __future__ import annotations

from typing import Any
from pathlib import Path

from prefect import flow, task

from .analysis import build_stub_analysis
from .models import (
    FullFlowResult,
    GateStatus,
    QCGateResult,
    SegmentAnalysis,
    TimeSpan,
)
from .planning import build_baseline_plan
from .qc import decide_qc
from .config import load_config
from .runtime_pipeline import run_candidate_analysis
from .analysis_packets import prepare_analysis_packets
from .models import VideoCandidateResult
from .visual_analysis import CodexVisualAnalyzer


def run_contract_only_pipeline(segment_id: str, span: TimeSpan) -> FullFlowResult:
    analysis = build_stub_analysis(segment_id=segment_id, span=span)
    execution_plan = build_baseline_plan(analysis)
    qc = decide_qc(
        segment_id=segment_id,
        gates=[
            QCGateResult(
                gate="ai_visual_review",
                status=GateStatus.SKIP,
                message="Visual AI adapter has not run.",
                evidence={"analyzer": analysis.analyzer},
            )
        ],
        analysis_confidence=analysis.confidence,
    )
    return FullFlowResult(
        analysis=analysis,
        execution_plan=execution_plan,
        qc=qc,
    )


@task(name="build-stub-segment-analysis")
def build_stub_analysis_task(segment_id: str, span_payload: dict[str, float]) -> dict[str, Any]:
    span = TimeSpan.model_validate(span_payload)
    return build_stub_analysis(segment_id=segment_id, span=span).model_dump(mode="json")


@task(name="build-baseline-execution-plan")
def build_plan_task(analysis_payload: dict[str, Any]) -> dict[str, Any]:
    analysis = SegmentAnalysis.model_validate(analysis_payload)
    return build_baseline_plan(analysis).model_dump(mode="json")


@task(name="decide-stub-quality-gate")
def decide_stub_qc_task(analysis_payload: dict[str, Any]) -> dict[str, Any]:
    analysis = SegmentAnalysis.model_validate(analysis_payload)
    report = decide_qc(
        segment_id=analysis.segment_id,
        gates=[
            QCGateResult(
                gate="ai_visual_review",
                status=GateStatus.SKIP,
                message="Visual AI adapter has not run.",
                evidence={"analyzer": analysis.analyzer},
            )
        ],
        analysis_confidence=analysis.confidence,
    )
    return report.model_dump(mode="json")


@flow(name="video-rebuild-analysis-contract")
def analysis_flow(segment_id: str, start_s: float, end_s: float) -> dict[str, Any]:
    return build_stub_analysis_task(
        segment_id,
        {"start_s": start_s, "end_s": end_s},
    )


@flow(name="video-rebuild-execution-contract")
def execution_flow(analysis_payload: dict[str, Any]) -> dict[str, Any]:
    return build_plan_task(analysis_payload)


@flow(name="video-rebuild-qc-contract")
def qc_flow(analysis_payload: dict[str, Any]) -> dict[str, Any]:
    return decide_stub_qc_task(analysis_payload)


@flow(name="video-rebuild-full-contract-only")
def full_rebuild_flow(segment_id: str, start_s: float, end_s: float) -> dict[str, Any]:
    analysis_payload = analysis_flow(segment_id, start_s, end_s)
    plan_payload = execution_flow(analysis_payload)
    qc_payload = qc_flow(analysis_payload)
    return {
        "analysis": analysis_payload,
        "execution_plan": plan_payload,
        "qc": qc_payload,
    }


@task(name="analyze-video-candidates")
def analyze_video_candidates_task(
    video_path: str,
    config_path: str,
    minimum_duration_s: float,
) -> dict[str, Any]:
    tools = load_config(Path(config_path))
    result = run_candidate_analysis(
        Path(video_path),
        tools,
        minimum_duration_s=minimum_duration_s,
    )
    return result.model_dump(mode="json")


@flow(name="video-rebuild-candidate-analysis")
def candidate_analysis_flow(
    video_path: str,
    config_path: str,
    minimum_duration_s: float = 0.5,
) -> dict[str, Any]:
    return analyze_video_candidates_task(
        video_path,
        config_path,
        minimum_duration_s,
    )


@task(name="prepare-video-analysis-packets")
def prepare_analysis_packets_task(
    video_path: str,
    config_path: str,
    output_root: str,
    candidate_payload: dict[str, Any],
    coarse_samples_per_second: float = 2.0,
    maximum_coarse_frames: int = 30,
    context_s: float = 0.5,
    maximum_fine_frames: int = 30,
    parallel_workers: int = 2,
) -> list[dict[str, Any]]:
    result = VideoCandidateResult.model_validate(candidate_payload)
    packets = prepare_analysis_packets(
        Path(video_path),
        result,
        load_config(Path(config_path)),
        Path(output_root),
        coarse_samples_per_second=coarse_samples_per_second,
        maximum_coarse_frames=maximum_coarse_frames,
        context_s=context_s,
        maximum_fine_frames=maximum_fine_frames,
        parallel_workers=parallel_workers,
    )
    return [packet.model_dump(mode="json") for packet in packets]


@flow(name="video-rebuild-analysis-packets")
def analysis_packet_flow(
    video_path: str,
    config_path: str,
    output_root: str,
    minimum_duration_s: float = 0.5,
    coarse_samples_per_second: float = 2.0,
    maximum_coarse_frames: int = 30,
    context_s: float = 0.5,
    maximum_fine_frames: int = 30,
    parallel_workers: int = 2,
) -> dict[str, Any]:
    candidate_payload = analyze_video_candidates_task(
        video_path, config_path, minimum_duration_s
    )
    packets = prepare_analysis_packets_task(
        video_path,
        config_path,
        output_root,
        candidate_payload,
        coarse_samples_per_second,
        maximum_coarse_frames,
        context_s,
        maximum_fine_frames,
        parallel_workers,
    )
    return {"candidate_result": candidate_payload, "analysis_packets": packets}


@task(name="codex-visual-analysis")
def codex_visual_analysis_task(
    packet_path: str,
    output_path: str,
    require_human_review: bool = False,
    codex_executable: str = "codex",
    schema_path: str | None = None,
    timeout_seconds: float = 600.0,
) -> dict[str, Any]:
    result = CodexVisualAnalyzer(
        codex_executable=codex_executable,
        schema_path=Path(schema_path) if schema_path else None,
        timeout_seconds=timeout_seconds,
    ).analyze(
        packet_path=Path(packet_path),
        output_path=Path(output_path),
        require_human_review=require_human_review,
    )
    return result.model_dump(mode="json")


@flow(name="video-rebuild-codex-visual-analysis")
def codex_visual_analysis_flow(
    packet_path: str,
    output_path: str,
    require_human_review: bool = False,
    codex_executable: str = "codex",
    schema_path: str | None = None,
    timeout_seconds: float = 600.0,
) -> dict[str, Any]:
    return codex_visual_analysis_task(
        packet_path,
        output_path,
        require_human_review,
        codex_executable,
        schema_path,
        timeout_seconds,
    )
