from __future__ import annotations

from .models import GateStatus, QCDecision, QCGateResult, QCReport


def decide_qc(
    segment_id: str,
    gates: list[QCGateResult],
    analysis_confidence: float,
) -> QCReport:
    if not 0 <= analysis_confidence <= 1:
        raise ValueError("analysis_confidence must be between 0 and 1")

    passed = sum(gate.status == GateStatus.PASS for gate in gates)
    auto_score = passed / len(gates) if gates else 0.0

    if any(gate.status == GateStatus.FAIL for gate in gates):
        decision = QCDecision.RETRY
        recommended_action = "Retry the failed deterministic or render step."
    elif analysis_confidence < 0.6:
        decision = QCDecision.HUMAN_REVIEW
        recommended_action = "Review visual semantics before accepting the result."
    elif any(gate.status in {GateStatus.WARN, GateStatus.SKIP} for gate in gates):
        decision = QCDecision.HUMAN_REVIEW
        recommended_action = "Review skipped or warning quality gates."
    else:
        decision = QCDecision.PASS
        recommended_action = "Proceed to the next stage."

    return QCReport(
        segment_id=segment_id,
        gates=gates,
        auto_score=auto_score,
        decision=decision,
        recommended_action=recommended_action,
    )

