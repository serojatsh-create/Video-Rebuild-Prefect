from __future__ import annotations

from .intervals import subtract_intervals
from .models import (
    CandidateSegment,
    TimeSpan,
    VideoCandidateResult,
    VideoMeta,
)


def _clamp_spans(spans: list[TimeSpan], duration_s: float) -> list[TimeSpan]:
    clamped: list[TimeSpan] = []
    for span in spans:
        start_s = min(span.start_s, duration_s)
        end_s = min(span.end_s, duration_s)
        if end_s > start_s:
            clamped.append(TimeSpan(start_s=start_s, end_s=end_s))
    return clamped


def build_candidate_result(
    *,
    video: VideoMeta,
    shots: list[TimeSpan],
    person_segments: list[TimeSpan],
    minimum_duration_s: float,
) -> VideoCandidateResult:
    bounded_shots = _clamp_spans(shots, video.duration_s)
    bounded_person_segments = _clamp_spans(person_segments, video.duration_s)
    candidate_spans = subtract_intervals(
        bounded_shots,
        bounded_person_segments,
        minimum_duration_s=minimum_duration_s,
    )
    anchored_candidate_spans: list[TimeSpan] = []
    for span in candidate_spans:
        start_frame = max(0, round(span.start_s * video.fps))
        end_frame = max(start_frame + 1, round(span.end_s * video.fps))
        anchored_candidate_spans.append(
            TimeSpan(
                start_s=span.start_s,
                end_s=span.end_s,
                start_frame=start_frame,
                end_frame=end_frame,
            )
        )
    candidates = [
        CandidateSegment(segment_id=f"candidate-{index:04d}", span=span)
        for index, span in enumerate(anchored_candidate_spans, start=1)
    ]
    return VideoCandidateResult(
        video=video,
        shots=bounded_shots,
        person_segments=bounded_person_segments,
        candidates=candidates,
    )
