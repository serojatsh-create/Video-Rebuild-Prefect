from __future__ import annotations


def frames_to_span_dicts(
    presence: list[bool],
    fps: float,
    gap_tolerance_frames: int = 0,
) -> list[dict[str, float | int]]:
    if fps <= 0:
        raise ValueError("fps must be greater than zero")
    if gap_tolerance_frames < 0:
        raise ValueError("gap_tolerance_frames must be non-negative")

    normalized = list(presence)
    if gap_tolerance_frames:
        index = 0
        while index < len(normalized):
            if normalized[index]:
                index += 1
                continue
            gap_start = index
            while index < len(normalized) and not normalized[index]:
                index += 1
            gap_end = index
            bounded_by_presence = gap_start > 0 and gap_end < len(normalized)
            if bounded_by_presence and gap_end - gap_start <= gap_tolerance_frames:
                normalized[gap_start:gap_end] = [True] * (gap_end - gap_start)

    spans: list[dict[str, float | int]] = []
    index = 0
    while index < len(normalized):
        if not normalized[index]:
            index += 1
            continue
        start_frame = index
        while index < len(normalized) and normalized[index]:
            index += 1
        spans.append(
            {
                "start_s": start_frame / fps,
                "end_s": index / fps,
                "start_frame": start_frame,
                "end_frame": index,
            }
        )
    return spans
