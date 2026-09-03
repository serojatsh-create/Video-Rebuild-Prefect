from __future__ import annotations

from .models import TimeSpan


def subtract_intervals(
    shots: list[TimeSpan],
    exclusions: list[TimeSpan],
    minimum_duration_s: float,
) -> list[TimeSpan]:
    if minimum_duration_s < 0:
        raise ValueError("minimum_duration_s must be non-negative")

    results: list[TimeSpan] = []
    ordered_exclusions = sorted(exclusions, key=lambda item: (item.start_s, item.end_s))

    for shot in sorted(shots, key=lambda item: (item.start_s, item.end_s)):
        fragments = [(shot.start_s, shot.end_s)]
        for exclusion in ordered_exclusions:
            next_fragments: list[tuple[float, float]] = []
            for start_s, end_s in fragments:
                if exclusion.end_s <= start_s or exclusion.start_s >= end_s:
                    next_fragments.append((start_s, end_s))
                    continue
                if exclusion.start_s > start_s:
                    next_fragments.append((start_s, min(exclusion.start_s, end_s)))
                if exclusion.end_s < end_s:
                    next_fragments.append((max(exclusion.end_s, start_s), end_s))
            fragments = next_fragments

        for start_s, end_s in fragments:
            if end_s - start_s >= minimum_duration_s:
                results.append(TimeSpan(start_s=start_s, end_s=end_s))

    return results

