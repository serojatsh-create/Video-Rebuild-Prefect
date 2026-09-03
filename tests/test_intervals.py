from __future__ import annotations

import unittest

from video_rebuild.intervals import subtract_intervals
from video_rebuild.models import TimeSpan


class SubtractIntervalsTests(unittest.TestCase):
    def test_splits_shot_around_person_interval(self) -> None:
        result = subtract_intervals(
            shots=[TimeSpan(start_s=0.0, end_s=10.0)],
            exclusions=[TimeSpan(start_s=3.0, end_s=7.0)],
            minimum_duration_s=0.5,
        )

        self.assertEqual(
            [item.model_dump() for item in result],
            [
                {"start_s": 0.0, "end_s": 3.0},
                {"start_s": 7.0, "end_s": 10.0},
            ],
        )

    def test_drops_fragments_shorter_than_minimum_duration(self) -> None:
        result = subtract_intervals(
            shots=[TimeSpan(start_s=0.0, end_s=2.0)],
            exclusions=[TimeSpan(start_s=0.2, end_s=1.8)],
            minimum_duration_s=0.5,
        )

        self.assertEqual(result, [])

    def test_clips_exclusion_to_shot_boundaries(self) -> None:
        result = subtract_intervals(
            shots=[TimeSpan(start_s=5.0, end_s=9.0)],
            exclusions=[TimeSpan(start_s=1.0, end_s=6.0)],
            minimum_duration_s=0.5,
        )

        self.assertEqual(
            [item.model_dump() for item in result],
            [{"start_s": 6.0, "end_s": 9.0}],
        )


if __name__ == "__main__":
    unittest.main()

