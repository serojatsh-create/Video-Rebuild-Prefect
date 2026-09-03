from __future__ import annotations

import unittest

from video_rebuild.candidates import build_candidate_result
from video_rebuild.models import TimeSpan, VideoMeta


class CandidatePipelineTests(unittest.TestCase):
    def test_subtracts_person_segments_from_each_detected_scene(self) -> None:
        result = build_candidate_result(
            video=VideoMeta(
                path="E:/reference.mp4",
                duration_s=10.0,
                width=1920,
                height=1080,
                fps=30.0,
                frame_count=300,
            ),
            shots=[TimeSpan(start_s=0.0, end_s=5.0), TimeSpan(start_s=5.0, end_s=10.0)],
            person_segments=[TimeSpan(start_s=2.0, end_s=3.0), TimeSpan(start_s=6.0, end_s=9.0)],
            minimum_duration_s=1.0,
        )

        self.assertEqual(
            [candidate.span.model_dump() for candidate in result.candidates],
            [
                {"start_s": 0.0, "end_s": 2.0, "start_frame": 0, "end_frame": 60},
                {"start_s": 3.0, "end_s": 5.0, "start_frame": 90, "end_frame": 150},
                {"start_s": 5.0, "end_s": 6.0, "start_frame": 150, "end_frame": 180},
                {"start_s": 9.0, "end_s": 10.0, "start_frame": 270, "end_frame": 300},
            ],
        )
        self.assertEqual(
            [candidate.segment_id for candidate in result.candidates],
            ["candidate-0001", "candidate-0002", "candidate-0003", "candidate-0004"],
        )

    def test_clamps_worker_ranges_to_video_duration(self) -> None:
        result = build_candidate_result(
            video=VideoMeta(
                path="E:/reference.mp4",
                duration_s=2.0,
                width=1080,
                height=1920,
                fps=30.0,
            ),
            shots=[TimeSpan(start_s=0.0, end_s=2.5)],
            person_segments=[TimeSpan(start_s=1.8, end_s=3.0)],
            minimum_duration_s=0.1,
        )

        self.assertEqual(
            [candidate.span.model_dump() for candidate in result.candidates],
            [{"start_s": 0.0, "end_s": 1.8, "start_frame": 0, "end_frame": 54}],
        )
        self.assertEqual(result.shots[0].end_s, 2.0)
        self.assertEqual(result.person_segments[0].end_s, 2.0)


if __name__ == "__main__":
    unittest.main()
