from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from pydantic import ValidationError

from video_rebuild.analysis_packets import (
    candidate_shards,
    map_object_frames,
    map_ocr_frames,
    select_measurement_frames,
    select_visual_evidence_sequences,
    select_coarse_frames,
    select_fine_frame_indices,
)
from video_rebuild.models import (
    AnalysisFrame,
    AnalysisPacket,
    FrameIndexEntry,
    OCRItem,
    TimeSpan,
    VideoMeta,
)


def frame_index(count: int, fps: float = 10.0) -> list[FrameIndexEntry]:
    return [
        FrameIndexEntry(
            frame_index=index,
            pts_s=index / fps,
            duration_s=1 / fps,
            key_frame=index == 0,
            picture_type="I" if index == 0 else "P",
        )
        for index in range(count)
    ]


class AnalysisPacketTests(unittest.TestCase):
    def test_visual_sequences_cover_both_boundaries_and_internal_peak(self) -> None:
        sequences = select_visual_evidence_sequences(
            TimeSpan(start_s=2.0, end_s=8.0),
            frame_index(100, fps=10),
            [{"peak_frame": 50, "peak_score": 0.5}],
            samples_per_second=10,
            boundary_context_s=0.4,
            boundary_inside_s=0.4,
            internal_radius_s=0.2,
        )
        by_purpose = {sequence.purpose: sequence for sequence in sequences}
        entrance = by_purpose["entrance"]
        exit_sequence = by_purpose["exit"]
        internal = by_purpose["internal"]
        self.assertLess(entrance.frames[0].pts_s, 2.0)
        self.assertGreaterEqual(entrance.frames[-1].pts_s, 2.0)
        self.assertLess(exit_sequence.frames[0].pts_s, 8.0)
        self.assertGreaterEqual(exit_sequence.frames[-1].pts_s, 8.0)
        self.assertIn(50, [frame.frame_index for frame in internal.frames])

    def test_candidates_are_split_into_bounded_parallel_shards(self) -> None:
        self.assertEqual(candidate_shards(list(range(7)), 2), [[0, 2, 4, 6], [1, 3, 5]])
        self.assertEqual(candidate_shards([1], 2), [[1]])
        with self.assertRaisesRegex(ValueError, "parallel_workers"):
            candidate_shards([1], 0)

    def test_coarse_sampling_uses_frame_anchors_and_context(self) -> None:
        selected = select_coarse_frames(
            TimeSpan(start_s=1.0, end_s=3.0),
            frame_index(50),
            samples_per_second=2.0,
            maximum_frames=20,
            context_s=0.3,
        )
        self.assertEqual([item.frame_index for item in selected if "coarse" in item.roles], [11, 17, 23, 29])
        self.assertEqual(selected[0].roles, ["context_before"])
        self.assertEqual(selected[-1].roles, ["context_after"])
        self.assertTrue(all(item.pts_s >= 0 for item in selected))

    def test_short_segment_produces_one_coarse_frame_without_duplicates(self) -> None:
        selected = select_coarse_frames(
            TimeSpan(start_s=1.0, end_s=1.4),
            frame_index(30),
            samples_per_second=2.0,
            maximum_frames=20,
            context_s=0.0,
        )
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0].frame_index, 12)

    def test_fine_frames_cover_change_windows_and_keep_peak_under_cap(self) -> None:
        selected = select_fine_frame_indices(
            [{"start_frame": 10, "end_frame": 14, "peak_frame": 12}],
            segment_start_frame=8,
            segment_end_frame=20,
            padding_frames=1,
            maximum_frames=5,
        )
        self.assertLessEqual(len(selected), 5)
        self.assertIn(12, selected)
        self.assertTrue(all(8 <= item < 20 for item in selected))

    def test_observations_map_by_resolved_input_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "frame.png"
            frames = [
                AnalysisFrame(
                    frame_index=10,
                    pts_s=1.0,
                    path=str(path),
                    roles=["coarse"],
                )
            ]
            ocr = map_ocr_frames(
                frames, {"frames": [{"input_path": str(path), "items": []}]}
            )
            objects = map_object_frames(
                frames, {"frames": [{"input_path": str(path), "items": []}]}
            )
            self.assertEqual(ocr[0].input_path, str(path))
            self.assertEqual(objects[0].input_path, str(path))

    def test_gpu_measurements_use_only_coarse_and_context_frames(self) -> None:
        frames = [
            AnalysisFrame(frame_index=1, pts_s=0.1, path="coarse.png", roles=["coarse"]),
            AnalysisFrame(frame_index=2, pts_s=0.2, path="fine.png", roles=["fine"]),
            AnalysisFrame(
                frame_index=3,
                pts_s=0.3,
                path="context.png",
                roles=["context_after"],
            ),
        ]
        selected = select_measurement_frames(frames, maximum_frames=2)
        self.assertEqual([frame.frame_index for frame in selected], [1, 3])

    def test_gpu_measurement_limit_fails_before_worker_launch(self) -> None:
        frames = [
            AnalysisFrame(
                frame_index=index,
                pts_s=index / 10,
                path=f"{index}.png",
                roles=["coarse"],
            )
            for index in range(3)
        ]
        with self.assertRaisesRegex(ValueError, "GPU measurement frame limit"):
            select_measurement_frames(frames, maximum_frames=2)

    def test_invalid_score_is_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            OCRItem(text="x", score=1.1, polygon=[])

    def test_context_frame_may_be_outside_candidate_but_inside_context(self) -> None:
        video = VideoMeta(
            path="E:/video.mp4", duration_s=10, width=100, height=100,
            fps=10, frame_count=100,
        )
        packet = AnalysisPacket(
            segment_id="candidate-0001",
            span=TimeSpan(start_s=1, end_s=2, start_frame=10, end_frame=20),
            context_span=TimeSpan(start_s=0.7, end_s=2.3, start_frame=7, end_frame=23),
            video=video,
            frames=[
                AnalysisFrame(
                    frame_index=7, pts_s=0.7, path="x.png", roles=["context_before"]
                )
            ],
            contact_sheet_path="sheet.jpg",
            ocr=[], objects=[], motion=None,
        )
        self.assertEqual(packet.frames[0].frame_index, 7)

    def test_frame_outside_context_is_rejected(self) -> None:
        video = VideoMeta(
            path="E:/video.mp4", duration_s=10, width=100, height=100,
            fps=10, frame_count=100,
        )
        with self.assertRaises(ValidationError):
            AnalysisPacket(
                segment_id="candidate-0001",
                span=TimeSpan(start_s=1, end_s=2),
                context_span=TimeSpan(start_s=0.7, end_s=2.3),
                video=video,
                frames=[
                    AnalysisFrame(
                        frame_index=30, pts_s=3.0, path="x.png", roles=["coarse"]
                    )
                ],
                contact_sheet_path="sheet.jpg",
                ocr=[], objects=[], motion=None,
            )


if __name__ == "__main__":
    unittest.main()
