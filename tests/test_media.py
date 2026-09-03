from __future__ import annotations

import unittest
from pathlib import Path

from video_rebuild.media import parse_ffprobe_payload


class FFprobePayloadTests(unittest.TestCase):
    def test_parses_primary_video_stream(self) -> None:
        payload = {
            "streams": [
                {
                    "codec_type": "video",
                    "width": 1920,
                    "height": 1080,
                    "avg_frame_rate": "30000/1001",
                    "r_frame_rate": "30000/1001",
                    "time_base": "1/30000",
                    "nb_frames": "180",
                },
                {"codec_type": "audio"},
            ],
            "format": {"duration": "6.006", "start_time": "0.000000"},
        }

        meta = parse_ffprobe_payload(Path("E:/reference.mp4"), payload)

        self.assertEqual(meta.width, 1920)
        self.assertEqual(meta.height, 1080)
        self.assertAlmostEqual(meta.fps, 29.97002997)
        self.assertEqual(meta.frame_count, 180)
        self.assertAlmostEqual(meta.duration_s, 6.006)
        self.assertEqual(meta.avg_frame_rate, "30000/1001")
        self.assertEqual(meta.reported_frame_rate, "30000/1001")
        self.assertEqual(meta.time_base, "1/30000")
        self.assertFalse(meta.variable_frame_rate)

    def test_rejects_payload_without_video_stream(self) -> None:
        with self.assertRaisesRegex(ValueError, "video stream"):
            parse_ffprobe_payload(
                Path("E:/audio.wav"),
                {"streams": [{"codec_type": "audio"}], "format": {"duration": "1"}},
            )


if __name__ == "__main__":
    unittest.main()
