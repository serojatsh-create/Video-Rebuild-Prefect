from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from .models import FrameIndexEntry, VideoMeta


class MediaProbeError(RuntimeError):
    pass


def _parse_rate(raw: str) -> float:
    numerator_text, denominator_text = raw.split("/", maxsplit=1)
    denominator = float(denominator_text)
    if denominator == 0:
        raise ValueError("frame-rate denominator must not be zero")
    return float(numerator_text) / denominator


def parse_ffprobe_payload(path: Path, payload: dict[str, Any]) -> VideoMeta:
    streams = payload.get("streams")
    if not isinstance(streams, list):
        raise ValueError("ffprobe payload has no streams list")

    video_stream = next(
        (stream for stream in streams if stream.get("codec_type") == "video"),
        None,
    )
    if video_stream is None:
        raise ValueError("ffprobe payload has no video stream")

    format_data = payload.get("format")
    if not isinstance(format_data, dict):
        raise ValueError("ffprobe payload has no format object")

    frame_count_raw = video_stream.get("nb_frames")
    frame_count = int(frame_count_raw) if frame_count_raw not in (None, "N/A") else None

    avg_frame_rate = str(video_stream["avg_frame_rate"])
    reported_frame_rate = str(video_stream.get("r_frame_rate", avg_frame_rate))
    avg_value = _parse_rate(avg_frame_rate)
    reported_value = _parse_rate(reported_frame_rate)
    return VideoMeta(
        path=str(path),
        duration_s=float(format_data["duration"]),
        width=int(video_stream["width"]),
        height=int(video_stream["height"]),
        fps=avg_value,
        frame_count=frame_count,
        avg_frame_rate=avg_frame_rate,
        reported_frame_rate=reported_frame_rate,
        time_base=(str(video_stream["time_base"]) if video_stream.get("time_base") else None),
        start_time_s=float(format_data.get("start_time") or 0.0),
        variable_frame_rate=abs(avg_value - reported_value) > 1e-6,
    )


def probe_video(path: Path, ffprobe_path: Path) -> VideoMeta:
    if not path.is_file():
        raise MediaProbeError(f"input video does not exist: {path}")
    if not ffprobe_path.is_file():
        raise MediaProbeError(f"ffprobe does not exist: {ffprobe_path}")

    command = [
        str(ffprobe_path),
        "-v",
        "error",
        "-show_streams",
        "-show_format",
        "-of",
        "json",
        str(path),
    ]
    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="strict",
        check=False,
    )
    if completed.returncode != 0:
        raise MediaProbeError(
            f"ffprobe failed with exit code {completed.returncode}; "
            f"stdout={completed.stdout!r}; stderr={completed.stderr!r}"
        )
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise MediaProbeError(f"ffprobe returned invalid JSON: {exc}") from exc
    return parse_ffprobe_payload(path, payload)


def probe_frame_index(path: Path, ffprobe_path: Path) -> list[FrameIndexEntry]:
    if not path.is_file():
        raise MediaProbeError(f"input video does not exist: {path}")
    command = [
        str(ffprobe_path), "-v", "error", "-select_streams", "v:0",
        "-show_frames", "-show_entries",
        "frame=best_effort_timestamp_time,pts_time,pkt_duration_time,key_frame,pict_type",
        "-of", "json", str(path),
    ]
    completed = subprocess.run(
        command, capture_output=True, text=True, encoding="utf-8",
        errors="strict", check=False,
    )
    if completed.returncode != 0:
        raise MediaProbeError(
            f"ffprobe frame index failed with exit code {completed.returncode}; "
            f"stdout={completed.stdout!r}; stderr={completed.stderr!r}"
        )
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise MediaProbeError(f"ffprobe frame index returned invalid JSON: {exc}") from exc
    entries: list[FrameIndexEntry] = []
    for index, frame in enumerate(payload.get("frames") or []):
        pts = frame.get("best_effort_timestamp_time", frame.get("pts_time"))
        if pts is None:
            raise MediaProbeError(f"decoded frame {index} has no timestamp")
        entries.append(
            FrameIndexEntry(
                frame_index=index,
                pts_s=float(pts),
                duration_s=float(frame.get("pkt_duration_time") or 0.0),
                key_frame=bool(int(frame.get("key_frame") or 0)),
                picture_type=frame.get("pict_type"),
            )
        )
    if not entries:
        raise MediaProbeError(f"ffprobe returned no decoded frames: {path}")
    return entries
