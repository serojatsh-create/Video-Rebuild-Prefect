from __future__ import annotations

import json
import statistics
import sys
import traceback
from pathlib import Path
from typing import Any


def _threshold(scores: list[float]) -> float:
    if not scores:
        return 0.0005
    ordered = sorted(scores)
    low = ordered[: min(len(ordered), max(2, len(ordered) // 2))]
    baseline = statistics.median(low)
    mad = statistics.median(abs(value - baseline) for value in low)
    return max(0.0005, baseline + 6.0 * mad)


def analyze(payload: dict[str, Any]) -> dict[str, Any]:
    import cv2

    video_path = Path(str(payload.get("video_path", "")))
    if not video_path.is_file():
        raise FileNotFoundError(f"input video does not exist: {video_path}")
    start = int(payload.get("start_frame", -1))
    end = int(payload.get("end_frame", -1))
    if start < 0 or end <= start:
        raise ValueError("end_frame must be greater than start_frame")
    resize_width = int(payload.get("resize_width", 320))
    if resize_width <= 0:
        raise ValueError("resize_width must be positive")

    capture = cv2.VideoCapture(str(video_path))
    try:
        if not capture.isOpened():
            raise RuntimeError(f"OpenCV could not open video: {video_path}")
        capture.set(cv2.CAP_PROP_POS_FRAMES, start)
        previous = None
        frame_scores: list[dict[str, float | int]] = []
        for frame_index in range(start, end):
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError(f"video decode stopped at frame {frame_index}")
            height, width = frame.shape[:2]
            resized_height = max(1, round(height * resize_width / width))
            frame = cv2.resize(frame, (resize_width, resized_height))
            gray = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (5, 5), 0)
            if previous is not None:
                score = float(cv2.absdiff(previous, gray).mean() / 255.0)
                frame_scores.append({"frame_index": frame_index, "score": score})
            previous = gray
    finally:
        capture.release()

    threshold = _threshold([float(item["score"]) for item in frame_scores])
    active = [item for item in frame_scores if float(item["score"]) > threshold]
    windows: list[dict[str, float | int]] = []
    for item in active:
        frame_index = int(item["frame_index"])
        score = float(item["score"])
        if not windows or frame_index > int(windows[-1]["end_frame"]) + 1:
            windows.append({
                "start_frame": frame_index,
                "end_frame": frame_index,
                "peak_frame": frame_index,
                "peak_score": score,
            })
        else:
            windows[-1]["end_frame"] = frame_index
            if score > float(windows[-1]["peak_score"]):
                windows[-1]["peak_frame"] = frame_index
                windows[-1]["peak_score"] = score
    return {
        "adapter": "opencv-frame-difference",
        "start_frame": start,
        "end_frame": end,
        "resize_width": resize_width,
        "threshold": threshold,
        "frame_scores": frame_scores,
        "windows": windows,
    }


def main() -> int:
    try:
        payload = json.loads(sys.stdin.read())
        if not isinstance(payload, dict):
            raise ValueError("worker input must be a JSON object")
        print(json.dumps(analyze(payload), ensure_ascii=False))
        return 0
    except Exception as exc:
        print(json.dumps({
            "adapter": "opencv-frame-difference",
            "error_type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }, ensure_ascii=False), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
