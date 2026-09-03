from __future__ import annotations

import importlib.util
import json
import sys
import traceback
from pathlib import Path
from typing import Any


def probe() -> dict[str, Any]:
    packages = {
        "scenedetect": importlib.util.find_spec("scenedetect") is not None,
        "cv2": importlib.util.find_spec("cv2") is not None,
    }
    return {
        "adapter": "pyscenedetect-content-cpu",
        "available": all(packages.values()),
        "packages": packages,
    }


def detect(payload: dict[str, Any]) -> dict[str, Any]:
    state = probe()
    if not state["available"]:
        raise RuntimeError(f"scene detection dependencies are unavailable: {state}")

    video_path = Path(str(payload.get("video_path", "")))
    if not video_path.is_file():
        raise FileNotFoundError(f"input video does not exist: {video_path}")

    threshold = float(payload.get("threshold", 27.0))
    minimum_scene_length_frames = int(
        payload.get("minimum_scene_length_frames", 15)
    )
    if threshold <= 0:
        raise ValueError("threshold must be positive")
    if minimum_scene_length_frames <= 0:
        raise ValueError("minimum_scene_length_frames must be positive")

    from scenedetect import SceneManager, open_video
    from scenedetect.detectors import ContentDetector

    video = open_video(str(video_path), backend="opencv")
    manager = SceneManager()
    manager.add_detector(
        ContentDetector(
            threshold=threshold,
            min_scene_len=minimum_scene_length_frames,
        )
    )
    frames_processed = manager.detect_scenes(video=video, show_progress=False)
    scene_list = manager.get_scene_list(start_in_scene=True)
    scenes = [
        {
            "start_s": float(start.seconds),
            "end_s": float(end.seconds),
            "start_frame": int(start.frame_num),
            "end_frame": int(end.frame_num),
        }
        for start, end in scene_list
        if end.seconds > start.seconds
    ]
    return {
        "adapter": "pyscenedetect-content-cpu",
        "video_path": str(video_path),
        "threshold": threshold,
        "minimum_scene_length_frames": minimum_scene_length_frames,
        "frames_processed": int(frames_processed),
        "scenes": scenes,
    }


def main() -> int:
    try:
        payload = json.loads(sys.stdin.read())
        if not isinstance(payload, dict):
            raise ValueError("worker input must be a JSON object")
        action = payload.get("action", "detect")
        if action == "probe":
            result = probe()
        elif action == "detect":
            result = detect(payload)
        else:
            raise ValueError(f"unsupported action: {action}")
        print(json.dumps(result, ensure_ascii=False))
        return 0
    except Exception as exc:
        error = {
            "adapter": "pyscenedetect-content-cpu",
            "error_type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
        print(json.dumps(error, ensure_ascii=False), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
