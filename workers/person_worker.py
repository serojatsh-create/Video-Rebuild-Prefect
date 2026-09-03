from __future__ import annotations

import importlib.util
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
YOLO_CONFIG_ROOT = PROJECT_ROOT / "runs" / "_ultralytics"
YOLO_CONFIG_ROOT.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("YOLO_CONFIG_DIR", str(YOLO_CONFIG_ROOT))
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))


def package_state() -> dict[str, bool]:
    return {
        "ultralytics": importlib.util.find_spec("ultralytics") is not None,
        "torch": importlib.util.find_spec("torch") is not None,
        "cv2": importlib.util.find_spec("cv2") is not None,
    }


def resolve_device(requested: str, *, cuda_available: bool) -> str:
    normalized = requested.strip().lower()
    if normalized == "auto":
        return "cuda:0" if cuda_available else "cpu"
    if normalized == "cuda":
        normalized = "cuda:0"
    if normalized == "cpu":
        return normalized
    if normalized.startswith("cuda:"):
        if not cuda_available:
            raise RuntimeError(f"requested CUDA device is unavailable: {normalized}")
        return normalized
    raise ValueError(f"unsupported person device: {requested}")


def probe(payload: dict[str, Any]) -> dict[str, Any]:
    model_path = Path(str(payload.get("model_path", "")))
    packages = package_state()
    model_exists = model_path.is_file()
    cuda_available = False
    if packages["torch"]:
        import torch

        cuda_available = torch.cuda.is_available()
    requested_device = str(payload.get("device", "auto"))
    try:
        device = resolve_device(requested_device, cuda_available=cuda_available)
        device_error = None
    except (RuntimeError, ValueError) as exc:
        device = None
        device_error = str(exc)
    return {
        "adapter": "person-ultralytics",
        "available": all(packages.values()) and model_exists and device_error is None,
        "packages": packages,
        "model_path": str(model_path),
        "model_exists": model_exists,
        "requested_device": requested_device,
        "device": device,
        "device_error": device_error,
    }


def detect(payload: dict[str, Any]) -> dict[str, Any]:
    state = probe(payload)
    if not state["available"]:
        raise RuntimeError(f"person detector dependencies are unavailable: {state}")

    video_path = Path(str(payload["video_path"]))
    model_path = Path(str(payload["model_path"]))
    if not video_path.is_file():
        raise FileNotFoundError(f"input video does not exist: {video_path}")

    import cv2
    from ultralytics import YOLO

    capture = cv2.VideoCapture(str(video_path))
    try:
        if not capture.isOpened():
            raise RuntimeError(f"OpenCV could not open video: {video_path}")
        fps = float(capture.get(cv2.CAP_PROP_FPS))
    finally:
        capture.release()
    if fps <= 0:
        raise RuntimeError(f"video reported invalid fps: {fps}")

    model = YOLO(str(model_path))
    presence: list[bool] = []
    results = model.predict(
        source=str(video_path),
        stream=True,
        classes=[0],
        conf=float(payload.get("confidence", 0.35)),
        imgsz=int(payload.get("image_size", 640)),
        device=state["device"],
        verbose=False,
    )
    for result in results:
        presence.append(result.boxes is not None and len(result.boxes) > 0)

    from video_rebuild.person_frames import frames_to_span_dicts

    spans = frames_to_span_dicts(
        presence,
        fps=fps,
        gap_tolerance_frames=int(payload.get("gap_tolerance_frames", 2)),
    )
    return {
        "adapter": "person-ultralytics",
        "video_path": str(video_path),
        "fps": fps,
        "frames_processed": len(presence),
        "person_segments": spans,
    }


def main() -> int:
    try:
        payload = json.loads(sys.stdin.read())
        if not isinstance(payload, dict):
            raise ValueError("worker input must be a JSON object")
        action = payload.get("action", "detect")
        if action == "probe":
            result = probe(payload)
        elif action == "detect":
            result = detect(payload)
        else:
            raise ValueError(f"unsupported action: {action}")
        print(json.dumps(result, ensure_ascii=False))
        return 0
    except Exception as exc:
        error = {
            "adapter": "person-ultralytics",
            "error_type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
        print(json.dumps(error, ensure_ascii=False), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
