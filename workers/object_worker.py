from __future__ import annotations

import importlib.util
import json
import sys
import traceback
from pathlib import Path
from typing import Any


def _device(requested: str) -> str:
    import torch

    normalized = requested.strip().lower()
    if normalized == "auto":
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    if normalized == "cuda":
        normalized = "cuda:0"
    if normalized == "cpu":
        return normalized
    if normalized.startswith("cuda:") and torch.cuda.is_available():
        return normalized
    raise ValueError(f"unsupported or unavailable object device: {requested}")


def prediction_options(payload: dict[str, Any], device: str) -> dict[str, Any]:
    return {
        "conf": float(payload.get("confidence", 0.25)),
        "imgsz": int(payload.get("image_size", 640)),
        "device": device,
        "verbose": False,
        "stream": True,
    }


def detect(payload: dict[str, Any]) -> dict[str, Any]:
    if importlib.util.find_spec("ultralytics") is None:
        raise RuntimeError("ultralytics is unavailable")
    from ultralytics import YOLO

    model_path = Path(str(payload.get("model_path", "")))
    if not model_path.is_file():
        raise FileNotFoundError(f"object model does not exist: {model_path}")
    frame_paths = [Path(str(item)) for item in payload.get("frame_paths", [])]
    if not frame_paths:
        raise ValueError("frame_paths must contain at least one image")
    missing = [str(path) for path in frame_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"object input images do not exist: {missing}")
    device = _device(str(payload.get("device", "auto")))
    model = YOLO(str(model_path))
    results = model.predict(
        source=[str(path) for path in frame_paths],
        **prediction_options(payload, device),
    )
    frames: list[dict[str, Any]] = []
    for input_path, result in zip(frame_paths, results, strict=True):
        items: list[dict[str, Any]] = []
        if result.boxes is not None:
            for box, score, class_id in zip(
                result.boxes.xyxy.tolist(),
                result.boxes.conf.tolist(),
                result.boxes.cls.tolist(),
                strict=True,
            ):
                numeric_id = int(class_id)
                items.append({
                    "class_id": numeric_id,
                    "label": str(result.names[numeric_id]),
                    "score": float(score),
                    "bbox": [float(value) for value in box],
                })
        frames.append({"input_path": str(input_path), "items": items})
    return {
        "adapter": "yolo11n-coco",
        "device": device,
        "model_path": str(model_path),
        "frames": frames,
    }


def main() -> int:
    try:
        payload = json.loads(sys.stdin.read())
        if not isinstance(payload, dict):
            raise ValueError("worker input must be a JSON object")
        print(json.dumps(detect(payload), ensure_ascii=False))
        return 0
    except Exception as exc:
        print(json.dumps({
            "adapter": "yolo11n-coco",
            "error_type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }, ensure_ascii=False), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
