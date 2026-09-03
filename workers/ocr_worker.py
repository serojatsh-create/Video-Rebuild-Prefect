from __future__ import annotations

import importlib.util
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PADDLEX_CACHE = PROJECT_ROOT / "runs" / "_paddlex"
PADDLEX_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("PADDLE_PDX_CACHE_HOME", str(PADDLEX_CACHE))


def package_state() -> dict[str, bool]:
    return {
        "paddle": importlib.util.find_spec("paddle") is not None,
        "paddleocr": importlib.util.find_spec("paddleocr") is not None,
        "paddlex": importlib.util.find_spec("paddlex") is not None,
        "cv2": importlib.util.find_spec("cv2") is not None,
    }


def _model_directory_ready(path: Path) -> bool:
    return path.is_dir() and any(item.is_file() for item in path.rglob("*"))


def resolve_device(requested: str, *, cuda_available: bool) -> str:
    normalized = requested.strip().lower()
    if normalized == "auto":
        return "gpu:0" if cuda_available else "cpu"
    if normalized == "gpu":
        normalized = "gpu:0"
    if normalized == "cpu":
        return normalized
    if normalized.startswith("gpu:"):
        if not cuda_available:
            raise RuntimeError(f"requested GPU device is unavailable: {normalized}")
        return normalized
    raise ValueError(f"unsupported OCR device: {requested}")


def probe(payload: dict[str, Any]) -> dict[str, Any]:
    detection_model_dir = Path(str(payload.get("detection_model_dir", "")))
    recognition_model_dir = Path(str(payload.get("recognition_model_dir", "")))
    detection_model_name = str(
        payload.get("detection_model_name", "PP-OCRv6_tiny_det")
    )
    recognition_model_name = str(
        payload.get("recognition_model_name", "PP-OCRv6_tiny_rec")
    )
    packages = package_state()
    detection_ready = _model_directory_ready(detection_model_dir)
    recognition_ready = _model_directory_ready(recognition_model_dir)
    cuda_available = False
    if packages["paddle"]:
        import paddle

        cuda_available = (
            paddle.is_compiled_with_cuda()
            and paddle.device.cuda.device_count() > 0
        )
    requested_device = str(payload.get("device", "auto"))
    try:
        device = resolve_device(requested_device, cuda_available=cuda_available)
        device_error = None
    except (RuntimeError, ValueError) as exc:
        device = None
        device_error = str(exc)
    return {
        "adapter": "paddleocr-v6",
        "available": (
            all(packages.values())
            and detection_ready
            and recognition_ready
            and device_error is None
        ),
        "packages": packages,
        "detection_model_name": detection_model_name,
        "detection_model_dir": str(detection_model_dir),
        "detection_model_ready": detection_ready,
        "recognition_model_name": recognition_model_name,
        "recognition_model_dir": str(recognition_model_dir),
        "recognition_model_ready": recognition_ready,
        "requested_device": requested_device,
        "device": device,
        "device_error": device_error,
    }


def _jsonable(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def normalize_prediction(prediction: Any) -> dict[str, Any]:
    if isinstance(prediction, dict):
        payload = prediction
    else:
        payload = getattr(prediction, "json", None)
        if callable(payload):
            payload = payload()
        if not isinstance(payload, dict):
            raise TypeError("PaddleOCR prediction must be a dict or expose JSON data")
        payload = payload.get("res")
    if not isinstance(payload, dict):
        raise ValueError("PaddleOCR prediction JSON must contain an object result")
    return payload


def recognize(payload: dict[str, Any]) -> dict[str, Any]:
    state = probe(payload)
    if not state["available"]:
        raise RuntimeError(f"OCR dependencies or local models are unavailable: {state}")

    frame_paths = [Path(str(item)) for item in payload.get("frame_paths", [])]
    if not frame_paths:
        raise ValueError("frame_paths must contain at least one image")
    missing = [str(path) for path in frame_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"OCR input images do not exist: {missing}")

    from paddleocr import PaddleOCR

    pipeline = PaddleOCR(
        text_detection_model_name=state["detection_model_name"],
        text_detection_model_dir=state["detection_model_dir"],
        text_recognition_model_name=state["recognition_model_name"],
        text_recognition_model_dir=state["recognition_model_dir"],
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        device=state["device"],
    )
    predictions = pipeline.predict(
        input=[str(path) for path in frame_paths],
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        text_rec_score_thresh=float(payload.get("minimum_score", 0.5)),
    )

    frames: list[dict[str, Any]] = []
    for prediction in predictions:
        normalized = normalize_prediction(prediction)
        texts = list(normalized.get("rec_texts", []))
        scores = _jsonable(normalized.get("rec_scores", []))
        polygons = _jsonable(normalized.get("rec_polys", []))
        items = []
        for index, text in enumerate(texts):
            items.append(
                {
                    "text": str(text),
                    "score": float(scores[index]),
                    "polygon": polygons[index],
                }
            )
        frames.append(
            {
                "input_path": str(normalized.get("input_path", "")),
                "items": items,
            }
        )
    return {
        "adapter": "paddleocr-v6",
        "device": state["device"],
        "frames": frames,
    }


def main() -> int:
    try:
        payload = json.loads(sys.stdin.read())
        if not isinstance(payload, dict):
            raise ValueError("worker input must be a JSON object")
        action = payload.get("action", "recognize")
        if action == "probe":
            result = probe(payload)
        elif action == "recognize":
            result = recognize(payload)
        else:
            raise ValueError(f"unsupported action: {action}")
        print(json.dumps(result, ensure_ascii=False))
        return 0
    except Exception as exc:
        error = {
            "adapter": "paddleocr-v6",
            "error_type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
        print(json.dumps(error, ensure_ascii=False), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
