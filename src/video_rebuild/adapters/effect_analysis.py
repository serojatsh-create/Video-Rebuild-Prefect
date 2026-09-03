from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class EffectAnalysisError(RuntimeError):
    pass


def unwrap_effect_result(payload: dict[str, Any]) -> dict[str, Any]:
    if payload.get("ok") is not True:
        error = payload.get("error", {})
        if isinstance(error, dict):
            code = error.get("code", "UNKNOWN")
            message = error.get("message", "Effect Analysis reported failure")
            raise EffectAnalysisError(f"{code}: {message}")
        raise EffectAnalysisError(f"Effect Analysis reported failure: {error!r}")
    result = payload.get("result")
    if not isinstance(result, dict):
        raise EffectAnalysisError("Effect Analysis result must be a JSON object")
    return result


@dataclass(frozen=True)
class EffectAnalysisAdapter:
    python_path: Path
    project_path: Path

    def _run(self, arguments: list[str]) -> dict[str, Any]:
        if not self.python_path.is_file():
            raise EffectAnalysisError(f"Effect Analysis Python not found: {self.python_path}")
        if not self.project_path.is_dir():
            raise EffectAnalysisError(f"Effect Analysis project not found: {self.project_path}")

        environment = os.environ.copy()
        environment["PYTHONNOUSERSITE"] = "1"
        completed = subprocess.run(
            [
                str(self.python_path),
                "-m",
                "effect_analysis.cli",
                *arguments,
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="strict",
            cwd=str(self.project_path),
            env=environment,
            check=False,
        )
        if completed.returncode != 0:
            raise EffectAnalysisError(
                f"Effect Analysis exited with {completed.returncode}; "
                f"stdout={completed.stdout!r}; stderr={completed.stderr!r}"
            )
        try:
            payload = json.loads(completed.stdout)
        except json.JSONDecodeError as exc:
            raise EffectAnalysisError(
                f"Effect Analysis returned invalid JSON; "
                f"stdout={completed.stdout!r}; stderr={completed.stderr!r}"
            ) from exc
        if not isinstance(payload, dict):
            raise EffectAnalysisError("Effect Analysis output must be a JSON object")
        return unwrap_effect_result(payload)

    def inspect_video(self, video_path: Path) -> dict[str, Any]:
        if not video_path.is_absolute():
            raise EffectAnalysisError("video path must be absolute")
        return self._run(["inspect-video", str(video_path)])

    def extract_frame(
        self,
        video_path: Path,
        time_s: float,
        cache_directory: Path,
    ) -> dict[str, Any]:
        if time_s < 0:
            raise ValueError("time_s must be non-negative")
        if not video_path.is_absolute() or not cache_directory.is_absolute():
            raise EffectAnalysisError("video and cache paths must be absolute")
        return self._run(
            [
                "extract-frame",
                str(video_path),
                "--time",
                str(time_s),
                "--cache-dir",
                str(cache_directory),
            ]
        )

