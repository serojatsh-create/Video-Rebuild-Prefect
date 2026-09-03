from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class DependencyUnavailableError(RuntimeError):
    pass


class JsonAdapterError(RuntimeError):
    pass


@dataclass(frozen=True)
class JsonSubprocessAdapter:
    name: str
    python_path: Path
    worker_path: Path

    def run(self, payload: dict[str, Any]) -> dict[str, Any]:
        if not self.python_path.is_file():
            raise DependencyUnavailableError(
                f"{self.name} adapter python was not found: {self.python_path}"
            )
        if not self.worker_path.is_file():
            raise DependencyUnavailableError(
                f"{self.name} adapter worker was not found: {self.worker_path}"
            )

        environment = os.environ.copy()
        environment["PYTHONNOUSERSITE"] = "1"
        environment["PYTHONIOENCODING"] = "utf-8"
        completed = subprocess.run(
            [str(self.python_path), str(self.worker_path)],
            input=json.dumps(payload, ensure_ascii=False),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="strict",
            cwd=str(self.worker_path.parent),
            env=environment,
            check=False,
        )
        if completed.returncode != 0:
            raise JsonAdapterError(
                f"{self.name} adapter failed with exit code {completed.returncode}; "
                f"stdout={completed.stdout!r}; stderr={completed.stderr!r}"
            )
        try:
            result = json.loads(completed.stdout)
        except json.JSONDecodeError as exc:
            raise JsonAdapterError(
                f"{self.name} adapter returned invalid JSON; "
                f"stdout={completed.stdout!r}; stderr={completed.stderr!r}"
            ) from exc
        if not isinstance(result, dict):
            raise JsonAdapterError(
                f"{self.name} adapter returned JSON {type(result).__name__}, expected object"
            )
        return result
