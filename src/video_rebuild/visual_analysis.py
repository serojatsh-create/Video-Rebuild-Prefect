from __future__ import annotations

import json
import subprocess
from collections.abc import Callable
from enum import StrEnum
from pathlib import Path
from typing import Any
from uuid import uuid4

from pydantic import BaseModel

from .models import AnalysisPacket, SegmentAnalysis, TimeSpan


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class VisualAnalysisError(RuntimeError):
    pass


class HumanReviewStatus(StrEnum):
    PENDING = "pending"
    NOT_REQUESTED = "not_requested"


class VisualAnalysisRunResult(BaseModel):
    analysis: SegmentAnalysis
    human_review_required: bool
    review_status: HumanReviewStatus


Runner = Callable[..., subprocess.CompletedProcess[str]]


def _resolve_path(path: Path, base: Path) -> Path:
    return (path if path.is_absolute() else base / path).resolve()


class CodexVisualAnalyzer:
    def __init__(
        self,
        *,
        codex_executable: str | Path = "codex",
        schema_path: Path | None = None,
        project_root: Path = PROJECT_ROOT,
        runner: Runner = subprocess.run,
        timeout_seconds: float = 600.0,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self.codex_executable = str(codex_executable)
        self.project_root = project_root.resolve()
        self.schema_path = _resolve_path(
            schema_path or Path("schemas/json/SegmentAnalysis.schema.json"),
            self.project_root,
        )
        self.runner = runner
        self.timeout_seconds = timeout_seconds

    def analyze(
        self,
        *,
        packet_path: Path,
        output_path: Path,
        require_human_review: bool = False,
    ) -> VisualAnalysisRunResult:
        packet_path = _resolve_path(packet_path, self.project_root)
        output_path = _resolve_path(output_path, self.project_root)
        if not packet_path.is_file():
            raise VisualAnalysisError(f"analysis packet does not exist: {packet_path}")
        if not self.schema_path.is_file():
            raise VisualAnalysisError(f"analysis schema does not exist: {self.schema_path}")

        try:
            packet = AnalysisPacket.model_validate_json(
                packet_path.read_text(encoding="utf-8")
            )
        except (OSError, ValueError) as error:
            raise VisualAnalysisError(
                f"analysis packet is invalid: {packet_path}: {error}"
            ) from error

        image_paths = self._evidence_images(packet, packet_path.parent)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        raw_output_path = output_path.with_name(
            f".{output_path.stem}.{uuid4().hex}.codex-output.json"
        )
        command = [
            self.codex_executable,
            "exec",
            "--sandbox",
            "read-only",
            "--ephemeral",
            "--skip-git-repo-check",
            "--cd",
            str(self.project_root),
            "--output-schema",
            str(self.schema_path),
            "--output-last-message",
            str(raw_output_path),
        ]
        for image_path in image_paths:
            command.extend(["--image", str(image_path)])
        command.append("-")

        prompt = self._build_prompt(packet, packet_path)
        try:
            completed = self.runner(
                command,
                input=prompt,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                check=False,
                cwd=str(self.project_root),
                timeout=self.timeout_seconds,
            )
            if completed.returncode != 0:
                raise VisualAnalysisError(
                    "Codex visual analysis failed with exit code "
                    f"{completed.returncode}; stdout={completed.stdout!r}; "
                    f"stderr={completed.stderr!r}"
                )
            if not raw_output_path.is_file():
                raise VisualAnalysisError(
                    f"Codex produced no result file: {raw_output_path}"
                )
            analysis = self._validate_result(raw_output_path, packet, packet_path)
            output_path.write_text(
                analysis.model_dump_json(indent=2) + "\n", encoding="utf-8"
            )
        except subprocess.TimeoutExpired as error:
            raise VisualAnalysisError(
                f"Codex visual analysis timed out after {self.timeout_seconds:g} seconds"
            ) from error
        except OSError as error:
            raise VisualAnalysisError(f"unable to run Codex: {error}") from error
        finally:
            raw_output_path.unlink(missing_ok=True)

        review_status = (
            HumanReviewStatus.PENDING
            if require_human_review
            else HumanReviewStatus.NOT_REQUESTED
        )
        return VisualAnalysisRunResult(
            analysis=analysis,
            human_review_required=require_human_review,
            review_status=review_status,
        )

    @staticmethod
    def _evidence_images(packet: AnalysisPacket, packet_root: Path) -> list[Path]:
        raw_paths = [packet.contact_sheet_path]
        raw_paths.extend(
            sequence.contact_sheet_path for sequence in packet.visual_sequences
        )
        images: list[Path] = []
        seen: set[Path] = set()
        for raw_path in raw_paths:
            image_path = _resolve_path(Path(raw_path), packet_root)
            if image_path in seen:
                continue
            if not image_path.is_file():
                raise VisualAnalysisError(f"visual evidence image does not exist: {image_path}")
            seen.add(image_path)
            images.append(image_path)
        return images

    @staticmethod
    def _build_prompt(packet: AnalysisPacket, packet_path: Path) -> str:
        return (
            "Analyze the attached contact sheets for the video segment described below. "
            "You are the visual analyst: infer the visible composition, layers, styling, "
            "entrance/internal/exit motion, timing, materials, implementation steps, test "
            "points, uncertainty, and confidence. Use only supplied evidence. Return one "
            "SegmentAnalysis JSON object matching the supplied schema. Do not edit files or "
            "ask a human to author JSON. Human review, when requested by the caller, happens "
            "only after your analysis. Preserve the packet segment_id and span exactly.\n\n"
            f"Packet path: {packet_path}\n"
            f"Analysis packet:\n{packet.model_dump_json(indent=2)}\n"
        )

    @staticmethod
    def _validate_result(
        raw_output_path: Path,
        packet: AnalysisPacket,
        packet_path: Path,
    ) -> SegmentAnalysis:
        try:
            payload: Any = json.loads(raw_output_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise VisualAnalysisError(f"Codex returned invalid JSON: {error}") from error
        if not isinstance(payload, dict):
            raise VisualAnalysisError("Codex result must be a JSON object")
        if payload.get("segment_id") != packet.segment_id:
            raise VisualAnalysisError("Codex result segment_id does not match the analysis packet")
        try:
            result_span = TimeSpan.model_validate(payload.get("span"))
        except ValueError as error:
            raise VisualAnalysisError(f"Codex result span is invalid: {error}") from error
        if result_span.model_dump(mode="json") != packet.span.model_dump(mode="json"):
            raise VisualAnalysisError("Codex result span does not match the analysis packet")
        payload["analyzer"] = "codex-cli"
        payload["source_packet_path"] = str(packet_path)
        try:
            return SegmentAnalysis.model_validate(payload)
        except ValueError as error:
            raise VisualAnalysisError(f"Codex result failed validation: {error}") from error
