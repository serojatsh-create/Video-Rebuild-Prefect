from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path

from video_rebuild.models import (
    AnalysisFrame,
    AnalysisPacket,
    AnalysisStatus,
    TimeSpan,
    VideoMeta,
    VisualEvidencePurpose,
    VisualEvidenceSequence,
)
from video_rebuild.visual_analysis import (
    CodexVisualAnalyzer,
    HumanReviewStatus,
    VisualAnalysisError,
)


def _analysis_payload(segment_id: str) -> dict[str, object]:
    return {
        "segment_id": segment_id,
        "span": {"start_s": 1.0, "end_s": 2.0, "start_frame": 30, "end_frame": 60},
        "status": "complete",
        "visual_summary": "A flat title enters, holds, and exits.",
        "evidence": [],
        "uncertainties": [],
        "confidence": 0.9,
        "analyzer": "model-supplied-value",
        "details": {},
        "reconstruction": {
            "schema_version": "1.0",
            "content_category": "motion_title",
            "design_style": ["flat"],
            "composition": "Centered title on a solid field.",
            "layers": [
                {
                    "layer_id": "title",
                    "name": "Title",
                    "kind": "text",
                    "z_index": 1,
                    "content": "Example",
                    "styling": {"color": "#ffffff"},
                    "states": [],
                    "evidence_frame_indices": [30, 45, 59],
                    "confidence": 0.9,
                    "uncertainties": [],
                }
            ],
            "phases": [
                {
                    "phase_id": "entrance",
                    "kind": "entrance",
                    "start_frame": 30,
                    "end_frame": 36,
                    "start_s": 1.0,
                    "end_s": 1.2,
                    "target_layer_ids": ["title"],
                    "effect": "fade",
                    "parameter_changes": {"opacity": [0.0, 1.0]},
                    "evidence_sequence_ids": ["entrance"],
                    "evidence_frame_indices": [30, 36],
                    "confidence": 0.9,
                    "uncertainties": [],
                },
                {
                    "phase_id": "exit",
                    "kind": "exit",
                    "start_frame": 54,
                    "end_frame": 59,
                    "start_s": 1.8,
                    "end_s": 1.966667,
                    "target_layer_ids": ["title"],
                    "effect": "fade",
                    "parameter_changes": {"opacity": [1.0, 0.0]},
                    "evidence_sequence_ids": ["exit"],
                    "evidence_frame_indices": [54, 59],
                    "confidence": 0.8,
                    "uncertainties": [],
                },
            ],
            "materials": [],
            "implementation": {
                "primary_tool": "html",
                "layer_build_order": ["title"],
                "technical_steps": ["Animate opacity by explicit frame number."],
                "required_effects": ["fade"],
                "test_points": [
                    {
                        "frame_index": 45,
                        "time_s": 1.5,
                        "expected_visible_layers": ["title"],
                        "checks": {"opacity": 1.0},
                    }
                ],
                "limitations": [],
            },
            "unresolved_questions": [],
        },
        "source_packet_path": "will-be-normalized-by-adapter",
    }


def _write_packet(root: Path) -> Path:
    images = root / "images"
    images.mkdir()
    contact_sheet = images / "contact-sheet.jpg"
    entrance_sheet = images / "entrance.jpg"
    exit_sheet = images / "exit.jpg"
    for path in (contact_sheet, entrance_sheet, exit_sheet):
        path.write_bytes(b"test-image")
    frame_a = AnalysisFrame(
        frame_index=30, pts_s=1.0, path=str(images / "frame-30.png"), roles=["coarse"]
    )
    frame_b = AnalysisFrame(
        frame_index=36, pts_s=1.2, path=str(images / "frame-36.png"), roles=["coarse"]
    )
    for frame in (frame_a, frame_b):
        Path(frame.path).write_bytes(b"test-frame")
    packet = AnalysisPacket(
        segment_id="candidate-0001",
        span=TimeSpan(start_s=1.0, end_s=2.0, start_frame=30, end_frame=60),
        context_span=TimeSpan(start_s=0.9, end_s=2.1, start_frame=27, end_frame=63),
        video=VideoMeta(
            path="reference.mp4", duration_s=3.0, width=1280, height=720, fps=30.0,
            frame_count=90,
        ),
        frames=[frame_a, frame_b],
        contact_sheet_path=str(contact_sheet),
        ocr=[],
        objects=[],
        visual_sequences=[
            VisualEvidenceSequence(
                sequence_id="entrance",
                purpose=VisualEvidencePurpose.ENTRANCE,
                window=TimeSpan(start_s=1.0, end_s=1.3, start_frame=30, end_frame=39),
                sampling_fps=10.0,
                frames=[frame_a, frame_b],
                contact_sheet_path=str(entrance_sheet),
            ),
            VisualEvidenceSequence(
                sequence_id="exit",
                purpose=VisualEvidencePurpose.EXIT,
                window=TimeSpan(start_s=1.0, end_s=1.3, start_frame=30, end_frame=39),
                sampling_fps=10.0,
                frames=[frame_a, frame_b],
                contact_sheet_path=str(exit_sheet),
            ),
        ],
    )
    packet_path = root / "analysis_packet.json"
    packet_path.write_text(packet.model_dump_json(indent=2), encoding="utf-8")
    return packet_path


class CodexVisualAnalyzerTests(unittest.TestCase):
    def test_codex_reads_evidence_and_writes_validated_analysis(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            packet_path = _write_packet(root)
            schema_path = root / "SegmentAnalysis.schema.json"
            schema_path.write_text("{}", encoding="utf-8")
            codex_path = root / "codex.exe"
            codex_path.write_bytes(b"")
            output_path = root / "segment_analysis.json"
            observed: dict[str, object] = {}

            def fake_runner(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
                observed["command"] = command
                observed["input"] = kwargs["input"]
                result_path = Path(command[command.index("--output-last-message") + 1])
                result_path.write_text(
                    json.dumps(_analysis_payload("candidate-0001")), encoding="utf-8"
                )
                return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

            analyzer = CodexVisualAnalyzer(
                codex_executable=codex_path,
                schema_path=schema_path,
                project_root=root,
                runner=fake_runner,
            )
            result = analyzer.analyze(
                packet_path=packet_path,
                output_path=output_path,
                require_human_review=True,
            )

            command = observed["command"]
            self.assertIn("read-only", command)
            self.assertIn("--ephemeral", command)
            self.assertEqual(command.count("--image"), 3)
            self.assertIn("candidate-0001", str(observed["input"]))
            self.assertEqual(result.analysis.status, AnalysisStatus.COMPLETE)
            self.assertEqual(result.analysis.analyzer, "codex-cli")
            self.assertEqual(result.review_status, HumanReviewStatus.PENDING)
            self.assertTrue(result.human_review_required)
            self.assertTrue(output_path.is_file())
            persisted = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(persisted["source_packet_path"], str(packet_path.resolve()))

    def test_human_review_is_optional_and_never_replaces_codex(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            packet_path = _write_packet(root)
            schema_path = root / "schema.json"
            schema_path.write_text("{}", encoding="utf-8")
            codex_path = root / "codex.exe"
            codex_path.write_bytes(b"")

            def fake_runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
                result_path = Path(command[command.index("--output-last-message") + 1])
                result_path.write_text(
                    json.dumps(_analysis_payload("candidate-0001")), encoding="utf-8"
                )
                return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

            result = CodexVisualAnalyzer(
                codex_executable=codex_path,
                schema_path=schema_path,
                project_root=root,
                runner=fake_runner,
            ).analyze(
                packet_path=packet_path,
                output_path=root / "analysis.json",
                require_human_review=False,
            )

            self.assertFalse(result.human_review_required)
            self.assertEqual(result.review_status, HumanReviewStatus.NOT_REQUESTED)
            self.assertEqual(result.analysis.analyzer, "codex-cli")

    def test_nonzero_codex_exit_fails_loudly(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            packet_path = _write_packet(root)
            schema_path = root / "schema.json"
            schema_path.write_text("{}", encoding="utf-8")
            codex_path = root / "codex.exe"
            codex_path.write_bytes(b"")

            def fake_runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
                return subprocess.CompletedProcess(command, 9, stdout="partial", stderr="failed")

            analyzer = CodexVisualAnalyzer(
                codex_executable=codex_path,
                schema_path=schema_path,
                project_root=root,
                runner=fake_runner,
            )
            with self.assertRaisesRegex(VisualAnalysisError, "exit code 9"):
                analyzer.analyze(
                    packet_path=packet_path,
                    output_path=root / "analysis.json",
                )


if __name__ == "__main__":
    unittest.main()
