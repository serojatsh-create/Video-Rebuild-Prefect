from __future__ import annotations

import unittest

from pydantic import ValidationError

from video_rebuild.models import (
    AnalysisFrame,
    AnalysisStatus,
    AnimationEffect,
    AnimationPhaseKind,
    AnimationPhaseSpec,
    ImplementationSpec,
    LayerKind,
    ReconstructionTool,
    SegmentAnalysis,
    TimeSpan,
    VisualEvidencePurpose,
    VisualEvidenceSequence,
    VisualLayerSpec,
    VisualReconstructionSpec,
    ValidationPoint,
)


def reconstruction_spec() -> VisualReconstructionSpec:
    return VisualReconstructionSpec(
        content_category="motion_title",
        design_style=["dark", "metallic"],
        composition="Centered title over a dark background.",
        layers=[
            VisualLayerSpec(
                layer_id="title",
                name="Main title",
                kind=LayerKind.TEXT,
                z_index=1,
                content="KIMI K3",
                evidence_frame_indices=[100, 106],
                confidence=0.9,
            )
        ],
        phases=[
            AnimationPhaseSpec(
                phase_id="entrance-title",
                kind=AnimationPhaseKind.ENTRANCE,
                start_frame=100,
                end_frame=106,
                start_s=3.333,
                end_s=3.533,
                target_layer_ids=["title"],
                effect=AnimationEffect.SCALE,
                parameter_changes={"scale": [0.7, 1.0], "opacity": [0.0, 1.0]},
                evidence_sequence_ids=["entrance"],
                evidence_frame_indices=[100, 103, 106],
                confidence=0.8,
            ),
            AnimationPhaseSpec(
                phase_id="exit-title",
                kind=AnimationPhaseKind.EXIT,
                start_frame=124,
                end_frame=130,
                start_s=4.133,
                end_s=4.333,
                target_layer_ids=["title"],
                effect=AnimationEffect.FADE,
                parameter_changes={"opacity": [1.0, 0.0]},
                evidence_sequence_ids=["exit"],
                evidence_frame_indices=[124, 127, 130],
                confidence=0.7,
            ),
        ],
        materials=[],
        implementation=ImplementationSpec(
            primary_tool=ReconstructionTool.SVG,
            layer_build_order=["title"],
            technical_steps=["Animate title scale and opacity."],
            required_effects=["drop-shadow"],
            test_points=[
                ValidationPoint(
                    frame_index=106,
                    time_s=3.533,
                    expected_visible_layers=["title"],
                    checks={"opacity": 1.0},
                )
            ],
        ),
    )


class VisualEvidenceSequenceTests(unittest.TestCase):
    def test_sequence_requires_ordered_multi_frame_evidence(self) -> None:
        with self.assertRaises(ValidationError):
            VisualEvidenceSequence(
                sequence_id="entrance",
                purpose=VisualEvidencePurpose.ENTRANCE,
                window=TimeSpan(start_s=1.0, end_s=1.2),
                sampling_fps=15,
                contact_sheet_path="entrance.jpg",
                frames=[
                    AnalysisFrame(
                        frame_index=30,
                        pts_s=1.0,
                        path="frame.png",
                        roles=["entrance_sequence"],
                    )
                ],
            )

        sequence = VisualEvidenceSequence(
            sequence_id="entrance",
            purpose=VisualEvidencePurpose.ENTRANCE,
            window=TimeSpan(start_s=1.0, end_s=1.2),
            sampling_fps=15,
            contact_sheet_path="entrance.jpg",
            frames=[
                AnalysisFrame(
                    frame_index=30,
                    pts_s=1.0,
                    path="a.png",
                    roles=["entrance_sequence"],
                ),
                AnalysisFrame(
                    frame_index=33,
                    pts_s=1.1,
                    path="b.png",
                    roles=["entrance_sequence"],
                ),
            ],
        )
        self.assertEqual([frame.frame_index for frame in sequence.frames], [30, 33])


class ReconstructionSchemaTests(unittest.TestCase):
    def test_complete_analysis_requires_typed_reconstruction(self) -> None:
        with self.assertRaisesRegex(ValidationError, "reconstruction"):
            SegmentAnalysis(
                segment_id="candidate-0001",
                span=TimeSpan(start_s=3.0, end_s=4.0),
                status=AnalysisStatus.COMPLETE,
                visual_summary="Title animation.",
                confidence=0.8,
                analyzer="codex-visual",
            )

        analysis = SegmentAnalysis(
            segment_id="candidate-0001",
            span=TimeSpan(start_s=3.0, end_s=4.0),
            status=AnalysisStatus.COMPLETE,
            visual_summary="Title animation.",
            confidence=0.8,
            analyzer="codex-visual",
            reconstruction=reconstruction_spec(),
            source_packet_path="analysis_packet.json",
        )
        self.assertEqual(
            analysis.reconstruction.phases[0].evidence_frame_indices,
            [100, 103, 106],
        )

    def test_animation_phase_must_reference_known_layer(self) -> None:
        payload = reconstruction_spec().model_dump()
        payload["phases"][0]["target_layer_ids"] = ["missing-layer"]
        with self.assertRaisesRegex(ValidationError, "unknown layer"):
            VisualReconstructionSpec.model_validate(payload)


if __name__ == "__main__":
    unittest.main()
