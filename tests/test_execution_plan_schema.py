from __future__ import annotations

import unittest

from pydantic import ValidationError

from video_rebuild.models import ExecutionPlan


def valid_plan_payload() -> dict:
    return {
        "segment_id": "candidate-test",
        "strategy": "html_svg",
        "router_version": "flat-html-svg-v1",
        "canvas": {
            "width": 1280,
            "height": 720,
            "fps": 30,
            "duration_frames": 30,
            "duration_s": 1.0,
            "background_color": "#001122",
        },
        "source_span": {
            "start_frame": 300,
            "end_frame": 330,
            "start_s": 10.0,
            "end_s": 11.0,
        },
        "constraints": {
            "forbidden_effects": ["3d", "gradient", "bevel", "webgl", "shader"],
            "allowed_primitives": ["solid fill", "html text", "svg stroke", "clip mask"],
        },
        "layers": [
            {
                "layer_id": "root",
                "kind": "group",
                "z_index": 0,
                "content": "root",
                "styling": {"opacity": 1},
            },
            {
                "layer_id": "title",
                "kind": "html_text",
                "z_index": 1,
                "parent_layer_id": "root",
                "content": "Title",
                "styling": {"color": "#ffffff"},
            },
        ],
        "operations": [
            {
                "id": "op-1",
                "tool": "renderFrame",
                "action": "opacity",
                "start_frame": 0,
                "end_frame": 15,
                "start_s": 0,
                "end_s": 0.5,
                "target": "title",
                "parameters": {"from": 0, "to": 1},
                "evidence_absolute_frames": [300, 315],
            },
            {
                "id": "op-2",
                "tool": "renderFrame",
                "action": "translate_x",
                "start_frame": 15,
                "end_frame": 30,
                "start_s": 0.5,
                "end_s": 1.0,
                "target": "root",
                "depends_on": ["op-1"],
                "parameters": {"from": 20, "to": 0},
            },
        ],
        "validation_targets": [
            {
                "name": "title visible",
                "metric": "opacity",
                "expected": 1,
                "frame": 29,
                "target_layer_id": "title",
            }
        ],
    }


class ExecutionPlanSchemaTests(unittest.TestCase):
    def test_accepts_structured_deterministic_plan(self) -> None:
        plan = ExecutionPlan.model_validate(valid_plan_payload())
        self.assertEqual(plan.canvas.duration_frames, 30)

    def test_rejects_unknown_layer_reference(self) -> None:
        payload = valid_plan_payload()
        payload["operations"][0]["target"] = "missing"
        with self.assertRaises(ValidationError):
            ExecutionPlan.model_validate(payload)

    def test_rejects_dependency_cycle(self) -> None:
        payload = valid_plan_payload()
        payload["operations"][0]["depends_on"] = ["op-2"]
        with self.assertRaises(ValidationError):
            ExecutionPlan.model_validate(payload)

    def test_rejects_forbidden_effect_in_styling(self) -> None:
        payload = valid_plan_payload()
        payload["layers"][1]["styling"]["background"] = "linear-gradient(red, blue)"
        with self.assertRaises(ValidationError):
            ExecutionPlan.model_validate(payload)

    def test_rejects_frame_time_mismatch(self) -> None:
        payload = valid_plan_payload()
        payload["operations"][0]["end_s"] = 0.7
        with self.assertRaises(ValidationError):
            ExecutionPlan.model_validate(payload)


if __name__ == "__main__":
    unittest.main()
