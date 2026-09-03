from __future__ import annotations

import json
import unittest
from pathlib import Path

from video_rebuild.models import ExecutionPlan

PLANS_DIR = Path(__file__).resolve().parent / "fixtures" / "visionless_plans"

PLAN_FILES = {
    "candidate-0006": {
        "path": PLANS_DIR / "candidate-0006.json",
        "frames": 97,
        "fps": 30.0,
        "strategy": "html_svg",
    },
    "candidate-0007": {
        "path": PLANS_DIR / "candidate-0007.json",
        "frames": 252,
        "fps": 30.0,
        "strategy": "html_svg",
    },
}


class VisionlessExecutionPlanTests(unittest.TestCase):
    def _load(self, name: str) -> tuple[dict, ExecutionPlan]:
        meta = PLAN_FILES[name]
        payload = json.loads(meta["path"].read_text(encoding="utf-8"))
        plan = ExecutionPlan.model_validate(payload)
        return payload, plan

    def test_plans_validate_and_match_spec(self) -> None:
        for name, meta in PLAN_FILES.items():
            with self.subTest(name=name):
                _, plan = self._load(name)
                self.assertEqual(plan.canvas.width, 1280)
                self.assertEqual(plan.canvas.height, 720)
                self.assertEqual(plan.canvas.fps, meta["fps"])
                self.assertEqual(plan.canvas.duration_frames, meta["frames"])
                self.assertEqual(plan.strategy.value, meta["strategy"])
                source_frames = (
                    plan.source_span.end_frame - plan.source_span.start_frame
                )
                self.assertEqual(source_frames, meta["frames"])
                self.assertGreater(len(plan.layers), 0)
                self.assertGreater(len(plan.operations), 0)
                self.assertGreater(len(plan.validation_targets), 2)

    def test_operations_stay_within_canvas(self) -> None:
        for name in PLAN_FILES:
            with self.subTest(name=name):
                _, plan = self._load(name)
                for op in plan.operations:
                    self.assertLessEqual(op.end_frame, plan.canvas.duration_frames)
                    self.assertGreater(op.end_frame, op.start_frame)

    def test_every_layer_has_styling(self) -> None:
        for name in PLAN_FILES:
            with self.subTest(name=name):
                _, plan = self._load(name)
                for layer in plan.layers:
                    self.assertIsInstance(layer.styling, dict)
                    self.assertIn("left", layer.styling)
                    self.assertIn("top", layer.styling)

    def test_forbidden_terms_absent_from_scanned_fields(self) -> None:
        c06_payload = json.loads(
            PLAN_FILES["candidate-0006"]["path"].read_text(encoding="utf-8")
        )
        forbidden = {
            t.strip().lower()
            for t in c06_payload["constraints"]["forbidden_effects"]
        }

        def has_term(value) -> bool:
            if isinstance(value, dict):
                return any(has_term(k) or has_term(v) for k, v in value.items())
            if isinstance(value, (list, tuple, set)):
                return any(has_term(v) for v in value)
            if isinstance(value, str):
                low = value.lower()
                return any(t in low for t in forbidden)
            return False

        for name in PLAN_FILES:
            with self.subTest(name=name):
                _, plan = self._load(name)
                for layer in plan.layers:
                    self.assertFalse(
                        has_term(
                            {
                                "kind": layer.kind,
                                "source": layer.source,
                                "styling": layer.styling,
                            }
                        ),
                        f"forbidden term in layer {layer.layer_id}",
                    )
                for op in plan.operations:
                    self.assertFalse(
                        has_term(
                            {
                                "tool": op.tool,
                                "action": op.action,
                                "easing": op.easing,
                                "parameters": op.parameters,
                            }
                        ),
                        f"forbidden term in op {op.id}",
                    )

    def test_validation_targets_reference_known_layers(self) -> None:
        for name in PLAN_FILES:
            with self.subTest(name=name):
                _, plan = self._load(name)
                layer_ids = {l.layer_id for l in plan.layers}
                for t in plan.validation_targets:
                    if t.target_layer_id is not None:
                        self.assertIn(t.target_layer_id, layer_ids)
                    if t.frame is not None:
                        self.assertLess(t.frame, plan.canvas.duration_frames)


if __name__ == "__main__":
    unittest.main()
