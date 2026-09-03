from __future__ import annotations

import unittest

from video_rebuild.adapters.effect_analysis import (
    EffectAnalysisError,
    unwrap_effect_result,
)


class EffectAnalysisPayloadTests(unittest.TestCase):
    def test_unwraps_success_result(self) -> None:
        result = unwrap_effect_result({"ok": True, "result": {"duration": 1.5}})

        self.assertEqual(result, {"duration": 1.5})

    def test_rejects_reported_failure(self) -> None:
        with self.assertRaisesRegex(EffectAnalysisError, "BAD_INPUT"):
            unwrap_effect_result(
                {"ok": False, "error": {"code": "BAD_INPUT", "message": "bad path"}}
            )

    def test_rejects_non_object_result(self) -> None:
        with self.assertRaisesRegex(EffectAnalysisError, "JSON object"):
            unwrap_effect_result({"ok": True, "result": []})


if __name__ == "__main__":
    unittest.main()

