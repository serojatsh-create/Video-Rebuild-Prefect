from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from pydantic import ValidationError

from video_rebuild.config import load_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class ConfigurationTests(unittest.TestCase):
    def test_example_config_selects_separate_gpu_runtimes(self) -> None:
        config = load_config(PROJECT_ROOT / "config" / "tools.example.json")

        self.assertEqual(config.person_python.parent.parent.name, "person_gpu")
        self.assertEqual(config.ocr_python.parent.parent.name, "ocr_gpu")

    def test_loads_absolute_tool_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            config_path = Path(temporary_directory) / "tools.json"
            config_path.write_text(
                json.dumps(
                    {
                        "ffmpeg": "E:/tools/ffmpeg.exe",
                        "ffprobe": "E:/tools/ffprobe.exe",
                        "effect_analysis_python": "E:/effect/.venv/Scripts/python.exe",
                        "effect_analysis_project": "E:/effect",
                        "scene_python": "E:/effect/.venv/Scripts/python.exe",
                        "scene_worker": "E:/project/workers/scene_worker.py",
                        "person_python": "E:/project/runtimes/person_cpu/Scripts/python.exe",
                        "person_worker": "E:/project/workers/person_worker.py",
                        "person_model": "E:/project/downloads/models/person/yolo11n.pt",
                        "ocr_python": "E:/project/runtimes/ocr_cpu/Scripts/python.exe",
                        "ocr_worker": "E:/project/workers/ocr_worker.py",
                        "ocr_detection_model": "E:/project/downloads/models/ocr/PP-OCRv6_tiny_det",
                        "ocr_recognition_model": "E:/project/downloads/models/ocr/PP-OCRv6_tiny_rec",
                    }
                ),
                encoding="utf-8",
            )

            config = load_config(config_path)

            self.assertTrue(config.ffmpeg.is_absolute())
            self.assertEqual(config.person_model.name, "yolo11n.pt")

    def test_rejects_relative_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            config_path = Path(temporary_directory) / "tools.json"
            config_path.write_text(
                json.dumps(
                    {
                        "ffmpeg": "relative/ffmpeg.exe",
                        "ffprobe": "E:/tools/ffprobe.exe",
                        "effect_analysis_python": "E:/effect/python.exe",
                        "effect_analysis_project": "E:/effect",
                        "scene_python": "E:/effect/python.exe",
                        "scene_worker": "E:/project/workers/scene_worker.py",
                        "person_python": "E:/person/python.exe",
                        "person_worker": "E:/person/worker.py",
                        "person_model": "E:/person/model.pt",
                        "ocr_python": "E:/ocr/python.exe",
                        "ocr_worker": "E:/ocr/worker.py",
                        "ocr_detection_model": "E:/ocr/PP-OCRv6_tiny_det",
                        "ocr_recognition_model": "E:/ocr/PP-OCRv6_tiny_rec",
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaises(ValidationError):
                load_config(config_path)


if __name__ == "__main__":
    unittest.main()
