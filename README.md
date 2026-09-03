# Video Rebuild Prefect

A Prefect-oriented pipeline for turning reference-video evidence into a structured,
frame-addressed reconstruction specification.

The pipeline detects candidate segments, extracts coarse and motion-focused frame
sequences, records OCR/object/motion measurements, and asks Codex to analyze the
visual evidence. Codex always produces the analysis. Human involvement is an
optional review flag after analysis; no human-authored JSON is required.

## Requirements

- Windows and Python 3.12+
- Codex CLI authenticated locally
- FFmpeg and FFprobe
- Project-specific OCR, object detection, and scene-analysis runtimes for packet generation

Install the Python package in a project-owned environment:

```powershell
python -m pip install -e .
```

Copy `config/tools.example.json` to `config/tools.local.json`, then replace the
placeholder values with absolute paths on your machine. The local config is ignored
by Git.

## Pipeline

Prepare evidence packets from a video:

```powershell
python scripts/prepare_analysis_packets.py reference.mp4 config/tools.local.json runs/example
```

Analyze one prepared packet with Codex:

```powershell
video-rebuild-codex-analyze runs/example/candidate-0001/analysis_packet.json runs/example/candidate-0001/segment_analysis.json
```

Add `--human-review` when the generated result should wait for optional manual
review. This does not replace or skip Codex analysis.

The analysis command attaches the packet's main contact sheet and each temporal
sequence contact sheet to `codex exec`, constrains the response with the exported
`SegmentAnalysis` JSON schema, validates the response with Pydantic, and writes the
normalized analysis JSON.

## Tests

```powershell
python -m unittest discover -s tests -v
```

Generated runs, model weights, private runtime paths, virtual environments, and
Prefect state are intentionally excluded from the repository.
