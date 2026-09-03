# Video Rebuild Prefect

A Codex-driven workflow for analyzing reference-video evidence and turning it
into a rendered reconstruction.

The pipeline detects candidate segments, extracts coarse and motion-focused frame
sequences, records OCR/object/motion measurements, and asks Codex to analyze the
visual evidence. Codex always produces the analysis and, for a reconstruction
request, continues from that plan into HyperFrames construction and MP4 rendering.
Human involvement is an optional review mode; no human-authored JSON is required.

The Python and Prefect components prepare deterministic evidence and validate
structured results. Codex is the construction agent: it writes the HyperFrames
composition from the validated plan, renders it, inspects key frames, and revises
the result. A plan JSON alone is complete only for an explicitly analysis-only job.

## Requirements

- Windows and Python 3.12+
- Codex CLI authenticated locally
- FFmpeg and FFprobe
- Node.js and HyperFrames for rendered-video output
- Project-specific OCR, object detection, and scene-analysis runtimes for packet generation

Install the Python package in a project-owned environment:

```powershell
python -m pip install -e .
```

Copy `config/tools.example.json` to `config/tools.local.json`, then replace the
placeholder values with absolute paths on your machine. The local config is ignored
by Git.

## Pipeline

For an end-to-end job, open the repository in Codex and ask it to use the bundled
`video-rebuild` skill, supplying the reference video and an output job name. Choose
`codex` review (default) or `human` review in natural language. Codex follows
`AGENTS.md`, runs the evidence and analysis stages, builds the HyperFrames
composition, and renders the MP4 under `runs/<job-name>/`.

The commands below expose the individual deterministic stages for inspection,
reuse, and debugging.

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
normalized analysis JSON. For reconstruction requests, the bundled Codex skill
then treats this JSON as a construction brief and continues through rendering.

## Tests

```powershell
python -m unittest discover -s tests -v
```

Generated runs, model weights, private runtime paths, virtual environments, and
Prefect state are intentionally excluded from the repository.
