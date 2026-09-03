# Codex Project Instructions

## Product Goal

Codex owns the complete reference-video reconstruction job. Unless the user
explicitly requests analysis only, a JSON analysis or construction plan is an
intermediate artifact, not the finished result. Completion requires a rendered
MP4 and verification against the supplied visual evidence.

Read `.agents/skills/video-rebuild/SKILL.md` before running or continuing a
reconstruction job.

## Responsibilities

- Use the Python/Prefect pipeline to locate candidate segments and prepare visual
  evidence packets.
- Analyze the contact sheets and sequences with Codex. Never require a person to
  write analysis JSON.
- Turn the resulting `SegmentAnalysis` reconstruction specification into an
  actual HyperFrames composition unless the user names another renderer.
- Render the composition to MP4, inspect representative frames, and revise
  material differences before reporting completion.
- Treat review as a natural-language choice: `codex` review by default, or
  `human` review when the user requests it. Human review does not replace Codex
  analysis or construction.

## Repository Boundaries

- Put generated packets, compositions, renders, comparisons, and reports under
  `runs/<job-name>/`; `runs/` is intentionally ignored by Git.
- Treat input media as read-only. Never overwrite or delete it.
- Do not commit local paths, credentials, model weights, runtimes, generated
  media, or `config/tools.local.json`.
- Inspect existing tools before installing dependencies. State missing external
  prerequisites plainly instead of describing an analysis-only result as a
  completed reconstruction.
