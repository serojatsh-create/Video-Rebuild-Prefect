---
name: video-rebuild
description: Reconstruct a reference-video segment end to end with this repository, from evidence extraction and Codex visual analysis through HyperFrames construction, MP4 rendering, and Codex or optional human review. Use when the requested deliverable is a recreated video, not merely an analysis JSON.
---

# Video Rebuild

Produce a rendered reconstruction. Codex is both the visual analyst and the
default construction agent; the user does not author JSON or translate the plan
into code.

## Choose the Finish Line

- Default to `codex` review and continue through a verified MP4.
- Use `human` review only when the user asks for it. Still analyze, construct,
  and render before pausing for their judgment.
- Stop at `SegmentAnalysis` only when the user explicitly requests analysis
  without a rendered reconstruction.

## Run the Job

1. Inspect the repository, the input path, existing job artifacts, and available
   runtimes. Reuse valid artifacts instead of restarting completed stages.
2. If no `AnalysisPacket` exists, configure `config/tools.local.json` from the
   example and run the candidate and packet preparation pipeline. Keep all new
   files under `runs/<job-name>/`.
3. Analyze each requested packet with `video-rebuild-codex-analyze`. The command
   attaches the main and temporal contact sheets, validates the structured
   response, and writes `segment_analysis.json`.
4. Read the validated analysis and its evidence. Use HyperFrames as the default
   renderer unless the user specifies another tool. When HyperFrames skills are
   available, load the `hyperframes` entry skill and follow its authoring and CLI
   rules. Create the composition under the same job directory.
5. Implement the layers, materials, frame timing, entrance/internal/exit phases,
   and test points described by the analysis. Resolve uncertainty with the source
   evidence; do not ask the user to turn the plan into code.
6. Validate the composition with the installed HyperFrames CLI, render an MP4,
   and inspect the analysis test frames plus entrance and exit frames. Compare
   them with the evidence sheets and fix material visual or timing errors.
7. In `codex` review mode, complete the best justified revisions and report the
   MP4 path, checks performed, and remaining uncertainty. In `human` mode, show
   the rendered result and wait for the user's acceptance or revision notes.

## External Prerequisites

The repository does not vendor Python runtimes, model weights, FFmpeg, Node.js,
or HyperFrames. Inspect the machine first. If a required component is absent,
name that exact component and the blocked stage. Installation or download is a
separate action subject to the user's authorization.
