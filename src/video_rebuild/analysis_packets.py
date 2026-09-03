from __future__ import annotations

import json
import math
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .adapters.motion import MotionAnalysisAdapter
from .adapters.objects import YOLOObjectAdapter
from .adapters.ocr import PaddleOCRAdapter
from .config import ToolPaths
from .media import probe_frame_index, probe_video
from .models import (
    AnalysisFrame,
    AnalysisPacket,
    AnalysisPacketEvidence,
    CandidateSegment,
    ChangeScore,
    ChangeWindow,
    FrameIndexEntry,
    MotionObservation,
    ObjectFrameObservation,
    OCRFrameObservation,
    TimeSpan,
    VideoCandidateResult,
    VisualEvidencePurpose,
    VisualEvidenceSequence,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class FrameExtractionError(RuntimeError):
    pass


@dataclass
class FrameSelection:
    frame_index: int
    pts_s: float
    roles: list[str]


@dataclass
class EvidenceSequenceSelection:
    sequence_id: str
    purpose: VisualEvidencePurpose
    window: TimeSpan
    frames: list[FrameSelection]


def candidate_shards(candidates: list[Any], parallel_workers: int) -> list[list[Any]]:
    if parallel_workers <= 0:
        raise ValueError("parallel_workers must be positive")
    worker_count = min(parallel_workers, len(candidates))
    return [candidates[index::worker_count] for index in range(worker_count)]


def _nearest(entries: list[FrameIndexEntry], target_s: float) -> FrameIndexEntry:
    if not entries:
        raise ValueError("no decoded frames are available for the requested span")
    return min(entries, key=lambda item: (abs(item.pts_s - target_s), item.frame_index))


def _merge_selection(
    selected: dict[int, FrameSelection], entry: FrameIndexEntry, role: str
) -> None:
    existing = selected.get(entry.frame_index)
    if existing is None:
        selected[entry.frame_index] = FrameSelection(
            frame_index=entry.frame_index, pts_s=entry.pts_s, roles=[role]
        )
    elif role not in existing.roles:
        existing.roles.append(role)


def select_coarse_frames(
    span: TimeSpan,
    frame_index: list[FrameIndexEntry],
    *,
    samples_per_second: float = 2.0,
    maximum_frames: int = 30,
    context_s: float = 0.5,
) -> list[FrameSelection]:
    if samples_per_second <= 0:
        raise ValueError("samples_per_second must be positive")
    if maximum_frames <= 0:
        raise ValueError("maximum_frames must be positive")
    if context_s < 0:
        raise ValueError("context_s must be non-negative")
    inside = [item for item in frame_index if span.start_s <= item.pts_s < span.end_s]
    if not inside:
        raise ValueError("candidate span contains no decoded frame")

    duration = span.end_s - span.start_s
    if duration < 1.0:
        target_times = [(span.start_s + span.end_s) / 2]
    else:
        count = min(maximum_frames, max(2, math.ceil(duration * samples_per_second)))
        inset = min(0.1, duration * 0.1)
        lower = span.start_s + inset
        upper = span.end_s - inset
        target_times = [
            lower + (upper - lower) * index / (count - 1) for index in range(count)
        ]

    selected: dict[int, FrameSelection] = {}
    for target_s in target_times:
        _merge_selection(selected, _nearest(inside, target_s), "coarse")

    if context_s > 0:
        context_start = max(0.0, span.start_s - context_s)
        before = [
            item for item in frame_index if context_start <= item.pts_s < span.start_s
        ]
        if before:
            _merge_selection(selected, before[0], "context_before")
        context_end = span.end_s + context_s
        after = [item for item in frame_index if span.end_s <= item.pts_s < context_end]
        if after:
            _merge_selection(selected, after[-1], "context_after")
    return [selected[index] for index in sorted(selected)]


def select_fine_frame_indices(
    windows: list[dict[str, Any]],
    *,
    segment_start_frame: int,
    segment_end_frame: int,
    padding_frames: int = 1,
    maximum_frames: int = 30,
    all_frames_window_limit: int = 12,
) -> list[int]:
    if segment_end_frame <= segment_start_frame:
        raise ValueError("segment frame range is invalid")
    if padding_frames < 0 or maximum_frames <= 0:
        raise ValueError("fine frame limits are invalid")
    candidates: set[int] = set()
    required: set[int] = set()
    for window in windows:
        raw_start = int(window["start_frame"])
        raw_end = int(window["end_frame"])
        start = max(segment_start_frame, raw_start - padding_frames)
        end = min(segment_end_frame - 1, raw_end + padding_frames)
        if raw_end - raw_start + 1 <= all_frames_window_limit:
            candidates.update(range(start, end + 1))
        else:
            candidates.update({
                start,
                max(segment_start_frame, raw_start),
                round(raw_start + (raw_end - raw_start) * 0.25),
                round(raw_start + (raw_end - raw_start) * 0.5),
                round(raw_start + (raw_end - raw_start) * 0.75),
                min(segment_end_frame - 1, raw_end),
                end,
            })
        required.add(int(window["peak_frame"]))
        candidates.add(int(window["peak_frame"]))
    ordered = sorted(candidates)
    if len(ordered) <= maximum_frames:
        return ordered
    kept = {item for item in required if item in candidates}
    slots = max(0, maximum_frames - len(kept))
    remainder = [item for item in ordered if item not in kept]
    if slots:
        if slots == 1:
            kept.add(remainder[len(remainder) // 2])
        else:
            for index in range(slots):
                position = round(index * (len(remainder) - 1) / (slots - 1))
                kept.add(remainder[position])
    return sorted(kept)[:maximum_frames]


def _sample_sequence_window(
    frame_index: list[FrameIndexEntry],
    start_s: float,
    end_s: float,
    samples_per_second: float,
    role: str,
) -> list[FrameSelection]:
    entries = [item for item in frame_index if start_s <= item.pts_s < end_s]
    if len(entries) < 2:
        raise ValueError("visual evidence window contains fewer than two frames")
    count = max(2, math.ceil((end_s - start_s) * samples_per_second))
    count = min(count, len(entries))
    targets = [
        entries[0].pts_s
        + (entries[-1].pts_s - entries[0].pts_s) * index / (count - 1)
        for index in range(count)
    ]
    selected: dict[int, FrameSelection] = {}
    for target_s in targets:
        _merge_selection(selected, _nearest(entries, target_s), role)
    if len(selected) < 2:
        for entry in (entries[0], entries[-1]):
            _merge_selection(selected, entry, role)
    return [selected[index] for index in sorted(selected)]


def select_visual_evidence_sequences(
    span: TimeSpan,
    frame_index: list[FrameIndexEntry],
    motion_windows: list[dict[str, Any]],
    *,
    samples_per_second: float = 15.0,
    boundary_context_s: float = 0.4,
    boundary_inside_s: float = 0.4,
    internal_radius_s: float = 0.2,
    maximum_internal_sequences: int = 3,
) -> list[EvidenceSequenceSelection]:
    if samples_per_second <= 0:
        raise ValueError("sequence samples_per_second must be positive")
    if min(boundary_context_s, boundary_inside_s, internal_radius_s) <= 0:
        raise ValueError("visual evidence window sizes must be positive")
    if maximum_internal_sequences < 0:
        raise ValueError("maximum_internal_sequences must be non-negative")
    video_start_s = frame_index[0].pts_s
    last = frame_index[-1]
    video_end_s = last.pts_s + max(last.duration_s, 1e-6)

    windows = [
        (
            "entrance",
            VisualEvidencePurpose.ENTRANCE,
            max(video_start_s, span.start_s - boundary_context_s),
            min(video_end_s, span.start_s + boundary_inside_s),
        ),
        (
            "exit",
            VisualEvidencePurpose.EXIT,
            max(video_start_s, span.end_s - boundary_inside_s),
            min(video_end_s, span.end_s + boundary_context_s),
        ),
    ]
    entry_by_index = {entry.frame_index: entry for entry in frame_index}
    ranked = sorted(
        motion_windows,
        key=lambda item: float(item.get("peak_score", 0.0)),
        reverse=True,
    )[:maximum_internal_sequences]
    for number, raw in enumerate(ranked, 1):
        peak = entry_by_index.get(int(raw["peak_frame"]))
        if peak is None:
            raise ValueError(f"motion peak frame is absent from frame index: {raw}")
        windows.append(
            (
                f"internal-{number:03d}",
                VisualEvidencePurpose.INTERNAL,
                max(span.start_s, peak.pts_s - internal_radius_s),
                min(span.end_s, peak.pts_s + internal_radius_s),
            )
        )

    sequences: list[EvidenceSequenceSelection] = []
    for sequence_id, purpose, start_s, end_s in windows:
        if end_s <= start_s:
            continue
        role = f"{purpose.value}_sequence"
        sequences.append(
            EvidenceSequenceSelection(
                sequence_id=sequence_id,
                purpose=purpose,
                window=TimeSpan(start_s=start_s, end_s=end_s),
                frames=_sample_sequence_window(
                    frame_index, start_s, end_s, samples_per_second, role
                ),
            )
        )
    return sequences


def _span_with_frames(
    span: TimeSpan, frame_index: list[FrameIndexEntry]
) -> TimeSpan:
    entries = [item for item in frame_index if span.start_s <= item.pts_s < span.end_s]
    if not entries:
        raise ValueError("span contains no decoded frames")
    return TimeSpan(
        start_s=span.start_s,
        end_s=span.end_s,
        start_frame=entries[0].frame_index,
        end_frame=entries[-1].frame_index + 1,
    )


def _context_span(
    span: TimeSpan,
    video_duration_s: float,
    frame_index: list[FrameIndexEntry],
    context_s: float,
) -> TimeSpan:
    start_s = max(0.0, span.start_s - context_s)
    end_s = min(video_duration_s, span.end_s + context_s)
    entries = [item for item in frame_index if start_s <= item.pts_s < end_s]
    return TimeSpan(
        start_s=start_s,
        end_s=end_s,
        start_frame=entries[0].frame_index,
        end_frame=entries[-1].frame_index + 1,
    )


def _run_ffmpeg(command: list[str], operation: str) -> None:
    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if completed.returncode != 0:
        raise FrameExtractionError(
            f"{operation} failed; command={command!r}; "
            f"exit_code={completed.returncode}; stdout={completed.stdout!r}; "
            f"stderr={completed.stderr!r}"
        )


def extract_analysis_frames(
    video_path: Path,
    selections: list[FrameSelection],
    frame_index: list[FrameIndexEntry],
    ffmpeg_path: Path,
    candidate_root: Path,
) -> list[AnalysisFrame]:
    index_by_number = {item.frame_index: item for item in frame_index}
    frames: list[AnalysisFrame] = []
    for selection in selections:
        entry = index_by_number[selection.frame_index]
        folder_name = (
            "coarse_frames"
            if any(role in {"coarse", "context_before", "context_after"} for role in selection.roles)
            else "fine_frames"
        )
        output_dir = candidate_root / folder_name
        output_dir.mkdir(parents=True, exist_ok=True)
        target = output_dir / f"frame-{entry.frame_index:06d}-{entry.pts_s:.6f}.png"
        if not target.is_file() or target.stat().st_size == 0:
            command = [
                str(ffmpeg_path), "-hide_banner", "-loglevel", "error", "-y",
                "-i", str(video_path), "-vf", f"select=eq(n\\,{entry.frame_index})",
                "-vsync", "0", "-frames:v", "1", str(target),
            ]
            _run_ffmpeg(command, f"exact extraction for frame {entry.frame_index}")
        if not target.is_file() or target.stat().st_size == 0:
            raise FrameExtractionError(f"ffmpeg created no frame file: {target}")
        frames.append(
            AnalysisFrame(
                frame_index=entry.frame_index,
                pts_s=entry.pts_s,
                path=str(target),
                roles=list(selection.roles),
            )
        )
    return frames


def build_contact_sheet(
    frames: list[AnalysisFrame],
    ffmpeg_path: Path,
    output_path: Path,
    *,
    columns: int = 4,
) -> Path:
    if not frames:
        raise ValueError("contact sheet requires at least one frame")
    if output_path.is_file() and output_path.stat().st_size > 0:
        return output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [str(ffmpeg_path), "-hide_banner", "-loglevel", "error", "-y"]
    for frame in frames:
        command.extend(["-i", frame.path])
    filters: list[str] = []
    labels: list[str] = []
    for index, frame in enumerate(frames):
        label = f"f{frame.frame_index} t{frame.pts_s:.3f}s"
        filters.append(
            f"[{index}:v]scale=320:180:force_original_aspect_ratio=decrease,"
            f"pad=320:200:(ow-iw)/2:(oh-ih)/2:black,"
            f"drawtext=text='{label}':x=6:y=h-th-6:fontsize=18:"
            f"fontcolor=white:box=1:boxcolor=black@0.65[v{index}]"
        )
        labels.append(f"[v{index}]")
    if len(frames) == 1:
        filters.append("[v0]null[out]")
    else:
        layout = "|".join(
            f"{(index % columns) * 320}_{(index // columns) * 200}"
            for index in range(len(frames))
        )
        filters.append(
            f"{''.join(labels)}xstack=inputs={len(frames)}:layout={layout}:fill=black[out]"
        )
    command.extend(
        ["-filter_complex", ";".join(filters), "-map", "[out]", "-frames:v", "1", "-update", "1", str(output_path)]
    )
    _run_ffmpeg(command, "contact sheet generation")
    if not output_path.is_file() or output_path.stat().st_size == 0:
        raise FrameExtractionError(f"contact sheet was not created: {output_path}")
    return output_path


def _map_frames(
    frames: list[AnalysisFrame], payload: dict[str, Any], model: type[Any]
) -> list[Any]:
    raw_frames = payload.get("frames")
    if not isinstance(raw_frames, list):
        raise ValueError("measurement payload field 'frames' must be a list")
    by_path: dict[str, dict[str, Any]] = {}
    for raw in raw_frames:
        if not isinstance(raw, dict) or not raw.get("input_path"):
            raise ValueError("measurement result is missing input_path")
        by_path[str(Path(str(raw["input_path"])).resolve())] = raw
    observations = []
    for frame in frames:
        key = str(Path(frame.path).resolve())
        if key not in by_path:
            raise ValueError(f"measurement result missing input frame: {frame.path}")
        observations.append(model.model_validate(by_path[key]))
    return observations


def map_ocr_frames(
    frames: list[AnalysisFrame], payload: dict[str, Any]
) -> list[OCRFrameObservation]:
    return _map_frames(frames, payload, OCRFrameObservation)


def map_object_frames(
    frames: list[AnalysisFrame], payload: dict[str, Any]
) -> list[ObjectFrameObservation]:
    return _map_frames(frames, payload, ObjectFrameObservation)


def _motion_observation(
    payload: dict[str, Any], frame_index: list[FrameIndexEntry]
) -> MotionObservation:
    index = {item.frame_index: item for item in frame_index}
    scores = [
        ChangeScore(
            frame_index=int(item["frame_index"]),
            pts_s=index[int(item["frame_index"])].pts_s,
            score=float(item["score"]),
        )
        for item in payload.get("frame_scores", [])
    ]
    windows: list[ChangeWindow] = []
    for item in payload.get("windows", []):
        start = index[int(item["start_frame"])]
        end = index[int(item["end_frame"])]
        peak = index[int(item["peak_frame"])]
        next_entry = index.get(end.frame_index + 1)
        windows.append(
            ChangeWindow(
                start_frame=start.frame_index,
                end_frame=end.frame_index,
                peak_frame=peak.frame_index,
                start_s=start.pts_s,
                end_s=(
                    next_entry.pts_s
                    if next_entry is not None
                    else end.pts_s + (end.duration_s or 0.0)
                ),
                peak_s=peak.pts_s,
                peak_score=float(item["peak_score"]),
            )
        )
    return MotionObservation(
        method=str(payload.get("adapter", "opencv-frame-difference")),
        start_frame=int(payload["start_frame"]),
        end_frame=int(payload["end_frame"]),
        threshold=float(payload["threshold"]),
        frame_scores=scores,
        windows=windows,
    )


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def _load_measurement_cache(
    path: Path, frames: list[AnalysisFrame]
) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    raw_frames = payload.get("frames") if isinstance(payload, dict) else None
    if not isinstance(raw_frames, list):
        return None
    expected = {str(Path(frame.path).resolve()) for frame in frames}
    observed = {
        str(Path(str(item.get("input_path", ""))).resolve())
        for item in raw_frames
        if isinstance(item, dict) and item.get("input_path")
    }
    return payload if expected == observed else None


def _load_measurement_subset(
    path: Path, frames: list[AnalysisFrame]
) -> dict[str, Any] | None:
    """Read a bounded subset from the one legacy global cache, without inference."""
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    raw_frames = payload.get("frames") if isinstance(payload, dict) else None
    if not isinstance(raw_frames, list):
        return None
    by_path = {
        str(Path(str(item["input_path"])).resolve()): item
        for item in raw_frames
        if isinstance(item, dict) and item.get("input_path")
    }
    expected = [str(Path(frame.path).resolve()) for frame in frames]
    if any(path not in by_path for path in expected):
        return None
    return {**payload, "frames": [by_path[path] for path in expected]}


def select_measurement_frames(
    frames: list[AnalysisFrame], *, maximum_frames: int = 32
) -> list[AnalysisFrame]:
    """Keep GPU measurements bounded to the storyboard/context evidence."""
    selected = [
        frame
        for frame in frames
        if any(
            role in {"coarse", "context_before", "context_after"}
            for role in frame.roles
        )
    ]
    if len(selected) > maximum_frames:
        raise ValueError(
            f"GPU measurement frame limit exceeded: {len(selected)} > {maximum_frames}"
        )
    return selected


def _load_motion_cache(path: Path) -> MotionObservation | None:
    if not path.is_file():
        return None
    try:
        return MotionObservation.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _load_packet_cache(path: Path) -> AnalysisPacket | None:
    if not path.is_file():
        return None
    try:
        packet = AnalysisPacket.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    evidence_paths = [Path(frame.path) for frame in packet.frames]
    evidence_paths.append(Path(packet.contact_sheet_path))
    evidence_paths.extend(
        Path(sequence.contact_sheet_path) for sequence in packet.visual_sequences
    )
    return (
        packet
        if packet.visual_sequences
        and all(path.is_file() and path.stat().st_size > 0 for path in evidence_paths)
        else None
    )


def prepare_analysis_packets(
    video_path: Path,
    candidate_result: VideoCandidateResult,
    tools: ToolPaths,
    output_root: Path,
    *,
    coarse_samples_per_second: float = 2.0,
    maximum_coarse_frames: int = 30,
    context_s: float = 0.5,
    maximum_fine_frames: int = 30,
    ocr_minimum_score: float = 0.5,
    parallel_workers: int = 2,
) -> list[AnalysisPacket]:
    output_root.mkdir(parents=True, exist_ok=True)
    if not candidate_result.candidates:
        return []
    shards = candidate_shards(candidate_result.candidates, parallel_workers)
    if len(shards) > 1:
        with ThreadPoolExecutor(max_workers=len(shards)) as executor:
            futures = [
                executor.submit(
                    prepare_analysis_packets,
                    video_path,
                    candidate_result.model_copy(update={"candidates": shard}),
                    tools,
                    output_root,
                    coarse_samples_per_second=coarse_samples_per_second,
                    maximum_coarse_frames=maximum_coarse_frames,
                    context_s=context_s,
                    maximum_fine_frames=maximum_fine_frames,
                    ocr_minimum_score=ocr_minimum_score,
                    parallel_workers=1,
                )
                for shard in shards
            ]
            completed = [packet for future in futures for packet in future.result()]
        by_id = {packet.segment_id: packet for packet in completed}
        return [by_id[candidate.segment_id] for candidate in candidate_result.candidates]
    total_started = time.perf_counter()
    index_started = time.perf_counter()
    frame_index = probe_frame_index(video_path, tools.ffprobe)
    index_elapsed = time.perf_counter() - index_started

    motion_adapter = MotionAnalysisAdapter(
        python_path=tools.effect_analysis_python,
        worker_path=PROJECT_ROOT / "workers" / "motion_worker.py",
    )
    ocr_adapter = PaddleOCRAdapter(
        python_path=tools.ocr_python,
        worker_path=tools.ocr_worker,
        detection_model_dir=tools.ocr_detection_model,
        recognition_model_dir=tools.ocr_recognition_model,
    )
    object_adapter = YOLOObjectAdapter(
        python_path=tools.person_python,
        worker_path=PROJECT_ROOT / "workers" / "object_worker.py",
        model_path=tools.person_model,
    )
    packets: list[AnalysisPacket] = []
    for candidate in candidate_result.candidates:
        candidate_started = time.perf_counter()
        candidate_root = output_root / candidate.segment_id
        cached_packet = _load_packet_cache(candidate_root / "analysis_packet.json")
        if cached_packet is not None:
            packets.append(cached_packet)
            continue
        span = _span_with_frames(candidate.span, frame_index)
        context_span = _context_span(
            span, candidate_result.video.duration_s, frame_index, context_s
        )
        selections = select_coarse_frames(
            span,
            frame_index,
            samples_per_second=coarse_samples_per_second,
            maximum_frames=maximum_coarse_frames,
            context_s=context_s,
        )
        motion_cache_path = candidate_root / "motion.json"
        motion = _load_motion_cache(motion_cache_path)
        motion_cache_hit = motion is not None
        motion_started = time.perf_counter()
        if motion is None:
            motion_payload = motion_adapter.analyze(
                video_path, int(span.start_frame), int(span.end_frame)
            )
            motion = _motion_observation(motion_payload, frame_index)
        else:
            motion_payload = motion.model_dump(mode="json")
        motion_elapsed = time.perf_counter() - motion_started
        fine_indices = select_fine_frame_indices(
            list(motion_payload.get("windows", [])),
            segment_start_frame=int(span.start_frame),
            segment_end_frame=int(span.end_frame),
            maximum_frames=maximum_fine_frames,
        )
        selected_by_index = {item.frame_index: item for item in selections}
        entry_by_index = {item.frame_index: item for item in frame_index}
        for frame_number in fine_indices:
            _merge_selection(selected_by_index, entry_by_index[frame_number], "fine")
        sequence_selections = select_visual_evidence_sequences(
            span,
            frame_index,
            list(motion_payload.get("windows", [])),
        )
        for sequence in sequence_selections:
            for selection in sequence.frames:
                for role in selection.roles:
                    _merge_selection(
                        selected_by_index,
                        entry_by_index[selection.frame_index],
                        role,
                    )
        selections = [selected_by_index[index] for index in sorted(selected_by_index)]

        extraction_started = time.perf_counter()
        frames = extract_analysis_frames(
            video_path, selections, frame_index, tools.ffmpeg, candidate_root
        )
        extraction_elapsed = time.perf_counter() - extraction_started
        coarse_frames = select_measurement_frames(frames)
        frame_by_index = {frame.frame_index: frame for frame in frames}
        visual_sequences: list[VisualEvidenceSequence] = []
        for sequence in sequence_selections:
            sequence_frames = [
                frame_by_index[selection.frame_index]
                for selection in sequence.frames
            ]
            sequence_sheet = build_contact_sheet(
                sequence_frames,
                tools.ffmpeg,
                candidate_root / "sequences" / f"{sequence.sequence_id}.jpg",
            )
            visual_sequences.append(VisualEvidenceSequence(
                sequence_id=sequence.sequence_id,
                purpose=sequence.purpose,
                window=sequence.window,
                sampling_fps=15.0,
                frames=sequence_frames,
                contact_sheet_path=str(sequence_sheet),
            ))
        sheet_started = time.perf_counter()
        contact_sheet = build_contact_sheet(
            coarse_frames, tools.ffmpeg, candidate_root / "contact_sheet.jpg"
        )
        sheet_elapsed = time.perf_counter() - sheet_started
        _write_json(candidate_root / "motion.json", motion.model_dump(mode="json"))
        _write_json(candidate_root / "frame_manifest.json", [
            frame.model_dump(mode="json") for frame in frames
        ])
        measurement_root = candidate_root / "_measurements"
        ocr_cache_path = measurement_root / "ocr_payload.json"
        ocr_payload = _load_measurement_cache(ocr_cache_path, coarse_frames)
        if ocr_payload is None:
            ocr_payload = _load_measurement_subset(
                output_root / "_measurements" / "ocr_batch.json", coarse_frames
            )
            if ocr_payload is not None:
                _write_json(ocr_cache_path, ocr_payload)
        ocr_cache_hit = ocr_payload is not None
        ocr_started = time.perf_counter()
        if ocr_payload is None:
            ocr_payload = ocr_adapter.recognize(
                [Path(frame.path) for frame in coarse_frames],
                minimum_score=ocr_minimum_score,
            )
            _write_json(ocr_cache_path, ocr_payload)
        ocr_elapsed = time.perf_counter() - ocr_started

        object_cache_path = measurement_root / "object_payload.json"
        object_payload = _load_measurement_cache(object_cache_path, coarse_frames)
        object_cache_hit = object_payload is not None
        object_started = time.perf_counter()
        if object_payload is None:
            object_payload = object_adapter.detect(
                [Path(frame.path) for frame in coarse_frames]
            )
            _write_json(object_cache_path, object_payload)
        object_elapsed = time.perf_counter() - object_started

        timings = {
            "frame_index": index_elapsed,
            "motion_analysis": motion_elapsed,
            "frame_extraction": extraction_elapsed,
            "contact_sheet": sheet_elapsed,
            "candidate_pre_measurement": ocr_started - candidate_started,
            "ocr_candidate": ocr_elapsed,
            "object_candidate": object_elapsed,
            "total_flow": time.perf_counter() - total_started,
        }
        evidence = AnalysisPacketEvidence(
            ocr_device=ocr_payload.get("device"),
            ocr_detection_model="PP-OCRv6_tiny_det",
            ocr_recognition_model="PP-OCRv6_tiny_rec",
            ocr_minimum_score=ocr_minimum_score,
            object_device=object_payload.get("device"),
            object_model=str(tools.person_model),
            coarse_samples_per_second=coarse_samples_per_second,
            measurement_cache_hits={"ocr": ocr_cache_hit, "objects": object_cache_hit},
            timings_s=timings,
        )
        warnings = [
            "Generic object observations are limited to YOLO11n COCO classes; UI, logos, and arbitrary cards may be absent."
        ]
        if not motion.windows:
            warnings.append("No frame-difference window exceeded the adaptive threshold.")
        packet = AnalysisPacket(
            segment_id=candidate.segment_id,
            span=span,
            context_span=context_span,
            video=candidate_result.video,
            frames=frames,
            contact_sheet_path=str(contact_sheet),
            ocr=map_ocr_frames(coarse_frames, ocr_payload),
            objects=map_object_frames(coarse_frames, object_payload),
            motion=motion,
            visual_sequences=visual_sequences,
            evidence=evidence,
            warnings=warnings,
        )
        root = candidate_root
        _write_json(root / "meta.json", {
            "segment_id": packet.segment_id,
            "span": packet.span.model_dump(mode="json"),
            "context_span": packet.context_span.model_dump(mode="json"),
            "video": packet.video.model_dump(mode="json"),
        })
        _write_json(root / "ocr.json", [value.model_dump(mode="json") for value in packet.ocr])
        _write_json(root / "objects.json", [value.model_dump(mode="json") for value in packet.objects])
        _write_json(root / "motion.json", packet.motion.model_dump(mode="json") if packet.motion else None)
        _write_json(root / "analysis_packet.json", packet.model_dump(mode="json"))
        packets.append(packet)
    return packets


def prepare_analysis_packet(
    video_path: Path,
    candidate: CandidateSegment,
    video_duration_s: float,
    tools: ToolPaths,
    output_root: Path,
) -> AnalysisPacket:
    video = probe_video(video_path, tools.ffprobe)
    if abs(video.duration_s - video_duration_s) > 0.1:
        raise ValueError("supplied video duration does not match ffprobe")
    result = VideoCandidateResult(
        video=video, shots=[candidate.span], person_segments=[], candidates=[candidate]
    )
    return prepare_analysis_packets(video_path, result, tools, output_root)[0]
