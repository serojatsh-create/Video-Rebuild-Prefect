from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field, model_serializer, model_validator


class TimeSpan(BaseModel):
    start_s: float = Field(ge=0)
    end_s: float = Field(gt=0)
    start_frame: int | None = Field(default=None, ge=0)
    end_frame: int | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def validate_order(self) -> "TimeSpan":
        if self.end_s <= self.start_s:
            raise ValueError("end_s must be greater than start_s")
        if (self.start_frame is None) != (self.end_frame is None):
            raise ValueError("start_frame and end_frame must be provided together")
        if (
            self.start_frame is not None
            and self.end_frame is not None
            and self.end_frame <= self.start_frame
        ):
            raise ValueError("end_frame must be greater than start_frame")
        return self

    @model_serializer(mode="wrap")
    def omit_missing_frame_anchors(self, handler: Any) -> dict[str, Any]:
        payload = handler(self)
        return {key: value for key, value in payload.items() if value is not None}


class AnalysisStatus(StrEnum):
    COMPLETE = "complete"
    NEEDS_AI = "needs_ai"
    NEEDS_HUMAN = "needs_human"
    FAILED = "failed"


class EvidenceRef(BaseModel):
    time_s: float = Field(ge=0)
    frame_path: str | None = None
    note: str


class VideoMeta(BaseModel):
    path: str
    duration_s: float = Field(gt=0)
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    fps: float = Field(gt=0)
    frame_count: int | None = Field(default=None, gt=0)
    avg_frame_rate: str | None = None
    reported_frame_rate: str | None = None
    time_base: str | None = None
    start_time_s: float = 0.0
    variable_frame_rate: bool = False


class CandidateSegment(BaseModel):
    segment_id: str
    span: TimeSpan


class VideoCandidateResult(BaseModel):
    video: VideoMeta
    shots: list[TimeSpan]
    person_segments: list[TimeSpan]
    candidates: list[CandidateSegment]


class FrameIndexEntry(BaseModel):
    frame_index: int = Field(ge=0)
    pts_s: float = Field(ge=0)
    duration_s: float = Field(default=0.0, ge=0)
    key_frame: bool = False
    picture_type: str | None = None


class AnalysisFrame(BaseModel):
    frame_index: int = Field(ge=0)
    pts_s: float = Field(ge=0)
    path: str = Field(min_length=1)
    roles: list[str] = Field(min_length=1)


class VisualEvidencePurpose(StrEnum):
    ENTRANCE = "entrance"
    INTERNAL = "internal"
    EXIT = "exit"
    TRANSITION = "transition"


class VisualEvidenceSequence(BaseModel):
    sequence_id: str = Field(min_length=1)
    purpose: VisualEvidencePurpose
    window: TimeSpan
    sampling_fps: float = Field(gt=0)
    frames: list[AnalysisFrame] = Field(min_length=2)
    contact_sheet_path: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_sequence(self) -> "VisualEvidenceSequence":
        frame_indices = [frame.frame_index for frame in self.frames]
        frame_times = [frame.pts_s for frame in self.frames]
        if frame_indices != sorted(set(frame_indices)):
            raise ValueError("visual evidence frames must have unique ascending indices")
        if frame_times != sorted(frame_times):
            raise ValueError("visual evidence frames must have ascending PTS")
        if any(
            not self.window.start_s <= frame.pts_s < self.window.end_s
            for frame in self.frames
        ):
            raise ValueError("visual evidence frame must be inside its sequence window")
        return self


class OCRItem(BaseModel):
    text: str
    score: float = Field(ge=0, le=1)
    polygon: list[list[float]]


class OCRFrameObservation(BaseModel):
    input_path: str = Field(min_length=1)
    items: list[OCRItem] = Field(default_factory=list)


class ObjectItem(BaseModel):
    class_id: int = Field(ge=0)
    label: str = Field(min_length=1)
    score: float = Field(ge=0, le=1)
    bbox: list[float] = Field(min_length=4, max_length=4)


class ObjectFrameObservation(BaseModel):
    input_path: str = Field(min_length=1)
    items: list[ObjectItem] = Field(default_factory=list)


class ChangeScore(BaseModel):
    frame_index: int = Field(ge=0)
    pts_s: float = Field(ge=0)
    score: float = Field(ge=0)


class ChangeWindow(BaseModel):
    start_frame: int = Field(ge=0)
    end_frame: int = Field(ge=0)
    peak_frame: int = Field(ge=0)
    start_s: float = Field(ge=0)
    end_s: float = Field(ge=0)
    peak_s: float = Field(ge=0)
    peak_score: float = Field(ge=0)


class MotionObservation(BaseModel):
    method: str
    start_frame: int = Field(ge=0)
    end_frame: int = Field(gt=0)
    threshold: float = Field(ge=0)
    frame_scores: list[ChangeScore] = Field(default_factory=list)
    windows: list[ChangeWindow] = Field(default_factory=list)


class AnalysisPacketEvidence(BaseModel):
    ocr_device: str | None = None
    ocr_detection_model: str
    ocr_recognition_model: str
    ocr_minimum_score: float = Field(ge=0, le=1)
    object_device: str | None = None
    object_model: str
    coarse_samples_per_second: float = Field(gt=0)
    measurement_cache_hits: dict[str, bool] = Field(default_factory=dict)
    timings_s: dict[str, float] = Field(default_factory=dict)


class AnalysisPacket(BaseModel):
    segment_id: str
    span: TimeSpan
    context_span: TimeSpan
    video: VideoMeta
    frames: list[AnalysisFrame]
    contact_sheet_path: str = Field(min_length=1)
    ocr: list[OCRFrameObservation]
    objects: list[ObjectFrameObservation]
    motion: MotionObservation | None = None
    visual_sequences: list[VisualEvidenceSequence] = Field(default_factory=list)
    evidence: AnalysisPacketEvidence | None = None
    warnings: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_frame_times(self) -> "AnalysisPacket":
        if (
            self.context_span.start_s > self.span.start_s
            or self.context_span.end_s < self.span.end_s
        ):
            raise ValueError("context span must contain candidate span")
        for frame in self.frames:
            if not self.context_span.start_s <= frame.pts_s < self.context_span.end_s:
                raise ValueError("analysis frame PTS must be inside packet context span")
        return self


class LayerKind(StrEnum):
    BACKGROUND = "background"
    VIDEO = "video"
    IMAGE = "image"
    TEXT = "text"
    LOGO = "logo"
    SHAPE = "shape"
    CHART = "chart"
    UI = "ui"
    SUBTITLE = "subtitle"
    EFFECT = "effect"
    THREE_D = "3d"
    UNKNOWN = "unknown"


class MaterialKind(StrEnum):
    VIDEO = "video"
    RASTER_IMAGE = "raster_image"
    SVG = "svg"
    TEXT = "text"
    LOGO = "logo"
    UI_CAPTURE = "ui_capture"
    THREE_D = "3d"
    GENERATED = "generated"
    UNKNOWN = "unknown"


class AnimationPhaseKind(StrEnum):
    ENTRANCE = "entrance"
    HOLD = "hold"
    INTERNAL = "internal"
    EXIT = "exit"
    TRANSITION = "transition"


class AnimationEffect(StrEnum):
    HARD_CUT = "hard_cut"
    FADE = "fade"
    SLIDE = "slide"
    SCALE = "scale"
    ROTATE = "rotate"
    MASK_REVEAL = "mask_reveal"
    WIPE = "wipe"
    TYPE_ON = "type_on"
    BLUR = "blur"
    FLASH = "flash"
    MORPH = "morph"
    CAMERA_PUSH = "camera_push"
    CAMERA_PAN = "camera_pan"
    HOLD = "hold"
    COMPOSITE = "composite"
    UNKNOWN = "unknown"


class ReconstructionTool(StrEnum):
    JIANYING = "jianying"
    SVG = "svg"
    HTML = "html"
    FFMPEG = "ffmpeg"
    THREE_D = "3d"
    HYBRID = "hybrid"
    UNKNOWN = "unknown"


class LayerState(BaseModel):
    frame_index: int = Field(ge=0)
    time_s: float = Field(ge=0)
    x_norm: float | None = Field(default=None, ge=0, le=1)
    y_norm: float | None = Field(default=None, ge=0, le=1)
    width_norm: float | None = Field(default=None, gt=0, le=1)
    height_norm: float | None = Field(default=None, gt=0, le=1)
    scale_x: float | None = Field(default=None, gt=0)
    scale_y: float | None = Field(default=None, gt=0)
    rotation_deg: float | None = None
    opacity: float | None = Field(default=None, ge=0, le=1)
    visible: bool = True


class VisualLayerSpec(BaseModel):
    layer_id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    kind: LayerKind
    z_index: int
    content: str | None = None
    styling: dict[str, Any] = Field(default_factory=dict)
    states: list[LayerState] = Field(default_factory=list)
    evidence_frame_indices: list[int] = Field(min_length=1)
    confidence: float = Field(ge=0, le=1)
    uncertainties: list[str] = Field(default_factory=list)


class AnimationPhaseSpec(BaseModel):
    phase_id: str = Field(min_length=1)
    kind: AnimationPhaseKind
    start_frame: int = Field(ge=0)
    end_frame: int = Field(gt=0)
    start_s: float = Field(ge=0)
    end_s: float = Field(gt=0)
    target_layer_ids: list[str] = Field(min_length=1)
    effect: AnimationEffect
    direction: str | None = None
    easing: str | None = None
    parameter_changes: dict[str, Any] = Field(default_factory=dict)
    evidence_sequence_ids: list[str] = Field(min_length=1)
    evidence_frame_indices: list[int] = Field(min_length=2)
    confidence: float = Field(ge=0, le=1)
    uncertainties: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_phase(self) -> "AnimationPhaseSpec":
        if self.end_frame <= self.start_frame or self.end_s <= self.start_s:
            raise ValueError("animation phase end must be after start")
        if self.evidence_frame_indices != sorted(set(self.evidence_frame_indices)):
            raise ValueError("animation evidence frames must be unique and ascending")
        if any(
            frame < self.start_frame or frame > self.end_frame
            for frame in self.evidence_frame_indices
        ):
            raise ValueError("animation evidence frame must be inside the phase")
        return self


class MaterialSpec(BaseModel):
    material_id: str = Field(min_length=1)
    kind: MaterialKind
    description: str = Field(min_length=1)
    source_path: str | None = None
    generation_notes: str | None = None
    transparent_background: bool | None = None
    evidence_frame_indices: list[int] = Field(default_factory=list)


class ValidationPoint(BaseModel):
    frame_index: int = Field(ge=0)
    time_s: float = Field(ge=0)
    expected_visible_layers: list[str] = Field(default_factory=list)
    checks: dict[str, Any] = Field(default_factory=dict)


class ImplementationSpec(BaseModel):
    primary_tool: ReconstructionTool
    layer_build_order: list[str] = Field(min_length=1)
    technical_steps: list[str] = Field(min_length=1)
    required_effects: list[str] = Field(default_factory=list)
    fallback_tool: ReconstructionTool | None = None
    test_points: list[ValidationPoint] = Field(min_length=1)
    limitations: list[str] = Field(default_factory=list)


class VisualReconstructionSpec(BaseModel):
    schema_version: str = "1.0"
    content_category: str = Field(min_length=1)
    design_style: list[str] = Field(min_length=1)
    composition: str = Field(min_length=1)
    layers: list[VisualLayerSpec] = Field(min_length=1)
    phases: list[AnimationPhaseSpec] = Field(min_length=2)
    materials: list[MaterialSpec] = Field(default_factory=list)
    implementation: ImplementationSpec
    unresolved_questions: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_reconstruction(self) -> "VisualReconstructionSpec":
        layer_ids = [layer.layer_id for layer in self.layers]
        if layer_ids != list(dict.fromkeys(layer_ids)):
            raise ValueError("visual layer IDs must be unique")
        phase_ids = [phase.phase_id for phase in self.phases]
        if phase_ids != list(dict.fromkeys(phase_ids)):
            raise ValueError("animation phase IDs must be unique")
        known_layers = set(layer_ids)
        for phase in self.phases:
            unknown = set(phase.target_layer_ids) - known_layers
            if unknown:
                raise ValueError(f"animation phase references unknown layer: {sorted(unknown)}")
        unknown_build_layers = set(self.implementation.layer_build_order) - known_layers
        if unknown_build_layers:
            raise ValueError(
                f"implementation references unknown layer: {sorted(unknown_build_layers)}"
            )
        phase_kinds = {phase.kind for phase in self.phases}
        if AnimationPhaseKind.ENTRANCE not in phase_kinds:
            raise ValueError("reconstruction must describe an entrance phase")
        if AnimationPhaseKind.EXIT not in phase_kinds:
            raise ValueError("reconstruction must describe an exit phase")
        return self


class SegmentAnalysis(BaseModel):
    segment_id: str
    span: TimeSpan
    status: AnalysisStatus
    visual_summary: str
    evidence: list[EvidenceRef] = Field(default_factory=list)
    uncertainties: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0, le=1)
    analyzer: str = "unavailable"
    details: dict[str, Any] = Field(default_factory=dict)
    reconstruction: VisualReconstructionSpec | None = None
    source_packet_path: str | None = None

    @model_validator(mode="after")
    def require_reconstruction_for_complete(self) -> "SegmentAnalysis":
        if self.status == AnalysisStatus.COMPLETE and self.reconstruction is None:
            raise ValueError("complete visual analysis requires reconstruction")
        if self.status == AnalysisStatus.COMPLETE and not self.source_packet_path:
            raise ValueError("complete visual analysis requires source_packet_path")
        return self


class ImplementationStrategy(StrEnum):
    JIANYING = "jianying"
    SVG = "svg"
    HYBRID = "hybrid"
    HTML_SVG = "html_svg"
    OTHER = "other"


class ExecutionCanvas(BaseModel):
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    fps: float = Field(gt=0)
    duration_frames: int = Field(gt=0)
    duration_s: float = Field(gt=0)
    background_color: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_duration(self) -> "ExecutionCanvas":
        expected_s = self.duration_frames / self.fps
        if abs(self.duration_s - expected_s) > (1.0 / self.fps):
            raise ValueError("canvas duration_s must match duration_frames / fps within one frame")
        return self


class ExecutionSourceSpan(BaseModel):
    start_frame: int = Field(ge=0)
    end_frame: int = Field(gt=0, description="Exclusive source-frame endpoint.")
    start_s: float = Field(ge=0)
    end_s: float = Field(gt=0)

    @model_validator(mode="after")
    def validate_order(self) -> "ExecutionSourceSpan":
        if self.end_frame <= self.start_frame:
            raise ValueError("source span end_frame must be greater than start_frame")
        if self.end_s <= self.start_s:
            raise ValueError("source span end_s must be greater than start_s")
        return self


class ExecutionConstraints(BaseModel):
    forbidden_effects: list[str] = Field(default_factory=list)
    allowed_primitives: list[str] = Field(default_factory=list)


class ExecutionLayer(BaseModel):
    layer_id: str = Field(min_length=1)
    kind: str = Field(min_length=1)
    z_index: int = 0
    parent_layer_id: str | None = None
    mask_layer_id: str | None = None
    content: Any | None = None
    source: str | None = None
    styling: dict[str, Any] = Field(default_factory=dict)


class ExecutionOperation(BaseModel):
    id: str
    tool: str
    action: str
    start_s: float = Field(ge=0)
    end_s: float = Field(gt=0)
    start_frame: int | None = Field(default=None, ge=0)
    end_frame: int | None = Field(
        default=None,
        gt=0,
        description="Exclusive local-frame endpoint.",
    )
    target: str | None = None
    easing: str = "linear"
    parameters: dict[str, Any] = Field(default_factory=dict)
    depends_on: list[str] = Field(default_factory=list)
    evidence_absolute_frames: list[int] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_operation_order(self) -> "ExecutionOperation":
        if self.end_s <= self.start_s:
            raise ValueError("operation end_s must be greater than start_s")
        if (self.start_frame is None) != (self.end_frame is None):
            raise ValueError("operation start_frame and end_frame must be provided together")
        if (
            self.start_frame is not None
            and self.end_frame is not None
            and self.end_frame <= self.start_frame
        ):
            raise ValueError("operation end_frame must be greater than start_frame")
        return self


class ValidationTarget(BaseModel):
    name: str
    metric: str
    expected: Any
    tolerance: float | None = Field(default=None, ge=0)
    frame: int | None = Field(default=None, ge=0)
    target_layer_id: str | None = None
    evidence_absolute_frame: int | None = Field(default=None, ge=0)


class ExecutionPlan(BaseModel):
    segment_id: str
    strategy: ImplementationStrategy
    router_version: str
    canvas: ExecutionCanvas | None = None
    source_span: ExecutionSourceSpan | None = None
    constraints: ExecutionConstraints | None = None
    layers: list[ExecutionLayer] = Field(default_factory=list)
    operations: list[ExecutionOperation] = Field(default_factory=list)
    validation_targets: list[ValidationTarget] = Field(default_factory=list)
    fallbacks: list[ImplementationStrategy] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_execution_graph(self) -> "ExecutionPlan":
        layer_ids = [layer.layer_id for layer in self.layers]
        if layer_ids != list(dict.fromkeys(layer_ids)):
            raise ValueError("execution layer IDs must be unique")
        known_layers = set(layer_ids)
        for layer in self.layers:
            for relation_name, relation_id in (
                ("parent", layer.parent_layer_id),
                ("mask", layer.mask_layer_id),
            ):
                if relation_id is not None and relation_id not in known_layers:
                    raise ValueError(
                        f"execution layer {relation_name} references unknown layer: {relation_id}"
                    )
                if relation_id == layer.layer_id:
                    raise ValueError(f"execution layer cannot use itself as {relation_name}")

        operation_ids = [operation.id for operation in self.operations]
        if operation_ids != list(dict.fromkeys(operation_ids)):
            raise ValueError("execution operation IDs must be unique")
        known_operations = set(operation_ids)
        for operation in self.operations:
            if self.layers and operation.target not in known_layers:
                raise ValueError(
                    f"execution operation references unknown target layer: {operation.target}"
                )
            unknown_dependencies = set(operation.depends_on) - known_operations
            if unknown_dependencies:
                raise ValueError(
                    "execution operation references unknown dependency: "
                    f"{sorted(unknown_dependencies)}"
                )

        graph = {operation.id: operation.depends_on for operation in self.operations}
        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(operation_id: str) -> None:
            if operation_id in visiting:
                raise ValueError("execution operation dependencies must be acyclic")
            if operation_id in visited:
                return
            visiting.add(operation_id)
            for dependency_id in graph[operation_id]:
                visit(dependency_id)
            visiting.remove(operation_id)
            visited.add(operation_id)

        for operation_id in graph:
            visit(operation_id)

        if self.canvas is not None:
            frame_duration_s = 1.0 / self.canvas.fps
            if self.source_span is not None:
                source_frames = self.source_span.end_frame - self.source_span.start_frame
                if source_frames != self.canvas.duration_frames:
                    raise ValueError("source span frame count must match canvas duration_frames")
                source_duration_s = self.source_span.end_s - self.source_span.start_s
                if abs(source_duration_s - self.canvas.duration_s) > frame_duration_s:
                    raise ValueError("source span seconds must match canvas duration within one frame")
            for operation in self.operations:
                if operation.start_frame is None or operation.end_frame is None:
                    continue
                if operation.end_frame > self.canvas.duration_frames:
                    raise ValueError("execution operation frame range exceeds canvas duration")
                if abs(operation.start_s - operation.start_frame / self.canvas.fps) > frame_duration_s:
                    raise ValueError("operation start frame/time mapping differs by more than one frame")
                if abs(operation.end_s - operation.end_frame / self.canvas.fps) > frame_duration_s:
                    raise ValueError("operation end frame/time mapping differs by more than one frame")
            for target in self.validation_targets:
                if target.frame is not None and target.frame >= self.canvas.duration_frames:
                    raise ValueError("validation target frame exceeds canvas duration")
                if target.target_layer_id is not None and target.target_layer_id not in known_layers:
                    raise ValueError("validation target references unknown layer")

        if self.constraints is not None:
            forbidden = {
                item.strip().lower()
                for item in self.constraints.forbidden_effects
                if item.strip()
            }

            def contains_forbidden(value: Any) -> bool:
                if isinstance(value, dict):
                    return any(
                        contains_forbidden(key) or contains_forbidden(item)
                        for key, item in value.items()
                    )
                if isinstance(value, (list, tuple, set)):
                    return any(contains_forbidden(item) for item in value)
                if isinstance(value, str):
                    normalized = value.lower()
                    return any(term in normalized for term in forbidden)
                return False

            if contains_forbidden(self.constraints.allowed_primitives):
                raise ValueError("forbidden effect appears in allowed_primitives")
            for layer in self.layers:
                if contains_forbidden(
                    {"kind": layer.kind, "source": layer.source, "styling": layer.styling}
                ):
                    raise ValueError(f"forbidden effect appears in layer {layer.layer_id}")
            for operation in self.operations:
                if contains_forbidden(
                    {
                        "tool": operation.tool,
                        "action": operation.action,
                        "easing": operation.easing,
                        "parameters": operation.parameters,
                    }
                ):
                    raise ValueError(f"forbidden effect appears in operation {operation.id}")
        return self


class GateStatus(StrEnum):
    PASS = "pass"
    FAIL = "fail"
    WARN = "warn"
    SKIP = "skip"


class QCGateResult(BaseModel):
    gate: str
    status: GateStatus
    message: str
    evidence: dict[str, Any] = Field(default_factory=dict)


class QCDecision(StrEnum):
    PASS = "pass"
    RETRY = "retry"
    REROUTE = "reroute"
    HUMAN_REVIEW = "human_review"
    FAIL = "fail"


class QCReport(BaseModel):
    segment_id: str
    gates: list[QCGateResult]
    auto_score: float = Field(ge=0, le=1)
    decision: QCDecision
    recommended_action: str


class FullFlowResult(BaseModel):
    analysis: SegmentAnalysis
    execution_plan: ExecutionPlan
    qc: QCReport
