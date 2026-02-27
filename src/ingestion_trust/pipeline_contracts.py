from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class PipelinePaths:
    project_root: Path
    artifacts_dir: Path
    metrics_dir: Path
    audit_dir: Path
    provenance_dir: Path
    curated_dir: Path
    debug_dir: Path
    figures_dir: Path
    model_dir: Path


@dataclass
class TrackExecutionSpec:
    track_mode: str
    metrics_dir: Path
    figures_dir: Path
    model_dir: Path
    lgbm_model_dir: Path
    seed: int
    enable_next_click: bool
    winner_style_config: Any


@dataclass
class TrustStageResult:
    spark: Any
    raw_df: Any
    cast_df: Any
    feat_df: Any
    validated_df: Any
    scored_df: Any
    curated_df: Any
    curated_pd: Any
    curated_path: Path
    validation_summary: Dict[str, Any]
    etl_time_sec: float
    baseline_stats: Dict[str, Any]
    input_fingerprint: str
    compatibility_copy_map: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class ModelingStageResult:
    track_summaries: Dict[str, Dict[str, Any]]
    track_descriptors: List[Dict[str, Any]]
    combined_metric_frames: List[Any]
    primary_track: str
    primary_track_summary: Dict[str, Any]
    train_time_sec: float
    stability_summary: Dict[str, Any]


@dataclass
class BatchPipelineResult:
    summary: Dict[str, Any]
    trust_stage: TrustStageResult
    modeling_stage: ModelingStageResult
    debug_outputs: Dict[str, Any]
    run_metadata: Optional[Dict[str, Any]] = None
