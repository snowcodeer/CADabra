from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, field_validator


class ProcessRequest(BaseModel):
    source: Literal["sample", "upload"]
    sample_id: str | None = None
    filename: str | None = None


class Feature(BaseModel):
    type: Literal["extrude", "cut", "pocket", "boss"]
    sketch_plane: Literal["XY", "XZ", "YZ"]
    profile_description: str
    height_mm: float
    position_mm: tuple[float, float, float] = (0.0, 0.0, 0.0)


class LLMReasoningOutput(BaseModel):
    reasoning: str
    features: list[Feature]
    cadquery_code: str
    confidence: Literal["high", "medium", "low"]

    @field_validator("cadquery_code")
    @classmethod
    def must_import_cadquery(cls, value: str) -> str:
        lowered = value.lower()
        if "import cadquery" not in lowered and "import cq" not in lowered:
            raise ValueError("cadquery_code must import cadquery")
        return value


class ValidationResult(BaseModel):
    executes: bool
    produces_solid: bool
    is_watertight: bool
    vertex_count: int | None
    face_count: int | None
    bounding_box_mm: tuple[float, float, float] | None
    error: str | None


class PipelineResult(BaseModel):
    input_id: str
    success: bool
    render_grid_url: str | None
    cadquery_code: str | None
    ground_truth_cadquery_code: str | None = None
    step_url: str | None
    stl_url: str | None
    validation: ValidationResult | None
    llm_reasoning: str | None
    confidence: Literal["high", "medium", "low"] | None = None
    error: str | None


class SampleInfo(BaseModel):
    sample_id: str
    display_name: str
    stl_filename: str
    point_cloud_filename: str | None = None
    ground_truth_filename: str | None = None
