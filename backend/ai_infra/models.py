"""Pydantic v2 schemas for the Claude Vision -> CadQuery handoff.

A ``PartDescription`` is the structured contract between the vision step
(``llm_client.call_claude``) and the downstream CadQuery builder. Every
field is something Claude can read off a 6-view orthographic render AND
something a CadQuery builder can pass directly into a method call.

Dimension fields are millimetres; positions are millimetre offsets from
the relevant face centre. ``ExecutionResult`` is a placeholder for the
later CadQuery execution stage and is defined here so the surface area
of this module is stable from day one.
"""

from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


Shape = Literal["rectangle", "circle", "l_shape", "t_shape", "polygon"]
SketchPlane = Literal["XY", "XZ", "YZ"]
Face = Literal["+Z", "-Z", "+X", "-X", "+Y", "-Y"]
FeatureType = Literal["boss", "hole", "pocket", "slot"]
FeatureShape = Literal["circle", "rectangle"]
DepthType = Literal["through", "blind"]
Confidence = Literal["high", "medium", "low"]

EdgeTreatmentType = Literal["fillet", "chamfer"]
EdgeSet = Literal[
    # Coverage shorthands the vision model can pick from. They map to
    # CadQuery edge selectors in the builder.
    "all",          # every edge of the base body
    "top_outer",    # edges on the +Z face perimeter
    "bottom_outer", # edges on the -Z face perimeter
    "vertical",     # edges parallel to Z (the "side" edges of a box/disk)
    "horizontal",   # edges perpendicular to Z (rim edges of top + bottom)
]


# Aliases Claude tends to invent when the rendered part is clearly a
# specific n-gon. We normalise to the canonical "polygon" enum value
# (and rely on BaseBody.sides for the side count) so the refinement
# pass doesn't crash on a literal_error.
_SHAPE_ALIASES: dict[str, str] = {
    "regular_polygon": "polygon",
    "n_gon": "polygon",
    "ngon": "polygon",
    "triangle": "polygon",
    "square": "rectangle",
    "pentagon": "polygon",
    "hexagon": "polygon",
    "heptagon": "polygon",
    "octagon": "polygon",
    "nonagon": "polygon",
    "decagon": "polygon",
}

_POLYGON_SIDES_FROM_NAME: dict[str, int] = {
    "triangle": 3,
    "pentagon": 5,
    "hexagon": 6,
    "heptagon": 7,
    "octagon": 8,
    "nonagon": 9,
    "decagon": 10,
}


def _check_positive(value: Optional[float]) -> Optional[float]:
    if value is not None and value <= 0:
        raise ValueError("dimension must be > 0 mm when provided")
    return value


class EdgeTreatment(BaseModel):
    """A fillet or chamfer applied to a coherent set of edges on the base body.

    The set is described by a coarse selector (``edges``) rather than a
    list of individual edge IDs because that's what a vision model can
    reliably read off six orthographic views. The CadQuery builder
    translates each selector to ``result.edges("...").fillet(size_mm)``
    or ``.chamfer(size_mm)`` and applies treatments AFTER all features
    so they don't perturb feature face tags.
    """

    type: EdgeTreatmentType
    edges: EdgeSet
    size_mm: float = Field(
        ...,
        description="Fillet radius or chamfer leg length in mm.",
    )

    @field_validator("size_mm", mode="after")
    @classmethod
    def _positive(cls, v: float) -> float:
        return _check_positive(v)  # type: ignore[return-value]


class BaseBody(BaseModel):
    """The primary extruded body. One per part."""

    shape: Shape
    sides: Optional[int] = Field(
        None,
        ge=3,
        description=(
            "Number of sides when shape='polygon'. Required for polygons "
            "(defaults to 6 if omitted). Ignored for non-polygon shapes."
        ),
    )
    width_mm: float = Field(..., description="X axis dimension")
    depth_mm: float = Field(..., description="Y axis dimension")
    height_mm: float = Field(..., description="Z axis (extrude) dimension")
    sketch_plane: SketchPlane = "XY"

    edge_treatments: list[EdgeTreatment] = Field(
        default_factory=list,
        description=(
            "Optional fillets/chamfers applied to base body edges. "
            "Applied AFTER all features so they do not move the face "
            "tags features rely on."
        ),
    )

    @model_validator(mode="before")
    @classmethod
    def _normalise_shape_and_sides(cls, data):
        """Coerce shape-name drifts AND infer ``sides`` from polygon names.

        Claude often returns ``"octagon"`` instead of the enum
        ``"polygon"`` plus ``sides=8``. We fold those together here so
        the strict enum validator never trips on a known alias.
        """
        if not isinstance(data, dict):
            return data
        raw_shape = data.get("shape")
        if isinstance(raw_shape, str):
            raw_low = raw_shape.strip().lower()
            data["shape"] = _SHAPE_ALIASES.get(raw_low, raw_low)
            if (
                data["shape"] == "polygon"
                and data.get("sides") is None
                and raw_low in _POLYGON_SIDES_FROM_NAME
            ):
                data["sides"] = _POLYGON_SIDES_FROM_NAME[raw_low]
        return data

    @model_validator(mode="after")
    def _default_polygon_sides(self) -> "BaseBody":
        """Polygons without an explicit side count default to 6 (hexagon)."""
        if self.shape == "polygon" and self.sides is None:
            object.__setattr__(self, "sides", 6)
        return self

    @field_validator("width_mm", "depth_mm", "height_mm", mode="after")
    @classmethod
    def _positive(cls, v: float) -> float:
        return _check_positive(v)  # type: ignore[return-value]


class Feature(BaseModel):
    """A secondary feature applied on a face of the base body.

    ``sketch_plane`` is implied by ``face`` (XY for top/bottom, XZ for
    front/back, YZ for sides) but kept explicit so the CadQuery builder
    can pass it straight to ``cq.Workplane(...)``.
    """

    type: FeatureType
    face: Face
    shape: FeatureShape
    sketch_plane: SketchPlane

    diameter_mm: Optional[float] = None
    width_mm: Optional[float] = None
    depth_mm: Optional[float] = None
    height_mm: Optional[float] = Field(
        None,
        description="Boss height, or blind hole/pocket depth",
    )
    depth_type: Optional[DepthType] = None

    position_x: float = 0.0
    position_y: float = 0.0

    @field_validator(
        "diameter_mm", "width_mm", "depth_mm", "height_mm", mode="after"
    )
    @classmethod
    def _positive(cls, v: Optional[float]) -> Optional[float]:
        return _check_positive(v)


class PartDescription(BaseModel):
    """Top-level structured description returned by the vision step."""

    base: BaseBody
    features: list[Feature] = Field(default_factory=list)
    confidence: Confidence
    notes: Optional[str] = None


class ExecutionResult(BaseModel):
    """Placeholder for the CadQuery execution stage.

    Filled in by a later module; defined here so the public schema of
    ``backend.ai_infra`` is fixed from the start.
    """

    success: bool
    step_path: Optional[str] = None
    stl_path: Optional[str] = None
    cadquery_code: str
    error: Optional[str] = None
