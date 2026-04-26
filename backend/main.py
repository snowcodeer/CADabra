from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env")

import logfire

logfire.configure()
logfire.instrument_pydantic()

import open3d as o3d
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from models import PipelineResult, ProcessRequest, SampleInfo
from pipeline.cad_executor import execute_cadquery
from pipeline.llm_client import infer_cadquery
from pipeline.preprocess import preprocess_mesh
from pipeline.render_views import render_views_grid
from validation.cad_validator import validate_outputs

OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "./outputs")).resolve()
SAMPLE_DATA_DIR = Path(os.getenv("SAMPLE_DATA_DIR", "./sample_data")).resolve()
MAX_UPLOAD_SIZE_MB = int(os.getenv("MAX_UPLOAD_SIZE_MB", "10"))

app = FastAPI(title="CADabra")
logfire.instrument_fastapi(app)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _load_manifest() -> list[dict]:
    manifest = SAMPLE_DATA_DIR / "manifest.json"
    if not manifest.exists():
        return []
    return json.loads(manifest.read_text())


def _error(stage: str, message: str, status_code: int = 400):
    return JSONResponse(
        status_code=status_code,
        content={"success": False, "error": message, "stage": stage},
    )


@app.middleware("http")
async def log_request(request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    print(f"ts={time.time():.3f} route={request.url.path} duration_ms={elapsed_ms:.2f}")
    return response


@app.get("/health")
def health():
    return {"status": "ok", "stages": ["preprocess", "render", "llm", "execute"]}


@app.get("/samples")
def samples():
    rows = _load_manifest()
    result = [
        SampleInfo(
            sample_id=row["sample_id"],
            display_name=row.get("display_name", row["sample_id"].replace("_", " ").title()),
            stl_filename=row["stl_file"],
            point_cloud_filename=row.get("ply_file"),
            ground_truth_filename=row.get("ground_truth_file"),
        ).model_dump()
        for row in rows
    ]
    return result


@app.get("/outputs/{filename}")
def outputs(filename: str):
    file_path = OUTPUT_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(file_path)


@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    suffix = Path(file.filename).suffix.lower()
    if suffix not in {".ply", ".xyz", ".stl"}:
        return _error("upload", "Only .ply, .xyz, or .stl files are accepted")

    content = await file.read()
    if len(content) > MAX_UPLOAD_SIZE_MB * 1024 * 1024:
        return _error("upload", f"File exceeds {MAX_UPLOAD_SIZE_MB}MB limit")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    file_id = str(uuid.uuid4())
    save_path = OUTPUT_DIR / f"{file_id}{suffix}"
    save_path.write_bytes(content)
    return {"filename": save_path.name}


@app.post("/process")
def process(payload: ProcessRequest):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if payload.source == "sample":
        manifest = _load_manifest()
        row = next((r for r in manifest if r["sample_id"] == payload.sample_id), None)
        if row is None:
            return _error("input", f"Unknown sample_id: {payload.sample_id}")
        input_id = payload.sample_id or "sample"
        mesh_path = SAMPLE_DATA_DIR / row["stl_file"]
        ground_truth_code = (SAMPLE_DATA_DIR / row["ground_truth_file"]).read_text() if row.get("ground_truth_file") else None
    else:
        if not payload.filename:
            return _error("input", "filename is required for upload source")
        input_id = Path(payload.filename).stem
        mesh_path = OUTPUT_DIR / payload.filename
        ground_truth_code = None

    if not mesh_path.exists():
        return _error("input", f"Input file missing: {mesh_path}")

    try:
        mesh, bbox = preprocess_mesh(mesh_path)
    except Exception as exc:
        return _error("preprocess", str(exc))

    try:
        render_grid = render_views_grid(mesh, OUTPUT_DIR, input_id)
    except Exception as exc:
        return _error("render", str(exc))

    bbox_dict = {"x_mm": round(bbox[0], 3), "y_mm": round(bbox[1], 3), "z_mm": round(bbox[2], 3)}

    try:
        llm_output = infer_cadquery(render_grid, bbox_dict)
    except Exception as exc:
        return _error("llm", str(exc))

    exec_result = execute_cadquery(llm_output.cadquery_code, OUTPUT_DIR, input_id)
    if not exec_result["success"]:
        return _error("execute", str(exec_result["error"]))

    try:
        validation = validate_outputs(exec_result["step_path"], exec_result["stl_path"])
    except Exception as exc:
        return _error("validate", str(exc))

    step_name = Path(str(exec_result["step_path"])) .name
    stl_name = Path(str(exec_result["stl_path"])) .name

    result = PipelineResult(
        input_id=input_id,
        success=True,
        render_grid_url=f"/outputs/{render_grid.name}",
        cadquery_code=llm_output.cadquery_code,
        ground_truth_cadquery_code=ground_truth_code,
        step_url=f"/outputs/{step_name}",
        stl_url=f"/outputs/{stl_name}",
        validation=validation,
        llm_reasoning=llm_output.reasoning,
        confidence=llm_output.confidence,
        error=None,
    )
    return result.model_dump()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
