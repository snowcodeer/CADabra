# CADabra

CADabra reverse-engineers 3D point clouds into editable parametric CAD code and exportable CAD files. The pipeline starts from a noisy point cloud (a stand-in for a real-world scan), reconstructs a topology-faithful surface mesh, renders it as orthographic depth+silhouette views, and then asks a vision LLM to author CadQuery (sketch + extrude) code that matches the views.

## Pipeline overview

Input: STL ground truth (DeepCAD subset). Output: clean orthographic views + a parametric CadQuery script.

```
ground-truth STL
      │
      │  virtual scan (Poisson disk sample + scanner-style noise)
      ▼
noisy point cloud
      │
      │  multi-strategy reconstruction (Poisson, Ball-Pivoting, Alpha-Shape)
      │  PCA-aligned envelope crop, statistical / radius outlier removal
      │  RANSAC plane snap for sharp corners and faces
      │  raycast-based through-hole detection (preserves real bores)
      │  edge-length-guarded hole fill (no long diagonal bridges)
      ▼
"raw recon" mesh (sharp edges preserved, real holes preserved)
      │
      │  6-axis orthographic ray-cast (depth + silhouette per face)
      ▼
geometry view PNG  ─►  vision-graded cohesion score (Claude Opus 4.7)
      │                tuner suggests RECON_PARAMS / SCORE_PARAMS /
      │                HOLE_FILL_PARAMS overrides per sample,
      │                rolls back if cohesion drops
      ▼
gpt-image-2 cleanup edit
      │  sharpens silhouettes, infills small surface noise,
      │  preserves the dark holes that signal real through-bores
      ▼
clean orthographic view PNG
      │
      │  Claude Opus 4.7 vision → CadQuery sketch+extrude script
      ▼
parametric CAD (.py) ─► CadQuery exec ─► STEP / STL
```

The frontend (`frontend-preview/deepcad-selector.html`) shows a per-sample 5-panel comparison: ground-truth STL, generated point cloud, raw reconstruction, geometry view (raw), and gpt-image-2 cleaned view.

## Reconstruction key ideas

- **Geometry preserves topological intent**, not watertightness. A real CAD bore that survives reconstruction as 1–2 boundary loops is more useful for downstream CAD inference than a forcibly closed mesh that fills the bore in.
- **Raycast through-hole detection** (`_detect_through_holes` in `scripts/reconstruct_meshes_from_pointclouds.py`) shoots probes along each boundary loop's best-fit normal. Loops that have mesh further along the axis (a tunnel, not a flat-face artifact) are marked `through_hole=True` and made permanently un-fillable for the rest of the pipeline.
- **Edge-length safeguard** (`_drop_long_fill_triangles`) drops any newly added fill triangle whose edges exceed `meshy_close_max_edge_frac × bbox_diag`, which kills the long diagonal bridges that aggressive hole-fill libraries are otherwise tempted to produce.
- **Hybrid vision tuning loop** (`scripts/vision_tune_recon.py`) runs a global pass that proposes universal `HOLE_FILL_PARAMS` improvements, then a per-sample cohesion pass that asks Claude Opus 4.7 to grade the side-by-side {point cloud, recon} views and suggest overrides. Best-of-N STLs are persisted by score.
- **Image-domain cleanup** (`scripts/synthesize_clean_views.py`) acknowledges that scan noise leaves small cosmetic gaps on faces. Instead of trying to repair them in 3D (which risks bridging real bores), the rendered orthographic grid is sent to OpenAI `gpt-image-2` with an edit prompt that explicitly preserves dark holes / white islands as real through-bores while sharpening silhouettes and infilling surface noise.

## Stack

- Backend: FastAPI + Open3D + PyVista + Anthropic SDK + OpenAI SDK + CadQuery execution
- Frontend: Vanilla JS + Three.js (CDN)

## Environment Variables

- `ANTHROPIC_API_KEY` (required for vision tuner + CadQuery generation)
- `OPENAI_API_KEY` (required for `gpt-image-2` view cleanup)
- `OUTPUT_DIR` (default `./outputs`)
- `SAMPLE_DATA_DIR` (default `./sample_data`)
- `MAX_UPLOAD_SIZE_MB` (default `10`)
- `CADQUERY_TIMEOUT_SECONDS` (default `15`)

## Run

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000
```

Serve frontend from `frontend-preview/` via any static server.

## Reproduce the test-sample pipeline

The hybrid tuner + image-gen cleanup is intentionally scoped to a small highlight set (`deepcadimg_000035`, `deepcadimg_002354`, `deepcadimg_117514`, `deepcadimg_128105`) to keep API spend bounded:

```bash
# from repo root, with .env populated
source backend/.venv/bin/activate
python scripts/vision_tune_recon.py --max-iters 2 --synth-quality high
```

Add `--no-synth-clean` to skip the `gpt-image-2` step. Add `--only deepcadimg_117514` to scope further.

## Known Constraints

- MVP handles prismatic mechanical parts only: planes, extrudes, cuts, cylindrical bosses
- Input geometry should be axis-aligned or near axis-aligned for best results
- No fillets, no chamfers, no organic surfaces, no assemblies
- LLM path works best on clean CAD-Recode-like inputs; noisy real scans degrade results
- Watertightness is **not** the primary metric. Reconstruction is graded on topology fidelity (real holes preserved) and on the cleanliness of the rendered orthographic views, since the views are what the downstream CAD-inference LLM actually sees.
