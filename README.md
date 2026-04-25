# CADabra

CADabra converts 3D geometry into editable parametric CAD code and exportable CAD files.

## Stack
- Backend: FastAPI + Open3D + PyVista + Anthropic SDK + CadQuery execution
- Frontend: Vanilla JS + Three.js (CDN)

## Environment Variables
- `ANTHROPIC_API_KEY` (required)
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

Serve frontend from `frontend/` via any static server.

## Known Constraints
- MVP handles prismatic mechanical parts only: planes, extrudes, cuts, cylindrical bosses
- Input geometry should be axis-aligned or near axis-aligned for best results
- No fillets, no chamfers, no organic surfaces, no assemblies
- LLM path works best on clean CAD-Recode-like inputs; noisy real scans degrade results
- Watertight validation may fail on complex topology; this is acceptable for MVP
