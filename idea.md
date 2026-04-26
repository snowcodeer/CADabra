# CADabra

**Problem:** Scanning gives you a point cloud. Product work needs **parametric CAD** (editable solids, STEP/STL, dimensions). Bridging that gap today is mostly **manual** reverse engineering.

**What we built:** A pipeline from **noisy point cloud** to **CadQuery parametric code** and **exported geometry**. We reconstruct a mesh that keeps **real holes and sharp features** (we avoid filling through-bores and smearing intent just to get watertight glass), render **six orthographic views** so a vision model sees the part like a drawing, then generate **sketch and extrude** style code. A FastAPI backend runs **preprocess → render → LLM → execute**; the React app includes a **workflow** and interactive **/demo** (Three.js) to show scan-to-CAD and editability.

**Stack:** FastAPI, Open3D, PyVista, CadQuery, Anthropic / OpenAI where needed, Vite + React + R3F.

**Current limits:** Prismatic parts and axis-aligned or near axis-aligned geometry work best. No fillets, organics, or assemblies in this MVP. Heavily noisy real-world scans are harder than clean synthetic CAD-derived inputs.

**Takeaway:** We turn capture into **parametric code and exports**, not a dead mesh, and we bias the middle of the stack so **topology stays interpretable** for the LLM and the engineer.
