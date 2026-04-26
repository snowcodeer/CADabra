/**
 * Typed wrappers for the CADabra FastAPI backend.
 *
 * Mirrors the Pydantic models in backend/models.py so a TypeScript caller
 * can rely on field names matching the wire payload exactly.
 *
 * Endpoint surface (see backend/main.py):
 *   GET  /health
 *   GET  /samples         -> SampleInfo[]
 *   GET  /outputs/{name}  -> file (STL/PNG/STEP)
 *   POST /upload          -> { filename }
 *   POST /process         -> PipelineResult
 */

const DEFAULT_BASE = "http://localhost:8000";

/** Resolve once at module load so consumers can override via VITE_API_BASE. */
export const API_BASE: string =
  (import.meta.env.VITE_API_BASE as string | undefined)?.replace(/\/$/, "") ??
  DEFAULT_BASE;

export type SampleInfo = {
  sample_id: string;
  display_name: string;
  stl_filename: string;
  point_cloud_filename: string | null;
  ground_truth_filename: string | null;
};

export type ValidationResult = {
  executes: boolean;
  produces_solid: boolean;
  is_watertight: boolean;
  vertex_count: number | null;
  face_count: number | null;
  bounding_box_mm: [number, number, number] | null;
  error: string | null;
};

export type PipelineResult = {
  input_id: string;
  success: boolean;
  render_grid_url: string | null;
  cadquery_code: string | null;
  ground_truth_cadquery_code: string | null;
  step_url: string | null;
  stl_url: string | null;
  validation: ValidationResult | null;
  llm_reasoning: string | null;
  confidence: "high" | "medium" | "low" | null;
  error: string | null;
};

export type ProcessRequest = {
  source: "sample" | "upload";
  sample_id?: string | null;
  filename?: string | null;
};

/** Server returns { success: false, stage, error } on stage failure. */
export type StageError = {
  success: false;
  stage: string;
  error: string;
};

export class PipelineError extends Error {
  stage: string;
  constructor(stage: string, message: string) {
    super(`[${stage}] ${message}`);
    this.stage = stage;
    this.name = "PipelineError";
  }
}

async function jsonOrThrow<T>(resp: Response): Promise<T> {
  const text = await resp.text();
  let payload: unknown;
  try {
    payload = text ? JSON.parse(text) : null;
  } catch {
    throw new Error(`Non-JSON response (${resp.status}): ${text.slice(0, 200)}`);
  }
  if (!resp.ok) {
    if (
      payload &&
      typeof payload === "object" &&
      "error" in payload &&
      "stage" in payload
    ) {
      const e = payload as StageError;
      throw new PipelineError(e.stage, e.error);
    }
    throw new Error(`HTTP ${resp.status}: ${text.slice(0, 200)}`);
  }
  return payload as T;
}

export async function getHealth(signal?: AbortSignal): Promise<{ status: string; stages: string[] }> {
  const resp = await fetch(`${API_BASE}/health`, { signal });
  return jsonOrThrow(resp);
}

export async function getSamples(signal?: AbortSignal): Promise<SampleInfo[]> {
  const resp = await fetch(`${API_BASE}/samples`, { signal });
  return jsonOrThrow<SampleInfo[]>(resp);
}

export async function processSample(
  sampleId: string,
  signal?: AbortSignal,
): Promise<PipelineResult> {
  const body: ProcessRequest = { source: "sample", sample_id: sampleId };
  const resp = await fetch(`${API_BASE}/process`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
    signal,
  });
  return jsonOrThrow<PipelineResult>(resp);
}

export async function uploadFile(
  file: File,
  signal?: AbortSignal,
): Promise<{ filename: string }> {
  const body = new FormData();
  body.append("file", file);
  const resp = await fetch(`${API_BASE}/upload`, { method: "POST", body, signal });
  return jsonOrThrow<{ filename: string }>(resp);
}

export async function processUpload(
  filename: string,
  signal?: AbortSignal,
): Promise<PipelineResult> {
  const body: ProcessRequest = { source: "upload", filename };
  const resp = await fetch(`${API_BASE}/process`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
    signal,
  });
  return jsonOrThrow<PipelineResult>(resp);
}

/** Resolve a backend-relative URL (e.g. render_grid_url) to a fetchable URL. */
export function resolveOutputUrl(relative: string | null | undefined): string | null {
  if (!relative) return null;
  if (relative.startsWith("http://") || relative.startsWith("https://")) {
    return relative;
  }
  return `${API_BASE}${relative.startsWith("/") ? "" : "/"}${relative}`;
}
