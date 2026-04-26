/** Shared types for the CADabra scene's extrusion model. */

export type FaceKey = "px" | "nx" | "py" | "ny" | "pz" | "nz";

export type Extrusion = Record<FaceKey, number>;

export const ZERO_EXTRUSION: Extrusion = {
  px: 0,
  nx: 0,
  py: 0,
  ny: 0,
  pz: 0,
  nz: 0,
};

/** Base brick body size (world units; 1 unit == 100mm). */
export const BASE_SIZE: [number, number, number] = [1.6, 0.7, 0.8];

/**
 * Compute the effective body dimensions, scale, and center offset given
 * a base size and an extrusion vector.
 */
export function computeBody(
  size: [number, number, number],
  ext: Extrusion,
) {
  const [w, h, d] = size;
  const sx = w + ext.px + ext.nx;
  const sy = h + ext.py + ext.ny;
  const sz = d + ext.pz + ext.nz;
  return {
    size: [sx, sy, sz] as [number, number, number],
    scale: [sx / w, sy / h, sz / d] as [number, number, number],
    offset: [
      (ext.px - ext.nx) / 2,
      (ext.py - ext.ny) / 2,
      (ext.pz - ext.nz) / 2,
    ] as [number, number, number],
  };
}
