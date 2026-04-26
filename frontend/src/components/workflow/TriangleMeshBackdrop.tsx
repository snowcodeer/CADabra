import { useEffect, useRef } from "react";

/* ============================================================
   TRIANGLE MESH BACKDROP — connected at rest, shatters on hover.

   Visual model:
     • At rest every triangle's 3 vertices sit exactly on a shared
       staggered grid, so the mesh reads as a single connected
       triangulation — neighbouring triangles share edges.
     • Near the cursor each triangle "detaches": its vertices drift
       AWAY from the shared grid points toward the triangle's own
       centroid-rooted positions, and the whole triangle is shoved
       outward by the cursor. Adjacent triangles (which used to
       share those vertices) drift toward THEIR own centroids, so
       a visible gap appears between them — the mesh fractures.
     • When the cursor leaves, every vertex springs back to its
       shared grid point and the mesh re-knits seamlessly.
   ============================================================ */

export function TriangleMeshBackdrop({
  /** Spacing between grid vertices, in CSS px. Smaller = denser. */
  spacing = 30,
  /** Stroke colour for the triangle edges (idle / far from cursor). */
  color = "rgba(71, 85, 105, 0.55)",
  /** Stroke colour for triangles fully detached under the cursor. */
  hotColor = "rgba(15, 23, 42, 0.95)",
  /** Cursor influence radius, CSS px. */
  radius = 220,
  /** Strength of cursor push on each detached triangle. */
  pushStrength = 520,
  /** How aggressively a triangle detaches from the grid near the cursor. */
  detachStrength = 1,
}: {
  spacing?: number;
  color?: string;
  hotColor?: string;
  radius?: number;
  pushStrength?: number;
  detachStrength?: number;
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const parent = canvas.parentElement;
    if (!parent) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    let w = 0;
    let h = 0;

    /**
     * Per-triangle state — flat Float32Array for cache friendliness.
     * Layout (18 floats per triangle):
     *   0..5   v0..v2 SHARED grid positions (the "home" — never changes)
     *          [v0x, v0y, v1x, v1y, v2x, v2y]
     *   6..11  current vertex positions     [x0, y0, x1, y1, x2, y2]
     *   12..13 centroid offset velocity     [cvx, cvy]   (rigid drift)
     *   14..15 centroid current offset      [cox, coy]   (drift from home)
     *
     * The "shared" home positions are identical for vertices that
     * neighbouring triangles share — that's why the mesh looks
     * connected at rest.
     */
    let tris = new Float32Array(0);
    let triCount = 0;

    const cursor = { x: -9999, y: -9999, active: false };

    const buildField = () => {
      const rect = parent.getBoundingClientRect();
      w = rect.width;
      h = rect.height;
      canvas.width = Math.round(w * dpr);
      canvas.height = Math.round(h * dpr);
      canvas.style.width = `${w}px`;
      canvas.style.height = `${h}px`;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

      const cols = Math.ceil(w / spacing) + 2;
      const rows = Math.ceil(h / spacing) + 2;

      // Build the staggered vertex grid first — 1 col/row of bleed.
      const grid = new Float32Array(cols * rows * 2);
      for (let r = 0; r < rows; r++) {
        for (let c = 0; c < cols; c++) {
          const i = (r * cols + c) * 2;
          const offset = r % 2 === 0 ? 0 : spacing / 2;
          // Tiny per-vertex jitter so the grid doesn't look mechanical.
          const jx = Math.sin(r * 9.13 + c * 4.77) * spacing * 0.12;
          const jy = Math.cos(r * 5.31 + c * 7.91) * spacing * 0.12;
          grid[i]     = (c - 1) * spacing + offset + jx;
          grid[i + 1] = (r - 1) * spacing * 0.866 + jy;
        }
      }

      // Walk every cell and emit two triangles per cell, both referencing
      // shared grid corners. The pattern alternates per row so triangles
      // tile cleanly with proper shared edges.
      const triList: number[] = [];
      const push = (i00: number, i10: number, i01: number) => {
        triList.push(
          grid[i00 * 2],     grid[i00 * 2 + 1],
          grid[i10 * 2],     grid[i10 * 2 + 1],
          grid[i01 * 2],     grid[i01 * 2 + 1],
        );
      };
      for (let r = 0; r < rows - 1; r++) {
        const evenRow = r % 2 === 0;
        for (let c = 0; c < cols - 1; c++) {
          const i00 = r * cols + c;
          const i10 = r * cols + c + 1;
          const i01 = (r + 1) * cols + c;
          const i11 = (r + 1) * cols + c + 1;
          if (evenRow) {
            push(i00, i10, i01);
            push(i10, i11, i01);
          } else {
            push(i00, i11, i01);
            push(i00, i10, i11);
          }
        }
      }

      triCount = triList.length / 6;
      tris = new Float32Array(triCount * 16);
      for (let n = 0; n < triCount; n++) {
        const src = n * 6;
        const dst = n * 16;
        // 0..5  shared home
        for (let k = 0; k < 6; k++) tris[dst + k] = triList[src + k];
        // 6..11 current = home (so the mesh starts perfectly connected)
        for (let k = 0; k < 6; k++) tris[dst + 6 + k] = triList[src + k];
        // 12..15 centroid offset state = 0
        tris[dst + 12] = 0; tris[dst + 13] = 0;
        tris[dst + 14] = 0; tris[dst + 15] = 0;
      }
    };

    buildField();
    const ro = new ResizeObserver(buildField);
    ro.observe(parent);

    const onMove = (e: MouseEvent) => {
      const rect = canvas.getBoundingClientRect();
      cursor.x = e.clientX - rect.left;
      cursor.y = e.clientY - rect.top;
      cursor.active = true;
    };
    const onLeave = () => { cursor.active = false; };
    parent.addEventListener("mousemove", onMove);
    parent.addEventListener("mouseleave", onLeave);

    let raf = 0;
    let last = performance.now();

    // Spring constants for the centroid offset (rigid drift of the
    // whole triangle). Higher SPRING = snappier detach + reattach.
    const C_SPRING = 22;
    const C_DAMP   = 7;

    // Spring constants for individual vertex → its target.
    // Target = home (when detach=0)  OR  home + (centroid offset)
    // scaled by detach amount near the cursor.
    const V_SPRING = 28;
    const V_DAMP   = 9;

    const R  = radius;
    const R2 = R * R;

    // Per-vertex velocity buffer (3 verts × 2 components per triangle)
    // and per-triangle detach amount (used for rendering colour).
    let vvel = new Float32Array(0);
    let detachAmt = new Float32Array(0);
    const ensureBuffers = () => {
      if (vvel.length !== triCount * 6) vvel = new Float32Array(triCount * 6);
      if (detachAmt.length !== triCount) detachAmt = new Float32Array(triCount);
    };

    const tick = (now: number) => {
      const dt = Math.min((now - last) / 1000, 0.05);
      last = now;
      ensureBuffers();

      const cActive = cursor.active;
      const cx = cursor.x;
      const cy = cursor.y;

      // ---- Per-triangle physics ----
      for (let n = 0; n < triCount; n++) {
        const o = n * 16;

        // Home (shared) vertices
        const h0x = tris[o],     h0y = tris[o + 1];
        const h1x = tris[o + 2], h1y = tris[o + 3];
        const h2x = tris[o + 4], h2y = tris[o + 5];

        // Home centroid
        const hcx = (h0x + h1x + h2x) / 3;
        const hcy = (h0y + h1y + h2y) / 3;

        // Detach amount based on home centroid distance to cursor
        // (using HOME so detachment doesn't run away once a tri starts moving).
        let detach = 0;
        let pushX = 0;
        let pushY = 0;
        if (cActive) {
          const dx = hcx - cx;
          const dy = hcy - cy;
          const d2 = dx * dx + dy * dy;
          if (d2 < R2) {
            const d = Math.sqrt(d2) || 0.0001;
            const falloff = 1 - d / R;          // 1 at centre, 0 at edge
            detach = falloff * detachStrength;  // 0..1
            pushX = (dx / d) * pushStrength * falloff;
            pushY = (dy / d) * pushStrength * falloff;
          }
        }

        detachAmt[n] = detach;

        // Centroid offset spring → drifts toward (pushX, pushY) when
        // detached, springs back to (0, 0) otherwise.
        const cox = tris[o + 14];
        const coy = tris[o + 15];
        const cvx = tris[o + 12];
        const cvy = tris[o + 13];
        const targetCox = (pushX / C_SPRING);   // steady-state offset under push
        const targetCoy = (pushY / C_SPRING);
        const cax = (targetCox - cox) * C_SPRING - cvx * C_DAMP;
        const cay = (targetCoy - coy) * C_SPRING - cvy * C_DAMP;
        tris[o + 12] = cvx + cax * dt;
        tris[o + 13] = cvy + cay * dt;
        tris[o + 14] = cox + tris[o + 12] * dt;
        tris[o + 15] = coy + tris[o + 13] * dt;

        // Each vertex's TARGET position blends:
        //   detach == 0  → exactly the shared home (mesh is connected)
        //   detach == 1  → home + centroid offset (rigid drift away from
        //                  shared neighbours, so this triangle separates)
        // Because neighbouring triangles compute their own targetCox/coy
        // using THEIR own centroids, the shared edges visibly split.
        const offX = tris[o + 14] * detach;
        const offY = tris[o + 15] * detach;

        // Vertex 0
        {
          const tx = h0x + offX;
          const ty = h0y + offY;
          const px = tris[o + 6];
          const py = tris[o + 7];
          const vx = vvel[n * 6];
          const vy = vvel[n * 6 + 1];
          const ax = (tx - px) * V_SPRING - vx * V_DAMP;
          const ay = (ty - py) * V_SPRING - vy * V_DAMP;
          vvel[n * 6]     = vx + ax * dt;
          vvel[n * 6 + 1] = vy + ay * dt;
          tris[o + 6] = px + vvel[n * 6]     * dt;
          tris[o + 7] = py + vvel[n * 6 + 1] * dt;
        }
        // Vertex 1
        {
          const tx = h1x + offX;
          const ty = h1y + offY;
          const px = tris[o + 8];
          const py = tris[o + 9];
          const vx = vvel[n * 6 + 2];
          const vy = vvel[n * 6 + 3];
          const ax = (tx - px) * V_SPRING - vx * V_DAMP;
          const ay = (ty - py) * V_SPRING - vy * V_DAMP;
          vvel[n * 6 + 2] = vx + ax * dt;
          vvel[n * 6 + 3] = vy + ay * dt;
          tris[o + 8] = px + vvel[n * 6 + 2] * dt;
          tris[o + 9] = py + vvel[n * 6 + 3] * dt;
        }
        // Vertex 2
        {
          const tx = h2x + offX;
          const ty = h2y + offY;
          const px = tris[o + 10];
          const py = tris[o + 11];
          const vx = vvel[n * 6 + 4];
          const vy = vvel[n * 6 + 5];
          const ax = (tx - px) * V_SPRING - vx * V_DAMP;
          const ay = (ty - py) * V_SPRING - vy * V_DAMP;
          vvel[n * 6 + 4] = vx + ax * dt;
          vvel[n * 6 + 5] = vy + ay * dt;
          tris[o + 10] = px + vvel[n * 6 + 4] * dt;
          tris[o + 11] = py + vvel[n * 6 + 5] * dt;
        }
      }

      // ---- Render ----
      ctx.clearRect(0, 0, w, h);

      // Pass 1: batched stroke for all "calm" triangles (detach < threshold).
      // One path = one draw call = fast.
      const HOT = 0.04;
      ctx.lineWidth = 1;
      ctx.strokeStyle = color;
      ctx.beginPath();
      for (let n = 0; n < triCount; n++) {
        if (detachAmt[n] >= HOT) continue;
        const o = n * 16;
        ctx.moveTo(tris[o + 6],  tris[o + 7]);
        ctx.lineTo(tris[o + 8],  tris[o + 9]);
        ctx.lineTo(tris[o + 10], tris[o + 11]);
        ctx.closePath();
      }
      ctx.stroke();

      // Pass 2: each "hot" triangle gets its own stroke colour interpolated
      // toward hotColor and a soft fill — both proportional to detach. This
      // makes the disrupted patch around the cursor visibly stand out.
      for (let n = 0; n < triCount; n++) {
        const d = detachAmt[n];
        if (d < HOT) continue;
        const o = n * 16;
        const x0 = tris[o + 6],  y0 = tris[o + 7];
        const x1 = tris[o + 8],  y1 = tris[o + 9];
        const x2 = tris[o + 10], y2 = tris[o + 11];

        // Linear blend hot → idle on the alpha channel using d.
        const strokeA = (0.55 + 0.4 * d).toFixed(3);
        const fillA   = (0.18 * d).toFixed(3);

        ctx.beginPath();
        ctx.moveTo(x0, y0);
        ctx.lineTo(x1, y1);
        ctx.lineTo(x2, y2);
        ctx.closePath();
        ctx.fillStyle = `rgba(71, 85, 105, ${fillA})`;
        ctx.fill();
        ctx.strokeStyle = `rgba(15, 23, 42, ${strokeA})`;
        ctx.lineWidth = 1 + d * 0.6;
        ctx.stroke();
      }

      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);

    return () => {
      cancelAnimationFrame(raf);
      ro.disconnect();
      parent.removeEventListener("mousemove", onMove);
      parent.removeEventListener("mouseleave", onLeave);
    };
  }, [spacing, color, radius, pushStrength, detachStrength]);

  return (
    <canvas
      ref={canvasRef}
      className="absolute inset-0 h-full w-full"
      aria-hidden
    />
  );
}
