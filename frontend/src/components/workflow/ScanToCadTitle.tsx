import { useEffect, useMemo, useRef, useState } from "react";
import { Canvas } from "@react-three/fiber";
import * as THREE from "three";

/**
 * "SCAN to CAD" lockup.
 *
 * - SCAN — rendered as a 3D POINT CLOUD. We rasterise the word into an
 *   offscreen canvas, sample filled pixels on a jittered grid, then draw
 *   each sample as a small dot in an SVG. A second, offset+faded copy
 *   of the dot field is drawn behind to imply scan depth.
 * - to   — plain Space Grotesk at the same cap height, low-emphasis.
 * - CAD  — rendered as a 3D WIREFRAME. Outlined glyphs with connecting
 *   "depth ribs" between a front face and a back face, plus thin internal
 *   construction lines to read as a wireframe model.
 */

// ---------- shared font loading guard ----------

function useFontsReady(families: string[]) {
  const [ready, setReady] = useState(false);
  useEffect(() => {
    let cancelled = false;
    const anyDoc = document as unknown as { fonts?: { load: (s: string) => Promise<unknown>; ready: Promise<unknown> } };
    if (!anyDoc.fonts) {
      setReady(true);
      return;
    }
    Promise.all(families.map((f) => anyDoc.fonts!.load(`64px ${f}`)))
      .then(() => anyDoc.fonts!.ready)
      .then(() => {
        if (!cancelled) setReady(true);
      })
      .catch(() => {
        if (!cancelled) setReady(true);
      });
    return () => {
      cancelled = true;
    };
  }, [families.join("|")]);
  return ready;
}

// ---------- SCAN: real 3D point cloud ----------

/** Inner three.js cloud — receives a Float32Array of XYZ positions and
 *  renders them as round, slightly-shaded points. Static, face-on. */
function PointCloudMesh({
  positions,
  pointSize,
  color,
}: {
  positions: Float32Array;
  pointSize: number;
  color: string;
}) {
  const geom = useMemo(() => {
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.BufferAttribute(positions, 3));
    g.computeBoundingSphere();
    return g;
  }, [positions]);

  return (
    <points geometry={geom}>
      <pointsMaterial
        color={color}
        size={pointSize}
        sizeAttenuation={false}
        transparent
        opacity={0.95}
        depthWrite={false}
      />
    </points>
  );
}

function PointCloudText({
  text,
  fontFamily,
  fontWeight = 400,
  fontSize,
  letterSpacing = "0.04em",
  /** Pixel pitch of the front/back face sampling grid. Smaller = denser. */
  pitch = 2,
  dotColor = "#2a2f3a",
  /** World-space depth of the extrusion (relative to text height). */
  depthRatio = 0.32,
  /** Number of intermediate slices between front and back for side walls. */
  depthSlices = 14,
}: {
  text: string;
  fontFamily: string;
  fontWeight?: number;
  fontSize: number;
  letterSpacing?: string;
  pitch?: number;
  dotColor?: string;
  depthRatio?: number;
  depthSlices?: number;
}) {
  const ready = useFontsReady([fontFamily.split(",")[0].replace(/['"]/g, "").trim()]);
  const measureRef = useRef<HTMLSpanElement | null>(null);
  const [size, setSize] = useState<{ w: number; h: number } | null>(null);

  useEffect(() => {
    if (!ready || !measureRef.current) return;
    const r = measureRef.current.getBoundingClientRect();
    setSize({ w: Math.ceil(r.width), h: Math.ceil(r.height) });
  }, [ready, text, fontFamily, fontWeight, fontSize, letterSpacing]);

  // Build a true 3D point cloud:
  //   - front + back face: every filled pixel on a dense grid
  //   - side walls: silhouette pixels duplicated across N depth slices
  const positions = useMemo<Float32Array | null>(() => {
    if (!size) return null;
    const SCALE = 2; // supersample for crisper glyph edges
    const W = (size.w + 16) * SCALE;
    const H = (size.h + 16) * SCALE;
    const canvas = document.createElement("canvas");
    canvas.width = W;
    canvas.height = H;
    const ctx = canvas.getContext("2d");
    if (!ctx) return new Float32Array();
    ctx.clearRect(0, 0, W, H);
    ctx.fillStyle = "#000";
    ctx.textBaseline = "alphabetic";
    ctx.font = `${fontWeight} ${fontSize * SCALE}px ${fontFamily}`;
    const lsEm = parseFloat(letterSpacing) || 0;
    const lsPx = lsEm * fontSize * SCALE;
    let x = 8 * SCALE;
    const baseY = 8 * SCALE + size.h * SCALE * 0.82;
    for (const ch of text) {
      ctx.fillText(ch, x, baseY);
      x += ctx.measureText(ch).width + lsPx;
    }
    const data = ctx.getImageData(0, 0, W, H).data;
    const filled = (px: number, py: number) =>
      px >= 0 && py >= 0 && px < W && py < H && data[(py * W + px) * 4 + 3] > 128;

    const step = Math.max(1, Math.round(pitch * SCALE));
    const cx = W / 2;
    const cy = H / 2;
    const toWorld = 1 / SCALE; // map supersampled px -> CSS px
    const depthPx = size.h * depthRatio;
    const halfD = depthPx / 2;

    const points: number[] = [];

    // 1) Front + back face fills.
    for (let py = 0; py < H; py += step) {
      for (let px = 0; px < W; px += step) {
        if (!filled(px, py)) continue;
        const wx = (px - cx) * toWorld;
        const wy = -(py - cy) * toWorld;
        const jx = (Math.random() - 0.5) * step * 0.6 * toWorld;
        const jy = (Math.random() - 0.5) * step * 0.6 * toWorld;
        points.push(wx + jx, wy + jy, halfD);
        points.push(wx + jx, wy + jy, -halfD);
      }
    }

    // 2) Side walls.
    const edgeStep = Math.max(1, Math.round(step * 0.6));
    const slices = Math.max(2, depthSlices);
    for (let py = 0; py < H; py += edgeStep) {
      for (let px = 0; px < W; px += edgeStep) {
        if (!filled(px, py)) continue;
        if (
          filled(px - 1, py) &&
          filled(px + 1, py) &&
          filled(px, py - 1) &&
          filled(px, py + 1)
        ) {
          continue;
        }
        const wx = (px - cx) * toWorld;
        const wy = -(py - cy) * toWorld;
        for (let i = 0; i < slices; i++) {
          const t = i / (slices - 1);
          const z = -halfD + depthPx * t;
          const jx = (Math.random() - 0.5) * 0.5;
          const jy = (Math.random() - 0.5) * 0.5;
          points.push(wx + jx, wy + jy, z);
        }
      }
    }

    return new Float32Array(points);
  }, [size, text, fontFamily, fontWeight, fontSize, letterSpacing, pitch, depthRatio, depthSlices]);

  const padX = size ? Math.round(size.h * depthRatio * 0.6) : 0;
  const padY = size ? Math.round(size.h * 0.15) : 0;
  const boxW = size ? size.w + padX * 2 : 0;
  const boxH = size ? size.h + padY * 2 : 0;
  const orthoBounds = size
    ? { left: -boxW / 2, right: boxW / 2, top: boxH / 2, bottom: -boxH / 2 }
    : null;

  return (
    <span
      style={{
        position: "relative",
        display: "inline-block",
        lineHeight: 1,
        verticalAlign: "baseline",
      }}
    >
      <span
        ref={measureRef}
        aria-hidden
        style={{
          position: "absolute",
          visibility: "hidden",
          whiteSpace: "pre",
          fontFamily,
          fontWeight,
          fontSize,
          letterSpacing,
          lineHeight: 1,
        }}
      >
        {text}
      </span>

      <span
        aria-hidden
        style={{
          display: "inline-block",
          width: size ? size.w : "auto",
          height: size ? size.h : fontSize,
        }}
      >
        {!size ? text : ""}
      </span>

      {size && positions && orthoBounds && (
        <div
          aria-label={text}
          style={{
            position: "absolute",
            left: -padX,
            top: -padY,
            width: boxW,
            height: boxH,
            pointerEvents: "none",
          }}
        >
          <Canvas
            orthographic
            dpr={[1, 2]}
            camera={{
              position: [0, 0, 1000],
              near: 0.1,
              far: 5000,
              zoom: 1,
              left: orthoBounds.left,
              right: orthoBounds.right,
              top: orthoBounds.top,
              bottom: orthoBounds.bottom,
            }}
            gl={{ antialias: true, alpha: true }}
            style={{ background: "transparent" }}
          >
            <PointCloudMesh
              positions={positions}
              pointSize={Math.max(1.4, fontSize * 0.022)}
              color={dotColor}
            />
          </Canvas>
        </div>
      )}
    </span>
  );
}

// ---------- CAD: wireframe ----------

function WireframeText({
  text,
  fontFamily,
  fontWeight = 800,
  fontSize,
  letterSpacing = "-0.02em",
  strokeColor = "#2a2f3a",
  strokeWidth = 1.6,
  /** Pixel offset for the back face. */
  depth = 10,
  /** Pixel spacing between depth ribs along glyph contours. */
  ribPitch = 10,
}: {
  text: string;
  fontFamily: string;
  fontWeight?: number;
  fontSize: number;
  letterSpacing?: string;
  strokeColor?: string;
  strokeWidth?: number;
  depth?: number;
  ribPitch?: number;
}) {
  const ready = useFontsReady([fontFamily.split(",")[0].replace(/['"]/g, "").trim()]);
  const measureRef = useRef<HTMLSpanElement | null>(null);
  const [size, setSize] = useState<{ w: number; h: number } | null>(null);

  useEffect(() => {
    if (!ready || !measureRef.current) return;
    const r = measureRef.current.getBoundingClientRect();
    setSize({ w: Math.ceil(r.width), h: Math.ceil(r.height) });
  }, [ready, text, fontFamily, fontWeight, fontSize, letterSpacing]);

  // Build "depth ribs": find points along the glyph silhouette
  // (pixels that are filled but have at least one empty 4-neighbour)
  // and emit a connector from each sampled edge point to its
  // back-face counterpart, translated by (depth, depth).
  const ribs = useMemo(() => {
    if (!size) return [] as Array<{ x: number; y: number }>;
    const pad = depth + 6;
    const W = size.w + pad * 2;
    const H = size.h + pad * 2;
    const canvas = document.createElement("canvas");
    canvas.width = W;
    canvas.height = H;
    const ctx = canvas.getContext("2d");
    if (!ctx) return [];
    ctx.clearRect(0, 0, W, H);
    ctx.fillStyle = "#000";
    ctx.textBaseline = "alphabetic";
    ctx.font = `${fontWeight} ${fontSize}px ${fontFamily}`;
    const lsEm = parseFloat(letterSpacing) || 0;
    const lsPx = lsEm * fontSize;
    let x = pad;
    const y = pad + size.h * 0.82;
    for (const ch of text) {
      ctx.fillText(ch, x, y);
      x += ctx.measureText(ch).width + lsPx;
    }
    const data = ctx.getImageData(0, 0, W, H).data;
    const filled = (px: number, py: number) => {
      if (px < 0 || py < 0 || px >= W || py >= H) return false;
      return data[(py * W + px) * 4 + 3] > 128;
    };

    // Collect all edge pixels.
    const edges: Array<{ x: number; y: number }> = [];
    for (let py = 0; py < H; py++) {
      for (let px = 0; px < W; px++) {
        if (!filled(px, py)) continue;
        if (
          !filled(px - 1, py) ||
          !filled(px + 1, py) ||
          !filled(px, py - 1) ||
          !filled(px, py + 1)
        ) {
          edges.push({ x: px, y: py });
        }
      }
    }

    // Subsample edges to roughly ribPitch spacing so we don't draw a
    // line for every pixel.
    if (edges.length === 0) return edges;
    const taken: Array<{ x: number; y: number }> = [];
    const step = Math.max(1, Math.round(ribPitch));
    for (let i = 0; i < edges.length; i += step) taken.push(edges[i]);
    return taken;
  }, [size, text, fontFamily, fontWeight, fontSize, letterSpacing, ribPitch, depth]);

  const pad = depth + 6;
  const W = size ? size.w + pad * 2 : 0;
  const H = size ? size.h + pad * 2 : 0;

  return (
    <span
      style={{
        position: "relative",
        display: "inline-block",
        lineHeight: 1,
        verticalAlign: "baseline",
      }}
    >
      <span
        ref={measureRef}
        aria-hidden
        style={{
          position: "absolute",
          visibility: "hidden",
          whiteSpace: "pre",
          fontFamily,
          fontWeight,
          fontSize,
          letterSpacing,
          lineHeight: 1,
        }}
      >
        {text}
      </span>

      <span
        aria-hidden
        style={{
          display: "inline-block",
          width: size ? size.w : "auto",
          height: size ? size.h : fontSize,
        }}
      >
        {!size ? text : ""}
      </span>

      {size && (
        <svg
          aria-label={text}
          width={W}
          height={H}
          viewBox={`0 0 ${W} ${H}`}
          style={{
            position: "absolute",
            left: -pad,
            top: -pad,
            pointerEvents: "none",
            overflow: "visible",
          }}
        >
          {/* Back face — outlined glyphs, offset, faded. */}
          <g
            transform={`translate(${depth} ${depth})`}
            style={{ fontFamily, fontWeight, fontSize }}
          >
            <text
              x={pad}
              y={pad + size.h * 0.82}
              fill="none"
              stroke={strokeColor}
              strokeWidth={strokeWidth * 0.75}
              opacity={0.45}
              letterSpacing={letterSpacing}
              style={{ fontFamily, fontWeight, fontSize }}
            >
              {text}
            </text>
          </g>

          {/* Depth ribs — connect each silhouette point on the front
              face to its translated counterpart on the back face. */}
          <g
            stroke={strokeColor}
            strokeWidth={strokeWidth * 0.55}
            opacity={0.55}
            strokeLinecap="round"
          >
            {ribs.map((p, i) => (
              <line
                key={i}
                x1={p.x}
                y1={p.y}
                x2={p.x + depth}
                y2={p.y + depth}
              />
            ))}
          </g>

          {/* Front face — outlined glyphs. */}
          <text
            x={pad}
            y={pad + size.h * 0.82}
            fill="none"
            stroke={strokeColor}
            strokeWidth={strokeWidth}
            letterSpacing={letterSpacing}
            style={{ fontFamily, fontWeight, fontSize }}
          >
            {text}
          </text>
        </svg>
      )}
    </span>
  );
}

// ---------- Public component ----------

export function ScanToCadTitle({
  className = "",
  scale = 1,
}: {
  className?: string;
  scale?: number;
}) {
  const size = Math.round(84 * scale);
  const ink = "#2a2f3a";
  return (
    <div
      className={className}
      style={{
        display: "flex",
        alignItems: "baseline",
        justifyContent: "flex-start",
        gap: Math.round(18 * scale),
        lineHeight: 1,
      }}
    >
      <PointCloudText
        text="SCAN"
        fontFamily="'Workbench', 'Major Mono Display', ui-monospace, monospace"
        fontWeight={400}
        fontSize={size}
        letterSpacing="0.04em"
        pitch={2}
        dotColor={ink}
        depthRatio={0.32}
        depthSlices={16}
      />

      <span
        style={{
          fontFamily: "'Space Grotesk', 'Inter', sans-serif",
          fontSize: Math.round(size * 0.7),
          fontWeight: 300,
          letterSpacing: "-0.02em",
          opacity: 0.7,
          color: ink,
          // Nudge upward so the "to" cap-height visually aligns with
          // the dotted SCAN glyphs (which sit slightly higher).
          position: "relative",
          top: Math.round(-size * 0.22),
        }}
      >
        to
      </span>

      <WireframeText
        text="CAD"
        fontFamily="'Bricolage Grotesque', 'Space Grotesk', 'Inter', sans-serif"
        fontWeight={800}
        fontSize={size}
        letterSpacing="-0.02em"
        strokeColor={ink}
        strokeWidth={Math.max(1.4, size * 0.022)}
        depth={Math.max(2, Math.round(3 * scale))}
        ribPitch={Math.max(4, Math.round(size * 0.06))}
      />
    </div>
  );
}
