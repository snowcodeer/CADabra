import { Suspense, useEffect, useMemo, useRef, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import {
  ArrowLeft,
  ArrowRight,
  Upload,
} from "lucide-react";
import * as THREE from "three";
import { Canvas, useFrame, useThree } from "@react-three/fiber";
import { Edges } from "@react-three/drei";
import { CanvasReflowText } from "@/components/workflow/CanvasReflowText";
import { ScanToCadTitle } from "@/components/workflow/ScanToCadTitle";
import { LegoPodiumScene } from "@/components/workflow/LegoPodiumScene";




/* ============================================================
   WORKFLOW PAGE
   A staged mock of the CADabra pipeline. The user is walked
   through 5 phases:
     0  Idle / start prompt
     1  Upload .ply (fake progress bar)
     2  Cleaning the cloud (animated point cloud + log lines)
     3  Reconstructing triangles (cloud morphs into mesh)
     4  Rendering 6 orthographic angles (image grid reveal)
     5  Done — CTA to /demo
   All data is mocked. The 3D content is generated procedurally
   so the page has zero binary dependencies.
   ============================================================ */

type Stage = 0 | 1 | 2 | 3 | 4 | 5;

const STAGE_LABELS: Record<Stage, string> = {
  0: "Awaiting scan",
  1: "Uploading point cloud",
  2: "Cleaning · denoising",
  3: "Reconstructing surface",
  4: "Unfolding orthographic views",
  5: "Reconstruction complete",
};


/* ---------------- Procedural geometry helpers ---------------- */

/**
 * Build a Fibonacci point cloud on a slightly bumpy "object" — used
 * as the raw scan input. The shape is a deformed sphere so it reads
 * as a real scanned artefact rather than a perfect ball.
 */
function useScanPoints(count: number) {
  // Base unit-sphere positions (Fibonacci distribution) plus the
  // spherical coords we need to drive a sound-wave ripple per point.
  return useMemo(() => {
    const base = new Float32Array(count * 3);
    const phis = new Float32Array(count);
    const thetas = new Float32Array(count);
    for (let i = 0; i < count; i++) {
      const phi = Math.acos(1 - (2 * (i + 0.5)) / count);
      const theta = Math.PI * (1 + Math.sqrt(5)) * i;
      phis[i] = phi;
      thetas[i] = theta;
      base[i * 3] = Math.cos(theta) * Math.sin(phi);
      base[i * 3 + 1] = Math.sin(theta) * Math.sin(phi);
      base[i * 3 + 2] = Math.cos(phi);
    }
    return { base, phis, thetas, count };
  }, [count]);
}

/* ---------------- Reconstructed shape source ----------------
 * Single source of truth for the "reconstructed" mesh that the rest
 * of the pipeline shows. Right now this returns a mock sphere; once
 * real .ply reconstruction lands, swap the body of this hook to
 * return a BufferGeometry built from the user's point cloud
 * (e.g. via Poisson reconstruction). Every downstream component
 * (final mesh, unfolded-cube view tiles, fold/spin showcase) will
 * automatically pick up the new shape — no other code changes needed.
 */
function useReconstructedGeometry(): THREE.BufferGeometry {
  return useMemo(() => {
    // MOCK: a sphere. Replace with parsed .ply mesh in the future.
    return new THREE.SphereGeometry(1.35, 64, 48);
  }, []);
}

/* ============================================================
   AMBIENT POINT-CLOUD BACKGROUND
   Drifting nebula of points used as the live backdrop for the
/* ============================================================
   AMBIENT POINT CLOUD — universe / starfield vibes
   A volumetric scatter of small dots with depth fade. The whole
   field gently rotates; individual dots breathe in place.

   Interaction: a `cursorWorld` ref (mouse position projected onto
   the camera plane) pushes nearby dots radially outward, then
   they spring back to their home positions. Same field powers
   stages 0 and 1; in `streaming` mode dots are also pulled toward
   the centre, where they recycle, selling "data flowing in".
   ============================================================ */
const cursorWorld = { x: 9999, y: 9999, active: false };

function AmbientCloud({ streaming = false }: { streaming?: boolean }) {
  const pointsRef = useRef<THREE.Points>(null);
  const matRef = useRef<THREE.PointsMaterial>(null);
  const COUNT = 7000;

  // Volumetric scatter — random points inside a wide, shallow box so
  // the field reads as a slab of universe rather than a sphere.
  const { positions, seeds } = useMemo(() => {
    const pos = new Float32Array(COUNT * 3);
    const sd = new Float32Array(COUNT);
    for (let i = 0; i < COUNT; i++) {
      // Bias toward the edges by sqrt-distributing radius — keeps the
      // centre breathable so foreground UI stays legible.
      const u = Math.random();
      const r = Math.sqrt(u) * 6.5;
      const a = Math.random() * Math.PI * 2;
      pos[i * 3]     = Math.cos(a) * r * 1.35;     // wide X
      pos[i * 3 + 1] = Math.sin(a) * r * 0.85;     // shorter Y
      pos[i * 3 + 2] = (Math.random() - 0.5) * 4;  // shallow Z
      sd[i] = Math.random();
    }
    return { positions: pos, seeds: sd };
  }, []);

  const home = useMemo(() => positions.slice(), [positions]);
  // Per-point velocity so cursor displacement decays smoothly.
  const vel = useMemo(() => new Float32Array(COUNT * 3), []);

  useFrame((state, delta) => {
    if (!pointsRef.current) return;
    const t = state.clock.getElapsedTime();
    pointsRef.current.rotation.y = t * 0.02;

    const attr = pointsRef.current.geometry.getAttribute(
      "position",
    ) as THREE.BufferAttribute;
    const arr = attr.array as Float32Array;
    const dt = Math.min(delta, 0.05);

    // Cursor push parameters
    const cx = cursorWorld.x;
    const cy = cursorWorld.y;
    const cursorActive = cursorWorld.active;
    const RADIUS = 1.6;
    const RADIUS2 = RADIUS * RADIUS;
    const PUSH = 38; // peak acceleration at cursor centre

    // Spring-back + damping
    const SPRING = 7;
    const DAMP   = 4;

    const streamPull = streaming ? 0.45 : 0;

    for (let i = 0; i < COUNT; i++) {
      const ix = i * 3;
      const hx = home[ix];
      const hy = home[ix + 1];
      const hz = home[ix + 2];

      let px = arr[ix];
      let py = arr[ix + 1];
      let pz = arr[ix + 2];

      // Subtle breathing — keeps the field alive even with no cursor.
      const breathe = Math.sin(t * 0.6 + seeds[i] * 6.28) * 0.015;

      // Spring back toward home
      let ax = (hx - px) * SPRING - vel[ix]     * DAMP;
      let ay = (hy - py) * SPRING - vel[ix + 1] * DAMP;
      let az = (hz - pz) * SPRING - vel[ix + 2] * DAMP;

      // Cursor repulsion (XY plane — cursor lives on z≈0)
      if (cursorActive) {
        const dx = px - cx;
        const dy = py - cy;
        const d2 = dx * dx + dy * dy;
        if (d2 < RADIUS2 && d2 > 0.0001) {
          const falloff = 1 - d2 / RADIUS2; // 1 at centre → 0 at edge
          const inv = 1 / Math.sqrt(d2);
          const f = PUSH * falloff * falloff;
          ax += dx * inv * f;
          ay += dy * inv * f;
        }
      }

      // Streaming inward pull
      if (streamPull > 0) {
        const k = 0.6 + seeds[i] * 0.9;
        ax -= px * streamPull * k;
        ay -= py * streamPull * k;
        ax += (Math.random() - 0.5) * 0.2; // jitter so it feels alive
      }

      vel[ix]     += ax * dt;
      vel[ix + 1] += ay * dt;
      vel[ix + 2] += az * dt;

      arr[ix]     = px + vel[ix]     * dt + breathe;
      arr[ix + 1] = py + vel[ix + 1] * dt + breathe;
      arr[ix + 2] = pz + vel[ix + 2] * dt;

      // Streaming recycle: pop back to home if too close to centre.
      if (streamPull > 0) {
        const d2c = arr[ix] * arr[ix] + arr[ix + 1] * arr[ix + 1];
        if (d2c < 0.2) {
          arr[ix]     = hx;
          arr[ix + 1] = hy;
          arr[ix + 2] = hz;
          vel[ix] = vel[ix + 1] = vel[ix + 2] = 0;
        }
      }
    }
    attr.needsUpdate = true;

    if (matRef.current) {
      matRef.current.opacity = streaming
        ? 0.85
        : 0.75 + Math.sin(t * 0.8) * 0.04;
    }
  });

  return (
    <points ref={pointsRef}>
      <bufferGeometry>
        <bufferAttribute
          attach="attributes-position"
          count={COUNT}
          array={positions}
          itemSize={3}
        />
      </bufferGeometry>
      <pointsMaterial
        ref={matRef}
        color="#0f172a"
        size={0.028}
        sizeAttenuation
        transparent
        opacity={0.78}
        depthWrite={false}
        blending={THREE.NormalBlending}
      />
    </points>
  );
}



/* ============================================================
   ABSTRACT PRIMITIVES — small wireframe shapes drifting along
   the outskirts of the canvas. Each has a home position and
   spring-physics: when the cursor enters its bubble it gets
   flung outward, then drifts back home.
   ============================================================ */
type PrimitiveSpec = {
  kind: "cube" | "sphere" | "tetra" | "octa" | "torus" | "square" | "ring";
  home: [number, number, number];
  size: number;
  color: string;
  /** rotation speed multiplier */
  spin: number;
};

const EDGE_PRIMITIVES: PrimitiveSpec[] = [
  { kind: "cube",   home: [-5.4,  2.2, -0.5], size: 0.55, color: "#0f172a", spin: 1.0 },
  { kind: "sphere", home: [ 5.6,  2.4, -0.8], size: 0.55, color: "#2563eb", spin: 0.7 },
  { kind: "tetra",  home: [-5.8, -2.0,  0.2], size: 0.55, color: "#475569", spin: 1.2 },
  { kind: "octa",   home: [ 5.4, -2.2,  0.0], size: 0.55, color: "#0f172a", spin: 0.9 },
  { kind: "torus",  home: [ 0.0,  3.4, -1.0], size: 0.5,  color: "#1f2937", spin: 0.8 },
  { kind: "square", home: [ 0.0, -3.2,  0.0], size: 0.85, color: "#0f172a", spin: 0.6 },
  { kind: "ring",   home: [-3.4,  0.0, -1.2], size: 0.7,  color: "#475569", spin: 0.5 },
  { kind: "cube",   home: [ 3.4,  0.0, -0.4], size: 0.45, color: "#2563eb", spin: 1.3 },
];

function FloatingPrimitive({ spec, idx }: { spec: PrimitiveSpec; idx: number }) {
  const ref = useRef<THREE.Mesh>(null);
  const { viewport } = useThree();
  // Live state outside React for perf.
  const state = useRef({
    px: spec.home[0],
    py: spec.home[1],
    pz: spec.home[2],
    vx: 0,
    vy: 0,
    vz: 0,
  });

  useFrame((s, delta) => {
    const m = ref.current;
    if (!m) return;
    const dt = Math.min(delta, 0.05);
    const t = s.clock.getElapsedTime();

    // Idle drift target — gentle lissajous around home position.
    const tx = spec.home[0] + Math.sin(t * 0.35 + idx) * 0.18;
    const ty = spec.home[1] + Math.cos(t * 0.42 + idx * 1.3) * 0.15;
    const tz = spec.home[2] + Math.sin(t * 0.28 + idx * 0.7) * 0.12;

    const SPRING = 4.5;
    const DAMP   = 2.2;

    let ax = (tx - state.current.px) * SPRING - state.current.vx * DAMP;
    let ay = (ty - state.current.py) * SPRING - state.current.vy * DAMP;
    let az = (tz - state.current.pz) * SPRING - state.current.vz * DAMP;

    // Cursor fling
    if (cursorWorld.active) {
      const dx = state.current.px - cursorWorld.x;
      const dy = state.current.py - cursorWorld.y;
      const d2 = dx * dx + dy * dy;
      const R  = 1.4 + spec.size;
      if (d2 < R * R && d2 > 0.0001) {
        const inv = 1 / Math.sqrt(d2);
        const falloff = 1 - Math.sqrt(d2) / R;
        const f = 80 * falloff * falloff;
        ax += dx * inv * f;
        ay += dy * inv * f;
      }
    }

    state.current.vx += ax * dt;
    state.current.vy += ay * dt;
    state.current.vz += az * dt;
    state.current.px += state.current.vx * dt;
    state.current.py += state.current.vy * dt;
    state.current.pz += state.current.vz * dt;

    // Viewport bounce — keep primitives inside the visible canvas.
    // viewport.{width,height} is the world-space size at z=0.
    const margin = spec.size * 0.7;
    const halfW = viewport.width  / 2 - margin;
    const halfH = viewport.height / 2 - margin;
    if (state.current.px >  halfW) { state.current.px =  halfW; state.current.vx = -Math.abs(state.current.vx) * 0.5; }
    if (state.current.px < -halfW) { state.current.px = -halfW; state.current.vx =  Math.abs(state.current.vx) * 0.5; }
    if (state.current.py >  halfH) { state.current.py =  halfH; state.current.vy = -Math.abs(state.current.vy) * 0.5; }
    if (state.current.py < -halfH) { state.current.py = -halfH; state.current.vy =  Math.abs(state.current.vy) * 0.5; }

    m.position.set(state.current.px, state.current.py, state.current.pz);
    m.rotation.x = t * 0.4 * spec.spin;
    m.rotation.y = t * 0.5 * spec.spin;
    m.rotation.z = t * 0.2 * spec.spin;
  });

  // Geometry per kind
  const geom = (() => {
    switch (spec.kind) {
      case "cube":   return <boxGeometry args={[spec.size, spec.size, spec.size]} />;
      case "sphere": return <sphereGeometry args={[spec.size * 0.6, 16, 12]} />;
      case "tetra":  return <tetrahedronGeometry args={[spec.size * 0.7]} />;
      case "octa":   return <octahedronGeometry args={[spec.size * 0.7]} />;
      case "torus":  return <torusGeometry args={[spec.size * 0.7, 0.05, 10, 48]} />;
      case "square": return <planeGeometry args={[spec.size, spec.size]} />;
      case "ring":   return <torusGeometry args={[spec.size, 0.012, 8, 96]} />;
    }
  })();

  // Solid (subtle) for torus/ring; everything else is wire-only via Edges.
  if (spec.kind === "torus" || spec.kind === "ring") {
    return (
      <mesh ref={ref} position={spec.home}>
        {geom}
        <meshBasicMaterial color={spec.color} transparent opacity={0.7} />
      </mesh>
    );
  }
  return (
    <mesh ref={ref} position={spec.home}>
      {geom}
      <meshBasicMaterial transparent opacity={0} depthWrite={false} />
      <Edges threshold={1} color={spec.color} />
    </mesh>
  );
}

function AbstractPrimitives() {
  return (
    <group>
      {EDGE_PRIMITIVES.map((spec, i) => (
        <FloatingPrimitive key={i} spec={spec} idx={i} />
      ))}
    </group>
  );
}

/* ============================================================
   CURSOR TRACKER — projects mouse position onto the camera's
   z=0 plane and writes to the shared `cursorWorld` ref so both
   the point cloud and the primitives can react.
   ============================================================ */
function CursorTracker() {
  const { camera, size } = useThree();
  useEffect(() => {
    const canvas = document.querySelector("canvas");
    if (!canvas) return;
    const ndc = new THREE.Vector2();
    const ray = new THREE.Raycaster();
    const plane = new THREE.Plane(new THREE.Vector3(0, 0, 1), 0);
    const hit = new THREE.Vector3();

    const onMove = (e: MouseEvent) => {
      const rect = (canvas as HTMLCanvasElement).getBoundingClientRect();
      ndc.x =  ((e.clientX - rect.left) / rect.width)  * 2 - 1;
      ndc.y = -((e.clientY - rect.top)  / rect.height) * 2 + 1;
      ray.setFromCamera(ndc, camera);
      if (ray.ray.intersectPlane(plane, hit)) {
        cursorWorld.x = hit.x;
        cursorWorld.y = hit.y;
        cursorWorld.active = true;
      }
    };
    const onLeave = () => { cursorWorld.active = false; };

    window.addEventListener("mousemove", onMove);
    window.addEventListener("mouseleave", onLeave);
    return () => {
      window.removeEventListener("mousemove", onMove);
      window.removeEventListener("mouseleave", onLeave);
    };
  }, [camera, size]);
  return null;
}

/* Wraps AmbientCloud + AbstractPrimitives + CursorTracker. */
function AmbientCloudCanvas({ streaming = false }: { streaming?: boolean }) {
  return (
    <Canvas
      dpr={[1, 1.5]}
      camera={{ position: [0, 0, 7], fov: 45 }}
      gl={{ antialias: true, alpha: true }}
      style={{ position: "absolute", inset: 0 }}
    >
      <CursorTracker />
      <AmbientCloud streaming={streaming} />
      <AbstractPrimitives />
    </Canvas>
  );
}



/* ---------------- Shared rotation state ----------------
 * The hero shape and the bottom-left axis gizmo live in two separate
 * <Canvas>es but must rotate as one. We keep a tiny module-scoped
 * mutable ref that any component can read or write each frame —
 * simpler than threading a React context through R3F.
 * --------------------------------------------------------- */
const sharedRotation = { y: 0 };

/* ---------------- 3D stage scene ---------------- */

interface PipelineSceneProps {
  stage: Stage;
}

/**
 * The hero 3D viewport that morphs as the pipeline progresses.
 * - Stage ≤1 : raw point cloud, full sound-wave ripple
 * - Stage 2  : ripple smoothly decays to a calm static sphere of points
 * - Stage 3  : 1) hold the calm cloud for ~1s
 *              2) triangles + wire build over the cloud
 *              3) points fade out as triangles cover them
 *              4) wireframe outline fades out, leaving the solid sphere
 * - Stage 4+ : static solid sphere
 */
function PipelineScene({ stage }: PipelineSceneProps) {
  const groupRef = useRef<THREE.Group>(null);
  const pointsRef = useRef<THREE.Points>(null);
  const pointsMatRef = useRef<THREE.PointsMaterial>(null);
  const solidGeomRef = useRef<THREE.BufferGeometry>(null);
  const wireGeomRef = useRef<THREE.BufferGeometry>(null);
  const wireMatRef = useRef<THREE.LineBasicMaterial>(null);
  const { base, phis, thetas, count } = useScanPoints(3200);

  // Mutable position buffer we update each frame for the sound-wave ripple.
  const positions = useMemo(() => new Float32Array(count * 3), [count]);

  // Build a triangle-soup sphere we can reveal incrementally via drawRange.
  // Triangles are sorted by their centroid Y (top → bottom) so the mesh
  // appears to grow over the surface rather than randomly flickering on.
  // Triangle-soup of the reconstructed mesh, sorted top→bottom so the
  // surface appears to grow over the point cloud rather than randomly
  // flickering on. Sourced from useReconstructedGeometry so that real
  // .ply input will Just Work here.
  const sourceGeometry = useReconstructedGeometry();
  const { triPositions, triNormals, wirePositions, triCount } = useMemo(() => {
    const idx = sourceGeometry.getIndex();
    const pos = sourceGeometry.getAttribute("position") as THREE.BufferAttribute;
    const nrm = sourceGeometry.getAttribute("normal") as THREE.BufferAttribute;

    // Build a triangle index list, falling back to sequential triples
    // for non-indexed geometries (typical of raw reconstructed meshes).
    const triN = idx ? idx.count / 3 : pos.count / 3;
    const getIdx = (i: number) => (idx ? idx.getX(i) : i);

    // Collect each triangle with its centroid Y for sorting.
    const tris: { a: number; b: number; c: number; y: number }[] = [];
    for (let i = 0; i < triN; i++) {
      const a = getIdx(i * 3);
      const b = getIdx(i * 3 + 1);
      const c = getIdx(i * 3 + 2);
      const y = (pos.getY(a) + pos.getY(b) + pos.getY(c)) / 3;
      tris.push({ a, b, c, y });
    }
    tris.sort((p, q) => q.y - p.y); // top first

    const triPos = new Float32Array(triN * 9);
    const triNrm = new Float32Array(triN * 9);
    // Wireframe: 3 edges per triangle, 2 verts per edge → 6 verts (18 floats).
    const wirePos = new Float32Array(triN * 18);

    for (let i = 0; i < triN; i++) {
      const { a, b, c } = tris[i];
      const verts = [a, b, c];
      for (let v = 0; v < 3; v++) {
        const vi = verts[v];
        triPos[i * 9 + v * 3] = pos.getX(vi);
        triPos[i * 9 + v * 3 + 1] = pos.getY(vi);
        triPos[i * 9 + v * 3 + 2] = pos.getZ(vi);
        triNrm[i * 9 + v * 3] = nrm.getX(vi);
        triNrm[i * 9 + v * 3 + 1] = nrm.getY(vi);
        triNrm[i * 9 + v * 3 + 2] = nrm.getZ(vi);
      }
      // Edges A-B, B-C, C-A as line segments.
      const edges: [number, number][] = [
        [a, b],
        [b, c],
        [c, a],
      ];
      for (let e = 0; e < 3; e++) {
        const [s, t] = edges[e];
        wirePos[i * 18 + e * 6] = pos.getX(s);
        wirePos[i * 18 + e * 6 + 1] = pos.getY(s);
        wirePos[i * 18 + e * 6 + 2] = pos.getZ(s);
        wirePos[i * 18 + e * 6 + 3] = pos.getX(t);
        wirePos[i * 18 + e * 6 + 4] = pos.getY(t);
        wirePos[i * 18 + e * 6 + 5] = pos.getZ(t);
      }
    }
    return {
      triPositions: triPos,
      triNormals: triNrm,
      wirePositions: wirePos,
      triCount: triN,
    };
  }, [sourceGeometry]);

  // Track when the current stage started so we can drive in-stage easings.
  const stageStartRef = useRef<number>(performance.now());
  useEffect(() => {
    stageStartRef.current = performance.now();
  }, [stage]);

  // Stage-driven visibility / look. Points stay rendered into stage 3 so
  // they can fade out *underneath* the growing triangle mesh.
  const showPoints = stage <= 3;
  const showSolid = stage >= 3;
  const showWire = stage === 3 || stage === 4;
  const pointColor =
    stage <= 1 ? "#94a3b8" : "#64748b";
  const pointSize = stage <= 1 ? 0.04 : 0.045;

  // Stage 3 sub-timeline (seconds since stage 3 started). ~30% faster.
  const HOLD_END = 0.7;        // calm static cloud
  const BUILD_END = 2.8;       // triangles grow
  const POINTS_FADE_END = 3.2; // points fade out
  const WIRE_FADE_END = 4.2;   // wireframe fades out
  // Stage 3 lasts 4.3s — keep this in sync with STAGE_DURATIONS[3].
  const STAGE3_DURATION = 4.3;

  useFrame((state, delta) => {
    const stageElapsed = (performance.now() - stageStartRef.current) / 1000;

    /* Rotation orchestration:
     *  - Stage ≤ 2: hold perfectly still at 0 so we have a known anchor.
     *  - Stage 3: drive a single eased 360° from 0 → 2π over the full
     *    stage duration. Crucially we land EXACTLY at 2π (≡ 0 mod 2π) so
     *    the very first frame of stage 4 already sees the shape face-on.
     *    This is what lets the FRONT silhouette of the unfolded T-cross
     *    drop in seamlessly on top of the just-finished sphere — same
     *    orientation, same outline, no jiggle.
     *  - Stage ≥ 4: leave sharedRotation alone; UnfoldFoldScene drives it.
     */
    if (groupRef.current) {
      let yRot = sharedRotation.y;
      if (stage <= 2) {
        yRot = 0;
      } else if (stage === 3) {
        // Drive an exact, complete 360°: clamp t to [0, 1] so the last
        // frame is guaranteed to land at k=1 → yRot = 2π (≡ 0 mod 2π).
        // This guarantees the shape returns to its starting orientation
        // regardless of frame timing or .ply geometry.
        const t = Math.min(1, Math.max(0, stageElapsed / STAGE3_DURATION));
        const k = easeInOut(t);
        // When fully complete, snap to exactly 0 (instead of 2π) so the
        // stage-4 handoff (which starts at rotation=0) has zero numeric
        // drift between frames.
        yRot = t >= 1 ? 0 : k * Math.PI * 2;
      }
      sharedRotation.y = yRot;
      groupRef.current.rotation.y = yRot;
    }

    // ---- Point cloud ripple ----
    if (pointsRef.current && showPoints) {
      const t = state.clock.elapsedTime;
      // Stage 0/1: full amplitude. Stage 2: ease 0.18 → 0 over ~3.4s. Stage 3+: calm.
      let amp = 0.18;
      if (stage === 2) {
        const k = Math.min(1, stageElapsed / 2.4);
        const ease = 1 - k * k * k; // cubic ease-out to zero
        amp = 0.18 * ease;
      } else if (stage >= 3) {
        amp = 0;
      }
      const baseR = 1.25;
      for (let i = 0; i < count; i++) {
        const phi = phis[i];
        const theta = thetas[i];
        const wave =
          Math.sin(phi * 6 - t * 2.4) * 0.6 +
          Math.cos(theta * 3 + t * 1.7) * 0.4;
        const r = baseR * (1 + amp * wave);
        positions[i * 3] = r * Math.cos(theta) * Math.sin(phi);
        positions[i * 3 + 1] = r * Math.sin(theta) * Math.sin(phi);
        positions[i * 3 + 2] = r * Math.cos(phi);
      }
      const attr = pointsRef.current.geometry.attributes
        .position as THREE.BufferAttribute;
      attr.needsUpdate = true;
    }

    // ---- Point cloud opacity (fades out partway through stage 3) ----
    if (pointsMatRef.current) {
      let opacity = 1;
      if (stage === 3) {
        if (stageElapsed < BUILD_END - 1) opacity = 1; // hold + early build
        else if (stageElapsed < POINTS_FADE_END) {
          const k = (stageElapsed - (BUILD_END - 1)) / (POINTS_FADE_END - (BUILD_END - 1));
          opacity = 1 - k;
        } else opacity = 0;
      } else if (stage >= 4) opacity = 0;
      pointsMatRef.current.opacity = opacity;
    }

    // ---- Triangle build ----
    if (solidGeomRef.current && wireGeomRef.current) {
      let progress = 0;
      if (stage === 3) {
        if (stageElapsed < HOLD_END) progress = 0;
        else if (stageElapsed < BUILD_END) {
          progress = (stageElapsed - HOLD_END) / (BUILD_END - HOLD_END);
        } else progress = 1;
      } else if (stage >= 4) {
        progress = 1;
      }
      const visibleTris = Math.floor(progress * triCount);
      solidGeomRef.current.setDrawRange(0, visibleTris * 3);
      wireGeomRef.current.setDrawRange(0, visibleTris * 6);
    }

    // ---- Wireframe fade ----
    if (wireMatRef.current) {
      let opacity = 0;
      if (stage === 3) {
        if (stageElapsed < BUILD_END) opacity = 0.9;
        else if (stageElapsed < WIRE_FADE_END) {
          const k = (stageElapsed - BUILD_END) / (WIRE_FADE_END - BUILD_END);
          opacity = 0.9 * (1 - k);
        } else opacity = 0;
      } else if (stage === 4) opacity = 0;
      wireMatRef.current.opacity = opacity;
    }
  });

  // Seed the position buffer once from base so first frame isn't empty.
  useMemo(() => {
    for (let i = 0; i < count; i++) {
      positions[i * 3] = base[i * 3] * 1.25;
      positions[i * 3 + 1] = base[i * 3 + 1] * 1.25;
      positions[i * 3 + 2] = base[i * 3 + 2] * 1.25;
    }
  }, [base, positions, count]);

  return (
    <group ref={groupRef}>
      {/* Raw scan point cloud (rippling sphere) */}
      {showPoints && (
        <points ref={pointsRef}>
          <bufferGeometry>
            <bufferAttribute
              attach="attributes-position"
              count={count}
              array={positions}
              itemSize={3}
            />
          </bufferGeometry>
          <pointsMaterial
            ref={pointsMatRef}
            size={pointSize}
            color={pointColor}
            sizeAttenuation
            transparent
          />
        </points>
      )}

      {/* Solid sphere built triangle-by-triangle */}
      {showSolid && (
        <mesh castShadow receiveShadow>
          <bufferGeometry ref={solidGeomRef}>
            <bufferAttribute
              attach="attributes-position"
              count={triPositions.length / 3}
              array={triPositions}
              itemSize={3}
            />
            <bufferAttribute
              attach="attributes-normal"
              count={triNormals.length / 3}
              array={triNormals}
              itemSize={3}
            />
          </bufferGeometry>
          <meshStandardMaterial
            color="#f5f7fa"
            roughness={0.32}
            metalness={0.15}
          />
        </mesh>
      )}

      {/* Wireframe overlay tracking the same triangles, fades out in stage 4 */}
      {showWire && (
        <lineSegments>
          <bufferGeometry ref={wireGeomRef}>
            <bufferAttribute
              attach="attributes-position"
              count={wirePositions.length / 3}
              array={wirePositions}
              itemSize={3}
            />
          </bufferGeometry>
          <lineBasicMaterial
            ref={wireMatRef}
            color="#2563eb"
            transparent
            opacity={0.9}
            depthTest={false}
          />
        </lineSegments>
      )}
    </group>
  );
}

/* ---------------- Stage 4: unfold → fold → spin → reveal ----------------
 * A 6-face "view cube" wraps the reconstructed mesh. Faces hinge open
 * into a T-cross net (one face per orthographic side), pause, then
 * fold back into a closed cube which spins 360°. Finally the cube
 * fades to reveal the real mesh underneath.
 *
 *   T-cross layout (FRONT stays in place at z=+1):
 *
 *              [TOP]
 *     [LEFT] [FRONT] [RIGHT] [BACK]
 *             [BOTTOM]
 *
 * For each face we render a small "view" of the shape. With the mock
 * sphere all six views read as circles (which is geometrically true
 * for a sphere). When a real .ply mesh replaces the mock geometry
 * via useReconstructedGeometry, the same per-face view automatically
 * renders the real shape from that side.
 * -------------------------------------------------------------------- */

type FaceId = "front" | "back" | "left" | "right" | "top" | "bottom";

/* Per-face projection: which world axes map to the face's local 2D (u, v).
 * Each face shows a flat orthographic *silhouette* of the shape — i.e. all
 * vertices projected onto the face's image plane. We compute a 2D convex
 * hull of the projected points and render it as a filled THREE.Shape, so
 * the result is a true 2D picture (not a 3D mini-mesh). For a sphere every
 * silhouette is a disc; for a real .ply mesh each face will produce its
 * distinct outline automatically. */
const FACE_PROJECTION: Record<
  FaceId,
  { u: "x" | "y" | "z"; v: "x" | "y" | "z"; uSign: 1 | -1; vSign: 1 | -1 }
> = {
  front:  { u: "x", uSign:  1, v: "y", vSign:  1 },
  back:   { u: "x", uSign: -1, v: "y", vSign:  1 },
  right:  { u: "z", uSign: -1, v: "y", vSign:  1 },
  left:   { u: "z", uSign:  1, v: "y", vSign:  1 },
  top:    { u: "x", uSign:  1, v: "z", vSign: -1 },
  bottom: { u: "x", uSign:  1, v: "z", vSign:  1 },
};

/** 2D convex hull (Andrew's monotone chain). Returns hull points CCW. */
function convexHull2D(pts: { x: number; y: number }[]): { x: number; y: number }[] {
  if (pts.length < 3) return pts.slice();
  const sorted = pts.slice().sort((a, b) => (a.x === b.x ? a.y - b.y : a.x - b.x));
  const cross = (
    o: { x: number; y: number },
    a: { x: number; y: number },
    b: { x: number; y: number },
  ) => (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x);
  const lower: typeof sorted = [];
  for (const p of sorted) {
    while (lower.length >= 2 && cross(lower[lower.length - 2], lower[lower.length - 1], p) <= 0)
      lower.pop();
    lower.push(p);
  }
  const upper: typeof sorted = [];
  for (let i = sorted.length - 1; i >= 0; i--) {
    const p = sorted[i];
    while (upper.length >= 2 && cross(upper[upper.length - 2], upper[upper.length - 1], p) <= 0)
      upper.pop();
    upper.push(p);
  }
  lower.pop();
  upper.pop();
  return lower.concat(upper);
}

/**
 * Build flat 2D silhouette shapes (one per face) by projecting every vertex
 * of the source geometry onto each face's image plane and taking the 2D
 * convex hull. Returned shapes are in TRUE world-space units (same units
 * as the source geometry), centered around (0, 0). The caller renders
 * them at scale=1 so each silhouette matches the exact size of the
 * underlying mesh from that orthographic angle. This is fully dynamic:
 * swap the source geometry (e.g. for a real .ply file) and the
 * silhouettes automatically resize to match the new shape.
 */
function useFaceSilhouettes(
  sourceGeometry: THREE.BufferGeometry,
): Record<FaceId, THREE.Shape> {
  return useMemo(() => {
    const pos = sourceGeometry.getAttribute("position") as THREE.BufferAttribute;
    const result = {} as Record<FaceId, THREE.Shape>;
    // Subsample for performance on dense meshes.
    const stride = Math.max(1, Math.floor(pos.count / 4000));
    (Object.keys(FACE_PROJECTION) as FaceId[]).forEach((face) => {
      const proj = FACE_PROJECTION[face];
      const pts: { x: number; y: number }[] = [];
      for (let i = 0; i < pos.count; i += stride) {
        const v = { x: pos.getX(i), y: pos.getY(i), z: pos.getZ(i) };
        pts.push({ x: v[proj.u] * proj.uSign, y: v[proj.v] * proj.vSign });
      }
      const hull = convexHull2D(pts);
      const shape = new THREE.Shape();
      hull.forEach((p, i) => {
        // No normalization — keep world-space coordinates so the silhouette
        // is the exact same size as the projected mesh outline.
        if (i === 0) shape.moveTo(p.x, p.y);
        else shape.lineTo(p.x, p.y);
      });
      shape.closePath();
      result[face] = shape;
    });
    return result;
  }, [sourceGeometry]);
}

/** Quintic ease in/out — softer acceleration & deceleration than cubic,
 *  giving long transitions (unfold/fold/spin) a more "premium" feel. */
function easeInOut(t: number) {
  return t < 0.5
    ? 16 * t * t * t * t * t
    : 1 - Math.pow(-2 * t + 2, 5) / 2;
}

/** Gentle ease-out for fades — starts quickly, settles softly. */
function easeOut(t: number) {
  return 1 - Math.pow(1 - t, 3);
}

/** Ease-out with a small overshoot — gives panels a subtle "settle" at the
 *  end of the fold so the motion reads as physical rather than mechanical. */
function easeOutBack(t: number, overshoot = 1.35) {
  const c1 = overshoot;
  const c3 = c1 + 1;
  return 1 + c3 * Math.pow(t - 1, 3) + c1 * Math.pow(t - 1, 2);
}

/** Clamp helper. */
function clamp01(t: number) {
  return t < 0 ? 0 : t > 1 ? 1 : t;
}

/**
 * One face of the view cube.
 *
 * The face is built with two nested groups so we can pivot the panel
 * around its hinge edge:
 *
 *   <hingeGroup>  (positioned ON the hinge axis; rotates 0 → unfold angle)
 *     <panelGroup>  (translates the panel so its hinge edge sits at the parent origin)
 *       <plane + border + miniature shape>
 *     </panelGroup>
 *   </hingeGroup>
 */
interface CubeFaceProps {
  face: FaceId;
  /** Hinge pivot point in cube-local space. */
  hingePos: [number, number, number];
  /** Axis to rotate around (unit vector). */
  hingeAxis: [number, number, number];
  /** Offset from hinge to the panel's local origin (where the plane is centered). */
  panelOffset: [number, number, number];
  /** Unfolded hinge angle in radians (0 = closed/folded into cube). */
  /** Fold angle in radians: rotation applied at progress=0 (folded into cube). At progress=1 the panel sits unfolded coplanar with FRONT. */
  foldAngle: number;
  /** Animation progress 0 (folded) → 1 (fully unfolded). */
  progress: number;
  /** Opacity of the panel (used for the final fade-out reveal). */
  opacity: number;
  /** Flat 2D silhouette of the shape from this face — rendered as a filled THREE.Shape. */
  silhouette: THREE.Shape;
  /** Cube edge length — used to size the face plane. */
  edge: number;
}

function CubeFace({
  face,
  hingePos,
  hingeAxis,
  panelOffset,
  foldAngle,
  progress,
  opacity,
  silhouette,
  edge,
}: CubeFaceProps) {
  const hingeRef = useRef<THREE.Group>(null);
  const panelMatRef = useRef<THREE.MeshStandardMaterial>(null);
  const borderMatRef = useRef<THREE.LineBasicMaterial>(null);
  const shapeMatRef = useRef<THREE.MeshStandardMaterial>(null);

  // Build a square outline once so each face has a crisp border.
  const borderPoints = useMemo(() => {
    const h = edge / 2;
    return new Float32Array([
      -h, -h, 0, h, -h, 0,
      h, -h, 0, h, h, 0,
      h, h, 0, -h, h, 0,
      -h, h, 0, -h, -h, 0,
    ]);
  }, [edge]);

  // Subtle physical thickness so lighting reveals the panel as a real card,
  // not a flat sticker. Tuned small so the folded cube still reads clean.
  const thickness = edge * 0.045;

  useFrame(() => {
    if (hingeRef.current) {
      const k = 1 - progress; // 1 = folded, 0 = unfolded
      hingeRef.current.rotation.x = hingeAxis[0] * foldAngle * k;
      hingeRef.current.rotation.y = hingeAxis[1] * foldAngle * k;
      hingeRef.current.rotation.z = hingeAxis[2] * foldAngle * k;
    }
    // All three layers fade together so the panel reads as a single solid
    // card. Background is fully opaque so the folded cube hides whatever's
    // behind it (no see-through to the bridge mesh).
    if (panelMatRef.current) panelMatRef.current.opacity = opacity;
    if (borderMatRef.current) borderMatRef.current.opacity = opacity * 0.65;
    if (shapeMatRef.current) shapeMatRef.current.opacity = opacity;
  });

  return (
    <group position={hingePos} ref={hingeRef}>
      <group position={panelOffset}>
        {/* Solid panel BODY — a thin slab with subtle thickness. The slab is
            shifted back along -Z by half its depth so its FRONT face sits
            exactly at the panel's z=0 plane (same plane silhouettes used
            before). This keeps the cube geometry exact while giving the
            panel real volume that the lights can shade. */}
        <mesh position={[0, 0, -thickness / 2]} castShadow receiveShadow>
          <boxGeometry args={[edge, edge, thickness]} />
          <meshStandardMaterial
            ref={panelMatRef}
            color="#fafbfd"
            roughness={0.78}
            metalness={0.02}
            transparent
            opacity={0}
          />
        </mesh>
        {/* Crisp thin border on the front face */}
        <lineSegments position={[0, 0, 0.001]}>
          <bufferGeometry>
            <bufferAttribute
              attach="attributes-position"
              count={borderPoints.length / 3}
              array={borderPoints}
              itemSize={3}
            />
          </bufferGeometry>
          <lineBasicMaterial
            ref={borderMatRef}
            color="#94a3b8"
            transparent
            opacity={0}
          />
        </lineSegments>
        {/* Flat 2D silhouette of the shape from this side — solid fill in a
            soft slate that reads against the white panel. Lit so it gets a
            faint gradient as panels fold (subtle but rich). */}
        {/* Silhouette is in true world-space units → render at scale=1 so
            it always matches the underlying mesh's exact projected size,
            regardless of geometry (works for sphere mock and real .ply). */}
        <mesh position={[0, 0, 0.003]}>
          <shapeGeometry args={[silhouette]} />
          <meshStandardMaterial
            ref={shapeMatRef}
            color="#cbd5e1"
            roughness={0.85}
            metalness={0}
            transparent
            opacity={0}
            side={THREE.DoubleSide}
          />
        </mesh>
      </group>
    </group>
  );
}

interface UnfoldFoldSceneProps {
  /** Stage-4 elapsed time in seconds. */
  elapsed: number;
}

function UnfoldFoldScene({ elapsed }: UnfoldFoldSceneProps) {
  const groupRef = useRef<THREE.Group>(null);
  const sphereMatRef = useRef<THREE.MeshStandardMaterial>(null);
  const sourceGeometry = useReconstructedGeometry();
  const silhouettes = useFaceSilhouettes(sourceGeometry);

  // Cube edge length — derived from the source geometry's bounding box
  // so it always matches the actual shape (sphere or future .ply mesh).
  // No padding: edge equals the largest bounding-box dimension exactly,
  // so each panel is the same size as the silhouette/mesh it overlays.
  // This makes the FRONT panel land at the exact size of the sphere
  // during cross-fade — no visible jump in scale.
  const edge = useMemo(() => {
    sourceGeometry.computeBoundingBox();
    const bb = sourceGeometry.boundingBox!;
    const size = new THREE.Vector3();
    bb.getSize(size);
    return Math.max(size.x, size.y, size.z);
  }, [sourceGeometry]);
  const half = edge / 2;

  /* Soft contact-shadow texture — a radial alpha gradient drawn on a 2D canvas
   * once and reused as a Three texture. Renders on a ground plane below the
   * cube to give the folded form a sense of weight and physical presence. */
  const shadowTexture = useMemo(() => {
    const size = 256;
    const cv = document.createElement("canvas");
    cv.width = cv.height = size;
    const ctx = cv.getContext("2d")!;
    const grad = ctx.createRadialGradient(size / 2, size / 2, 0, size / 2, size / 2, size / 2);
    grad.addColorStop(0,    "rgba(15, 23, 42, 0.42)");
    grad.addColorStop(0.45, "rgba(15, 23, 42, 0.18)");
    grad.addColorStop(1,    "rgba(15, 23, 42, 0)");
    ctx.fillStyle = grad;
    ctx.fillRect(0, 0, size, size);
    const tex = new THREE.CanvasTexture(cv);
    tex.needsUpdate = true;
    return tex;
  }, []);
  const shadowMatRef = useRef<THREE.MeshBasicMaterial>(null);

  /* ----------------- Sub-timeline (seconds within stage 4) -----------------
   *  ~30% faster than before, and the reconstructed mesh ("bridge sphere")
   *  is now the FRONT pane: it stays visible the entire sequence so there's
   *  no cross-dissolve. The 5 surrounding panels (top/left/right/bottom/back)
   *  fade in around it, fold up around it, spin with it, then fade away
   *  to leave just the mesh.
   *
   *   0.0 →  2.5   FADE_IN     5 surrounding panels appear in sequence
   *   2.5 →  2.9   HOLD_NET    brief settle so the eye registers the full net
   *   2.9 →  4.4   ZOOM_OUT    camera pulls back (mesh stays put as front)
   *   4.4 →  4.8   HOLD_WIDE   beat to read the layout before folding
   *   4.8 →  7.0   FOLD        net folds into a cube as a SEQUENTIAL CASCADE
   *                            (bottom → left → right → top → back) with a
   *                            small back-ease overshoot on each panel so it
   *                            "settles" into place rather than snapping shut.
   *   7.0 →  7.4   HOLD_CUBE   beat on the closed cube
   *   7.4 → 10.6   SPIN        cube does a single, slow 360° around Y
   *  10.6 → 11.0   HOLD_SPUN   beat after the spin completes
   *  11.0 → 12.3   REVEAL      surrounding panels fade out (mesh remains)
   *  12.3 → end    AMBIENT     final mesh slowly rotating
   * ------------------------------------------------------------------------ */
  const FADE_IN_END   = 2.5;
  const HOLD_NET_END  = 2.8;
  const ZOOM_OUT_END  = 3.7;   // 0.9s zoom-out
  const HOLD_WIDE_END = 4.1;
  const FOLD_END      = 6.3;   // 2.2s fold window preserved
  const HOLD_CUBE_END = 6.7;
  const SPIN_END      = 9.9;
  const HOLD_SPUN_END = 10.3;
  const REVEAL_END    = 11.6;

  // Zoom levels. ZOOM_IN is set so the bridge mesh appears at EXACTLY the
  // same on-screen size as the stage-3 sphere (zero-jump handoff).
  // Stage-3 camera at z=4.5, stage-4 camera at z=8.6 → ratio 8.6/4.5.
  const ZOOM_IN = 8.6 / 4.5; // ≈ 1.911
  const ZOOM_OUT = 0.85;

  // Order in which all 6 panels fade in. FRONT goes first (index 0) so it
  // appears in lockstep with the other panels. The bridge mesh (3D shaded
  // sphere) cross-fades OUT in sync with FRONT's fade-in (see bridgeOpacity
  // below) — so the swap from "real mesh" → "flat silhouette" happens
  // gradually as the panels arrive, not as a separate later step.
  const FADE_ORDER: FaceId[] = ["front", "top", "left", "right", "bottom", "back"];
  const FACE_FADE_DURATION = 0.5;
  const FACE_FADE_STAGGER  = 0.32;

  const faceOpacityFor = (face: FaceId): number => {
    if (elapsed >= FADE_IN_END) return 1;
    const order = FADE_ORDER.indexOf(face);
    const start = order * FACE_FADE_STAGGER;
    if (elapsed < start) return 0;
    return easeOut(Math.min(1, (elapsed - start) / FACE_FADE_DURATION));
  };

  // Bridge mesh fade-out tracks FRONT's fade-in (index 0 of FADE_ORDER) so
  // the silhouette swap is invisible — what disappears underneath is exactly
  // what's appearing on top, at the same rate.
  const bridgeFadeOutOpacity = (): number => {
    const start = 0; // FRONT is index 0
    if (elapsed < start) return 1;
    const k = easeOut(Math.min(1, (elapsed - start) / FACE_FADE_DURATION));
    return 1 - k;
  };


  /* ----- Cascade fold (overlapping, smooth) -----
   * Each panel still folds in cascade order:
   *
   *   bottom → left → right → top → back
   *
   * but the per-panel windows OVERLAP heavily so adjacent panels are
   * always moving together. This eliminates the visual "gap" you'd see
   * with strictly sequential folds (one panel mid-air while its
   * neighbour is still flat). The motion uses a plain ease-in-out —
   * no overshoot — so panels meet their cube edges cleanly with no
   * tuck-in past zero (which was creating tiny visible seams). */
  const FOLD_DURATION = FOLD_END - HOLD_WIDE_END; // 2.2s
  const FOLD_ORDER: FaceId[] = ["bottom", "left", "right", "top", "back"];
  // Each panel's fold takes 60% of the FOLD window; subsequent panels
  // start at 10% increments so 3-4 panels are folding simultaneously.
  const FOLD_PER_FACE = 0.6;   // fraction of FOLD_DURATION for one panel
  const FOLD_STAGGER  = 0.10;  // fraction between consecutive panel starts
  const FOLD_OFFSETS: Record<FaceId, number> = {
    front:  0,
    bottom: 0 * FOLD_STAGGER,
    left:   1 * FOLD_STAGGER,
    right:  2 * FOLD_STAGGER,
    top:    3 * FOLD_STAGGER,
    back:   4 * FOLD_STAGGER,
  };

  const foldProgressFor = (face: FaceId): number => {
    if (face === "front") return 1; // FRONT is the anchor, never folds
    if (elapsed <= HOLD_WIDE_END) return 1;
    if (elapsed >= FOLD_END) return 0;
    const offset = FOLD_OFFSETS[face] ?? 0;
    const localStart = HOLD_WIDE_END + offset * FOLD_DURATION;
    const localEnd   = localStart + FOLD_PER_FACE * FOLD_DURATION;
    if (elapsed <= localStart) return 1;
    if (elapsed >= localEnd) return 0;
    const t = clamp01((elapsed - localStart) / (localEnd - localStart));
    // Smooth ease-in-out, no overshoot — guarantees panels land flush
    // against their neighbours with no gap or tuck-in.
    const eased = easeInOut(t);
    return clamp01(1 - eased);
  };

  // Phase-driven values
  let unfoldProgress = 1;   // 1 = fully unfolded T-cross, 0 = closed cube
  let zoom = ZOOM_IN;
  let cubeOpacity = 1;      // multiplier for all 6 face opacities
  // Bridge mesh stays hidden through the entire unfold/fold/spin sequence.
  // Bridge mesh holds the EXACT stage-3 sphere position/size/material
  // through the entire fade-in / hold / zoom-out / hold-wide phases so the
  // user perceives no jump between stage 3 and stage 4 — only the
  // surrounding panels fading in around it. As the cube starts folding,
  // it cross-fades out in lockstep with the FRONT silhouette fading in.
  let bridgeOpacity = 1;
  // Soft ground-shadow opacity. 0 while the net is unfolded (no cube yet),
  // ramps in during FOLD as the form gains volume, full during HOLD/SPIN,
  // ramps out during REVEAL alongside cubeOpacity.
  let shadowOpacity = 0;

  if (elapsed < FADE_IN_END) {
    // Panels (incl. FRONT) fade in. Bridge mesh fades out tracking FRONT.
    unfoldProgress = 1;
    zoom = ZOOM_IN;
    bridgeOpacity = bridgeFadeOutOpacity();
  } else if (elapsed < HOLD_NET_END) {
    unfoldProgress = 1;
    zoom = ZOOM_IN;
    bridgeOpacity = 0;
  } else if (elapsed < ZOOM_OUT_END) {
    unfoldProgress = 1;
    const k = easeInOut((elapsed - HOLD_NET_END) / (ZOOM_OUT_END - HOLD_NET_END));
    zoom = ZOOM_IN + (ZOOM_OUT - ZOOM_IN) * k;
    bridgeOpacity = 0;
  } else if (elapsed < HOLD_WIDE_END) {
    zoom = ZOOM_OUT;
    unfoldProgress = 1;
    bridgeOpacity = 0;
  } else if (elapsed < FOLD_END) {
    zoom = ZOOM_OUT;
    unfoldProgress = 1 - easeInOut((elapsed - HOLD_WIDE_END) / (FOLD_END - HOLD_WIDE_END));
    shadowOpacity = easeOut((elapsed - HOLD_WIDE_END) / (FOLD_END - HOLD_WIDE_END));
    bridgeOpacity = 0;
  } else if (elapsed < HOLD_CUBE_END) {
    zoom = ZOOM_OUT;
    unfoldProgress = 0;
    shadowOpacity = 1;
    bridgeOpacity = 0;
  } else if (elapsed < SPIN_END) {
    zoom = ZOOM_OUT;
    unfoldProgress = 0;
    shadowOpacity = 1;
    bridgeOpacity = 0;
  } else if (elapsed < HOLD_SPUN_END) {
    zoom = ZOOM_OUT;
    unfoldProgress = 0;
    shadowOpacity = 1;
    bridgeOpacity = 0;
  } else if (elapsed < REVEAL_END) {
    zoom = ZOOM_OUT;
    unfoldProgress = 0;
    const k = easeInOut((elapsed - HOLD_SPUN_END) / (REVEAL_END - HOLD_SPUN_END));
    cubeOpacity = 1 - k;       // panels fade away
    bridgeOpacity = k;         // mesh appears in lockstep
    shadowOpacity = 1 - k;
  } else {
    zoom = ZOOM_OUT;
    unfoldProgress = 0;
    cubeOpacity = 0;
    bridgeOpacity = 1;
    shadowOpacity = 0;
  }

  const bridgeMatRef = useRef<THREE.MeshStandardMaterial>(null);

  // Capture the rotation at the moment the cube finishes folding so the
  // 360° spin starts from THERE instead of snapping back to 0.
  const spinAnchorRef = useRef<number | null>(null);

  useFrame((_, delta) => {
    /* Rotation orchestration:
     *  - Stage 3 has just finished a perfectly-timed 360°, leaving the
     *    shape face-on (sharedRotation ≡ 0). So stage 4 simply locks
     *    rotation at 0 through the entire unfold/zoom-out/fold sequence —
     *    the FRONT silhouette drops in over the bridge sphere with zero
     *    motion. No jiggle, no teleport.
     *  - HOLD_CUBE_END → SPIN_END: a single eased 360° anchored at 0.
     *  - After: gentle drift on the revealed mesh.
     */
    let yRot: number;
    if (elapsed < FOLD_END) {
      yRot = 0;
    } else if (elapsed < HOLD_CUBE_END) {
      if (spinAnchorRef.current === null) spinAnchorRef.current = 0;
      yRot = spinAnchorRef.current;
    } else if (elapsed < SPIN_END) {
      if (spinAnchorRef.current === null) spinAnchorRef.current = 0;
      const k = easeInOut((elapsed - HOLD_CUBE_END) / (SPIN_END - HOLD_CUBE_END));
      yRot = spinAnchorRef.current + k * Math.PI * 2;
    } else {
      const finalAnchor = (spinAnchorRef.current ?? 0) + Math.PI * 2;
      yRot = finalAnchor + (elapsed - SPIN_END) * 0.18;
    }
    sharedRotation.y = yRot;
    if (groupRef.current) {
      groupRef.current.rotation.y = yRot;
      groupRef.current.scale.setScalar(zoom);
    }
    if (bridgeMatRef.current) bridgeMatRef.current.opacity = bridgeOpacity;
    // sphereMatRef retained for back-compat; bridge mesh is now the reveal.
    if (sphereMatRef.current) sphereMatRef.current.opacity = 0;
    if (shadowMatRef.current) shadowMatRef.current.opacity = shadowOpacity;
  });

  /* Hinge configuration per face. In folded state every panel sits flush
   * with one cube face. As progress goes 0→1 we rotate around the shared
   * hinge edge with FRONT until each panel is coplanar with FRONT and
   * forms part of the T-cross net.
   *
   * BACK is special — it cascades off the RIGHT panel rather than the
   * cube itself, so it lands to the right of RIGHT in the unfolded net. */
  const faces: CubeFaceProps[] = [
    {
      face: "front",
      hingePos: [0, 0, half],
      hingeAxis: [0, 0, 0],
      panelOffset: [0, 0, 0],
      foldAngle: 0,
      progress: foldProgressFor("front"),
      opacity: cubeOpacity * faceOpacityFor("front"),
      silhouette: silhouettes.front,
      edge,
    },
    {
      face: "right",
      hingePos: [half, 0, half],
      hingeAxis: [0, 1, 0],
      panelOffset: [half, 0, 0],
      foldAngle: Math.PI / 2,
      progress: foldProgressFor("right"),
      opacity: cubeOpacity * faceOpacityFor("right"),
      silhouette: silhouettes.right,
      edge,
    },
    {
      face: "left",
      hingePos: [-half, 0, half],
      hingeAxis: [0, 1, 0],
      panelOffset: [-half, 0, 0],
      foldAngle: -Math.PI / 2,
      progress: foldProgressFor("left"),
      opacity: cubeOpacity * faceOpacityFor("left"),
      silhouette: silhouettes.left,
      edge,
    },
    {
      face: "top",
      hingePos: [0, half, half],
      hingeAxis: [1, 0, 0],
      panelOffset: [0, half, 0],
      foldAngle: -Math.PI / 2,
      progress: foldProgressFor("top"),
      opacity: cubeOpacity * faceOpacityFor("top"),
      silhouette: silhouettes.top,
      edge,
    },
    {
      face: "bottom",
      hingePos: [0, -half, half],
      hingeAxis: [1, 0, 0],
      panelOffset: [0, -half, 0],
      foldAngle: Math.PI / 2,
      progress: foldProgressFor("bottom"),
      opacity: cubeOpacity * faceOpacityFor("bottom"),
      silhouette: silhouettes.bottom,
      edge,
    },
  ];

  // BACK has TWO hinges:
  //   1) An OUTER cascade hinge attached to RIGHT — this must follow
  //      RIGHT's fold progress so BACK travels along with RIGHT and
  //      doesn't appear to float in space while RIGHT folds.
  //   2) Its OWN inner panel hinge that folds BACK onto the cube
  //      AFTER right has landed (uses BACK's own dedicated window).
  const backCascadeProgress = foldProgressFor("right");
  const backPanelProgress   = foldProgressFor("back");

  return (
    /* Initial scale=ZOOM_IN and rotation=sharedRotation.y are set on the
       element itself (not just inside useFrame) so the very first rendered
       frame already matches the just-finished stage-3 sphere. Without this
       the cube would briefly pop in at scale 1 / rotation 0 before useFrame
       ran — that's the "glitch" the user was seeing. */
    <group
      ref={groupRef}
      scale={ZOOM_IN}
      rotation={[0, 0, 0]}
    >
      {faces.map((p) => (
        <CubeFace key={p.face} {...p} />
      ))}

      {/* BACK face cascades off RIGHT — outer hinge follows RIGHT's
          progress (so they move together), inner panel hinge folds
          afterwards using BACK's own progress window. */}
      <group
        position={[half, 0, half]}
        rotation={[0, (Math.PI / 2) * (1 - backCascadeProgress), 0]}
      >
        <group position={[edge, 0, 0]}>
          <CubeFace
            face="back"
            hingePos={[0, 0, 0]}
            hingeAxis={[0, 1, 0]}
            panelOffset={[half, 0, 0]}
            foldAngle={Math.PI / 2}
            progress={backPanelProgress}
            opacity={cubeOpacity * faceOpacityFor("back")}
            silhouette={silhouettes.back}
            edge={edge}
          />
        </group>
      </group>

      {/* Bridge sphere — same reconstructed mesh shown at the end of stage 3.
          It stays put as the panels fade in over it (giving visual continuity
          between "triangles built" and "unfolded image"), then fades out as
          the camera pulls back to expose the full T-cross net. */}
      <mesh geometry={sourceGeometry}>
        <meshStandardMaterial
          ref={bridgeMatRef}
          color="#f5f7fa"
          roughness={0.32}
          metalness={0.15}
          transparent
          opacity={1}
        />
      </mesh>

      {/* Real reconstructed mesh — fades in at the very end of stage 4. */}
      <mesh geometry={sourceGeometry}>
        <meshStandardMaterial
          ref={sphereMatRef}
          color="#f5f7fa"
          roughness={0.32}
          metalness={0.15}
          transparent
          opacity={0}
        />
      </mesh>

      {/* Soft contact shadow disk — radial-gradient texture on a flat plane
          sitting just under the cube. Fades in with the fold and out with
          the reveal so the cube/mesh visually rests on the surface rather
          than floating in space. Rotated -90° around X to lay flat. */}
      <mesh
        position={[0, -half - 0.02, 0]}
        rotation={[-Math.PI / 2, 0, 0]}
        renderOrder={-1}
      >
        <planeGeometry args={[edge * 2.4, edge * 2.4]} />
        <meshBasicMaterial
          ref={shadowMatRef}
          map={shadowTexture}
          transparent
          opacity={0}
          depthWrite={false}
        />
      </mesh>
    </group>
  );
}

/* ---------------- Axis gizmo ----------------
 * A small XYZ trihedron pinned to the bottom-left of the viewport.
 * Lives in its own <Canvas> overlay so it never interferes with the
 * main scene. It mirrors `sharedRotation.y` each frame so it spins
 * in lockstep with the hero shape across every stage.
 * ----------------------------------------------------------------- */
function GizmoTrihedron() {
  const groupRef = useRef<THREE.Group>(null);
  useFrame(() => {
    if (groupRef.current) groupRef.current.rotation.y = sharedRotation.y;
  });
  // Minimal trihedron: three thin coloured line segments, no arrowheads,
  // no hub. Just X (red), Y (green), Z (blue) emanating from origin.
  const axisLines = useMemo(() => {
    const make = (x: number, y: number, z: number) =>
      new Float32Array([0, 0, 0, x, y, z]);
    return {
      x: make(1, 0, 0),
      y: make(0, 1, 0),
      z: make(0, 0, 1),
    };
  }, []);
  const Axis = ({ array, color }: { array: Float32Array; color: string }) => (
    <lineSegments>
      <bufferGeometry>
        <bufferAttribute
          attach="attributes-position"
          count={2}
          array={array}
          itemSize={3}
        />
      </bufferGeometry>
      <lineBasicMaterial color={color} transparent opacity={0.85} />
    </lineSegments>
  );
  return (
    <group ref={groupRef}>
      <Axis array={axisLines.x} color="#ef4444" />
      <Axis array={axisLines.y} color="#22c55e" />
      <Axis array={axisLines.z} color="#3b82f6" />
    </group>
  );
}

function AxisGizmo() {
  return (
    <div className="pointer-events-none absolute bottom-4 left-4 h-16 w-16">
      <Canvas
        dpr={[1, 2]}
        camera={{ position: [1.6, 1.4, 2.2], fov: 35 }}
        gl={{ antialias: true, alpha: true }}
      >
        <GizmoTrihedron />
      </Canvas>
    </div>
  );
}

/* ---------------- Stage 5: revealed mesh ----------------
 * The final, slowly-rotating reconstructed mesh. Sourced from the same
 * useReconstructedGeometry hook so it'll automatically pick up the user's
 * real .ply mesh once that lands. */
function RevealedMesh() {
  const meshRef = useRef<THREE.Group>(null);
  const sourceGeometry = useReconstructedGeometry();
  useFrame((_, delta) => {
    if (meshRef.current) {
      sharedRotation.y += delta * 0.18;
      meshRef.current.rotation.y = sharedRotation.y;
    }
  });
  return (
    <group ref={meshRef}>
      <mesh geometry={sourceGeometry}>
        <meshStandardMaterial color="#f5f7fa" roughness={0.32} metalness={0.15} />
      </mesh>
    </group>
  );
}

/* ---------------- Depth-perception vertical net ----------------
 * Unfolded cube net rotated to read top-to-bottom: the spine
 * (TOP → FRONT → BOTTOM → BACK) runs vertically, with LEFT and
 * RIGHT branching off FRONT. Each cell is an empty placeholder
 * for the orthographic depth render that will be plugged in later.
 *
 *            [ TOP   ]
 *   [LEFT]  [ FRONT ]  [RIGHT]
 *            [BOTTOM ]
 *            [ BACK  ]
 *
 * Cells animate in via a left→right slice scan so the panel
 * composes itself like a depth scanner pass.
 * --------------------------------------------------------------- */
const DEPTH_CELLS: { id: string; row: number; col: number; order: number }[] = [
  { id: "TOP",    row: 1, col: 2, order: 0 },
  { id: "LEFT",   row: 2, col: 1, order: 1 },
  { id: "FRONT",  row: 2, col: 2, order: 2 },
  { id: "RIGHT",  row: 2, col: 3, order: 3 },
  { id: "BOTTOM", row: 3, col: 2, order: 4 },
  { id: "BACK",   row: 4, col: 2, order: 5 },
];
const DEPTH_SLICES_PER_CELL = 12;

function DepthColumn() {
  return (
    <div className="depth-panel-in flex h-full max-h-full w-full max-w-[340px] flex-col items-start gap-4">
      <div className="font-wordmark text-[13px] font-medium leading-snug tracking-tight text-foreground/80">
        Depth Perception
      </div>
      <div
        className="grid w-full overflow-hidden rounded-md border border-border bg-surface/60 backdrop-blur-sm"
        style={{
          gridTemplateColumns: "repeat(3, 1fr)",
          gridTemplateRows: "repeat(4, 1fr)",
          aspectRatio: "3 / 4",
        }}
      >
        {DEPTH_CELLS.map((c) => {
          // Borders only on shared edges so cells read as one connected net,
          // and only when the neighbouring grid cell is actually occupied.
          const has = (row: number, col: number) =>
            DEPTH_CELLS.some((d) => d.row === row && d.col === col);
          const borderRight =
            has(c.row, c.col + 1) ? "1px solid hsl(var(--border))" : "none";
          const borderBottom =
            has(c.row + 1, c.col) ? "1px solid hsl(var(--border))" : "none";
          return (
            <div
              key={c.id}
              style={{
                gridColumn: c.col,
                gridRow: c.row,
                animationDelay: `${c.order * 110}ms`,
                borderRight,
                borderBottom,
              }}
              className="depth-cell relative flex h-full w-full items-center justify-center overflow-hidden"
            >
              <div className="pointer-events-none absolute inset-0 flex">
                {Array.from({ length: DEPTH_SLICES_PER_CELL }).map((_, i) => (
                  <div
                    key={i}
                    className="depth-slice h-full flex-1 bg-foreground/[0.04]"
                    style={{
                      animationDelay: `${c.order * 110 + 200 + i * 30}ms`,
                      borderRight:
                        i < DEPTH_SLICES_PER_CELL - 1
                          ? "1px solid hsl(var(--border) / 0.18)"
                          : "none",
                    }}
                  />
                ))}
              </div>
              <span className="relative z-10 font-mono text-[9px] uppercase tracking-[0.28em] text-muted-foreground/70">
                {c.id}
              </span>
            </div>
          );
        })}
      </div>
    </div>
  );
}

/* ---------------- Page ---------------- */

/* ---------------- Phase narration ----------------
 * Subtle left-anchored micro-captions that describe what's happening on
 * screen right now. Each phase has a tiny line; lines cross-fade as the
 * underlying animation moves between phases. Lives outside any Canvas so
 * we can use plain CSS for the fades.
 *
 * Timings here MUST stay in sync with the corresponding scene logic
 * (PipelineScene's HOLD_END/BUILD_END/etc. and UnfoldFoldScene's phase
 * boundaries). If you change one, change both.
 * --------------------------------------------------- */
interface PhaseStep {
  /** Local time within the stage (seconds) at which this caption becomes active. */
  at: number;
  /** Tiny eyebrow label rendered above the caption (mono, uppercase). */
  label: string;
  /** Caption body — short phrase describing what's currently happening. */
  text: string;
}

const PHASE_STEPS: Record<2 | 3 | 4, PhaseStep[]> = {
  // Stage 2 (2.5s total) — denoising the raw scan
  2: [
    { at: 0.0, label: "", text: "Filtering noise from the raw scan" },
    { at: 1.3, label: "", text: "Stabilising point density" },
  ],
  // Stage 3 (4.3s total) — HOLD_END=0.7, BUILD_END=2.8, points fade after
  3: [
    { at: 0.0, label: "", text: "Holding the cleaned point cloud" },
    { at: 0.7, label: "", text: "Stitching triangles across the surface" },
    { at: 2.8, label: "", text: "Resolving the watertight mesh" },
    { at: 3.4, label: "", text: "Fading the underlying points" },
  ],
  // Stage 4 (13.1s total) — see UnfoldFoldScene timeline
  4: [
    { at: 0.0,  label: "", text: "Projecting orthographic views around the mesh" },
    { at: 2.5,  label: "", text: "Holding the unfolded net" },
    { at: 2.8,  label: "", text: "Pulling the camera back to reveal the layout" },
    { at: 4.1,  label: "", text: "Folding the net into a closed cube" },
    { at: 6.3,  label: "", text: "Holding the assembled cube" },
    { at: 6.7,  label: "", text: "Rotating the cube a full 360°" },
    { at: 9.9,  label: "", text: "Holding after the spin" },
    { at: 11.0, label: "", text: "Dissolving the cube, exposing the mesh" },
  ],
};

function PhaseNarration({ stage, elapsed }: { stage: Stage; elapsed: number }) {
  const steps = PHASE_STEPS[stage as 2 | 3 | 4];

  // Always call hooks unconditionally
  let activeIdx = 0;
  if (steps) {
    for (let i = 0; i < steps.length; i++) {
      if (elapsed >= steps[i].at) activeIdx = i;
      else break;
    }
  }
  // Unique key for the (stage, step) pair so React treats each phase
  // as a different element and triggers the flip animation.
  const phaseKey = steps ? `${stage}-${activeIdx}` : "none";

  // Track the previous phase so we can render the OUTGOING line on top
  // of the incoming one for the duration of the flip.
  const [prevKey, setPrevKey] = useState<string | null>(null);
  const [prevText, setPrevText] = useState<string>("");
  const lastKeyRef = useRef<string>(phaseKey);

  useEffect(() => {
    if (lastKeyRef.current !== phaseKey) {
      // Capture the line that's leaving so we can animate it out.
      const prevIdx = steps
        ? Math.max(0, parseInt(lastKeyRef.current.split("-")[1] ?? "0", 10))
        : 0;
      setPrevKey(lastKeyRef.current);
      setPrevText(steps?.[prevIdx]?.text ?? "");
      lastKeyRef.current = phaseKey;
      // Clear the outgoing line once its animation is done.
      const t = setTimeout(() => setPrevKey(null), 700);
      return () => clearTimeout(t);
    }
  }, [phaseKey, steps]);

  if (!steps) return null;
  const active = steps[activeIdx];

  return (
    <div className="pointer-events-none absolute left-2 top-1/2 z-20 w-[min(22rem,92vw)] max-w-[460px] -translate-y-1/2 sm:left-6 sm:w-[min(24rem,88vw)] md:left-9 lg:left-10">
      {/* Single line, flip transition. The wrapper preserves vertical
          space so the layout doesn't jump while the line is mid-flip. */}
      <div
        className="relative h-6 overflow-visible text-[13px] font-light leading-6 tracking-tight text-foreground/85"
        style={{ perspective: "600px" }}
      >
        {prevKey && (
          <p
            key={prevKey}
            className="phase-flip-out absolute inset-0 m-0 whitespace-nowrap"
          >
            {prevText}
          </p>
        )}
        <p
          key={phaseKey}
          className="phase-flip-in absolute inset-0 m-0 whitespace-nowrap"
        >
          {active.text}
        </p>
      </div>
    </div>
  );
}


const Workflow = () => {
  const navigate = useNavigate();
  const [stage, setStage] = useState<Stage>(0);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [stageElapsed, setStageElapsed] = useState(0);

  // Per-stage durations (ms). Stage 4 hosts the unfold→fold→spin→fade
  // showcase, so it gets the most time.
  const STAGE_DURATIONS: Record<Exclude<Stage, 0 | 5>, number> = {
    1: 2500,
    2: 2500,
    3: 4300,
    4: 13100,
  };

  // Weights used to compute an overall pipeline progress (0–100).
  const STAGE_WEIGHTS: Record<Exclude<Stage, 0 | 5>, number> = {
    1: 8,
    2: 16,
    3: 36,
    4: 40,
  };
  const TOTAL_WEIGHT = Object.values(STAGE_WEIGHTS).reduce((a, b) => a + b, 0);

  // Drive the upload progress when in stage 1.
  useEffect(() => {
    if (stage !== 1) return;
    setUploadProgress(0);
    const id = window.setInterval(() => {
      setUploadProgress((p) => {
        if (p >= 100) {
          window.clearInterval(id);
          return 100;
        }
        return Math.min(100, p + 4 + Math.random() * 6);
      });
    }, 80);
    return () => window.clearInterval(id);
  }, [stage]);

  // Auto-advance through stages once started.
  useEffect(() => {
    if (stage === 0 || stage === 5) return;
    const id = window.setTimeout(
      () => setStage((s) => ((s + 1) as Stage)),
      STAGE_DURATIONS[stage as Exclude<Stage, 0 | 5>],
    );
    return () => window.clearTimeout(id);
  }, [stage]);

  // Track elapsed time within the current stage to drive the loading bar.
  useEffect(() => {
    setStageElapsed(0);
    if (stage === 0 || stage === 5) return;
    const start = performance.now();
    const id = window.setInterval(() => {
      setStageElapsed(performance.now() - start);
    }, 80);
    return () => window.clearInterval(id);
  }, [stage]);

  // Compute overall pipeline progress (0–100) across all stages.
  const overallProgress =
    stage === 0
      ? 0
      : stage === 5
        ? 100
        : (() => {
            const completed = Object.entries(STAGE_WEIGHTS)
              .filter(([id]) => Number(id) < stage)
              .reduce((a, [, w]) => a + w, 0);
            const dur = STAGE_DURATIONS[stage as Exclude<Stage, 0 | 5>];
            const w = STAGE_WEIGHTS[stage as Exclude<Stage, 0 | 5>];
            const within = Math.min(1, stageElapsed / dur) * w;
            return ((completed + within) / TOTAL_WEIGHT) * 100;
          })();

  return (
    <main className="relative flex h-screen-dvh min-h-0 w-full flex-col overflow-hidden stage-bg text-foreground site-pad-bottom">
      {/* Faint blueprint grid backdrop */}
      <div
        aria-hidden
        className="pointer-events-none absolute inset-0 opacity-[0.42]"
        style={{
          backgroundImage:
            "linear-gradient(hsl(220 14% 84% / 0.82) 1px, transparent 1px), linear-gradient(90deg, hsl(220 14% 84% / 0.82) 1px, transparent 1px)",
          backgroundSize: "64px 64px",
          maskImage:
            "radial-gradient(ellipse at 50% 40%, black 0%, transparent 75%)",
          WebkitMaskImage:
            "radial-gradient(ellipse at 50% 40%, black 0%, transparent 75%)",
        }}
      />

      {/* Top bar */}
      <header className="site-gutter-x relative z-10 flex flex-shrink-0 items-center justify-between py-2.5 sm:py-3">
        <Link
          to="/"
          className="inline-flex min-h-10 min-w-10 items-center justify-center gap-1.5 rounded-lg text-sm font-medium text-muted-foreground -ml-1 px-0.5 transition-colors hover:text-foreground sm:min-h-0 sm:min-w-0 sm:justify-start sm:rounded-none"
        >
          <ArrowLeft className="h-4 w-4 shrink-0" strokeWidth={1.8} />
          <span>Back</span>
        </Link>
        <div>
          <span className="font-wordmark text-[length:clamp(1.35rem,calc(0.5rem+1.1vw),1.85rem)] font-bold leading-none tracking-[-0.02em] text-foreground sm:text-3xl">
            CAD
          </span>
          <span className="font-wordmark text-[length:clamp(1.35rem,calc(0.5rem+1.1vw),1.85rem)] font-light italic leading-none tracking-[-0.02em] text-foreground/80 sm:text-3xl">
            abra
          </span>
        </div>
        {/* Right slot left intentionally empty — kept for layout balance */}
        <span aria-hidden className="w-10 min-[480px]:w-[3.25rem] sm:w-[3.75rem]" />
      </header>

      {/* Main column: viewport + loading bar — flexes to fill remaining height */}
      <section className="site-gutter-x relative z-10 mx-auto mt-1 flex w-full max-w-[min(100%,87.5rem)] flex-1 min-h-0 flex-col pb-2 sm:pb-3 2xl:max-w-[min(100%,96rem)]">
        <div className="flex min-h-0 flex-1 flex-col">
          <div className="relative flex-1 min-h-0 overflow-hidden">

            {/* Stage 0 — symmetric two-column hero. The grid is split
                into two rows so the title aligns with the brick and the
                paragraph aligns with the upload button. */}
            {stage === 0 && (
              <div className="absolute inset-0 animate-fade-in">
                <div className="flex h-full w-full items-stretch justify-center px-3 py-6 sm:px-5 sm:py-8 md:px-8 md:py-8 lg:px-10 lg:py-10">
                  <div
                    className="grid h-full w-full min-h-0 max-w-[920px] grid-cols-1 gap-y-8 md:grid-cols-2 md:gap-y-0"
                    style={{
                      columnGap: "min(1.5rem, 3vw)",
                    }}
                  >
                    {/* LEFT — Title + description, both nudged downward
                        and inset toward the center. */}
                    <div className="order-2 flex h-full min-h-0 flex-col items-center pl-0 pt-4 sm:pl-2 md:order-1 md:pl-4 md:pt-10 lg:pl-5 lg:pt-16">
                      <div className="w-full">
                        <ScanToCadTitle className="text-foreground" />
                      </div>
                      <div className="mt-4 flex w-full justify-center sm:mt-5 md:mt-6">
                        <div
                          className="w-full max-w-[22.5rem] sm:max-w-[20rem] md:max-w-[18rem] lg:max-w-[20rem] xl:max-w-[22.5rem]"
                          style={{ height: "min(50vh, 20rem, 320px)" }}
                        >
                          <CanvasReflowText
                            text="Drop in a raw .ply point cloud, the messy noisy swarm of 3D dots a scanner spits out after capturing a real object. CADabra cleans it up: it strips out stray points, smooths the surface, and stitches the dots into a watertight triangle mesh that actually looks like the thing you scanned. Then it spins the model into six clean orthographic views (front, back, left, right, top, bottom), the exact projections an engineer needs to rebuild the part in CAD."
                            lineHeight={22}
                            font='italic 300 14px Inter, ui-sans-serif, system-ui, sans-serif'
                            duration={6000}
                            startDelay={350}
                            className="text-foreground"
                          />
                        </div>
                      </div>
                    </div>

                    {/* RIGHT — Brick + upload, nudged slightly higher
                        and inset toward the center. */}
                    <div className="order-1 flex h-full min-h-[min(50vh,22rem)] flex-col items-center pr-0 sm:pr-2 md:order-2 md:min-h-0 md:pr-4 md:pt-0 lg:pr-5">
                      <div className="relative mx-auto min-h-0 w-full max-w-[min(32rem,92vw)] flex-1 -translate-y-0 sm:-translate-y-1 md:max-w-[520px] md:-translate-y-3">
                        <LegoPodiumScene />
                      </div>
                      <div className="-mt-10 flex w-full flex-col items-center gap-3" />
                    </div>
                  </div>
                  <div className="pointer-events-none absolute inset-x-0 bottom-8 z-10 flex flex-col items-center gap-3">
                    <button
                      onClick={() => setStage(1)}
                      className="pointer-events-auto group inline-flex w-full max-w-xs items-center justify-center gap-2.5 rounded-full bg-white px-6 py-4 text-sm font-semibold uppercase tracking-[0.18em] text-black shadow-lg transition-all hover:-translate-y-0.5 hover:shadow-xl sm:max-w-md sm:px-8 sm:py-4 sm:text-base md:w-auto md:max-w-none md:px-10 md:py-5"
                    >
                      <Upload className="h-4 w-4" strokeWidth={1.8} />
                      UPLOAD POINT CLOUD
                    </button>
                    <p className="text-xs font-light text-muted-foreground">
                      (.ply file of point cloud.)
                    </p>
                  </div>
                </div>
              </div>
            )}

            {/* Stage 1 — plain backdrop, hairline progress at the bottom */}
            {stage === 1 && (
              <div className="absolute inset-0 animate-fade-in">
                <div className="absolute inset-x-0 top-[46%] z-10 -translate-y-1/2 px-6 text-center">
                  <p className="text-sm font-medium tracking-[0.03em] text-foreground/80">
                    abraCADabring...
                  </p>
                </div>
                {/* Bottom-anchored hairline progress */}
                <div className="absolute inset-x-0 bottom-16 z-10 mx-auto flex max-w-sm flex-col items-center gap-3 px-6">
                  <div className="font-mono text-[10px] uppercase tracking-[0.35em] text-muted-foreground">
                    {String(Math.round(uploadProgress)).padStart(2, "0")}
                  </div>
                  <div className="relative h-px w-full overflow-hidden bg-border/70">
                    <div
                      className="relative h-full overflow-hidden bg-foreground transition-[width] duration-150 ease-out"
                      style={{ width: `${uploadProgress}%` }}
                    >
                      <span
                        aria-hidden
                        className="absolute inset-y-0 -left-1/3 w-1/3"
                        style={{
                          background:
                            "linear-gradient(90deg, transparent, hsl(0 0% 100% / 0.7), transparent)",
                          animation: "shimmer-bar 1.4s linear infinite",
                        }}
                      />
                    </div>
                  </div>
                </div>
              </div>
            )}


            {/* Stages 2 – 3 — point cloud → triangle reconstruction */}
            {stage >= 2 && stage <= 3 && (
              <Canvas
                dpr={[1, 2]}
                camera={{ position: [0, 0, 4.5], fov: 38 }}
                gl={{ antialias: true, alpha: true }}
              >
                <ambientLight intensity={0.85} />
                <directionalLight position={[4, 5, 3]} intensity={0.7} />
                <directionalLight position={[-3, -2, -1]} intensity={0.25} />
                <Suspense fallback={null}>
                  <PipelineScene stage={stage} />
                </Suspense>
              </Canvas>
            )}

            {/* Stage 4 — unfold T-cross → fold cube → spin → reveal */}
            {stage === 4 && (
              <Canvas
                dpr={[1, 2]}
                /* Straight-on camera so the unfolded T-cross net reads as a
                   flat orthographic-style layout when stage 4 mounts. Same
                   FOV as stage 3 for perfect continuity. */
                camera={{ position: [0, 0, 8.6], fov: 38 }}
                gl={{ antialias: true, alpha: true }}
                onCreated={({ camera }) => camera.lookAt(0, 0, 0)}
              >
                {/* Lighting matched to stage 3 so the bridge sphere has the
                    exact same shading the moment stage 4 mounts — no flash. */}
                <ambientLight intensity={0.85} />
                <directionalLight position={[4, 5, 3]} intensity={0.7} />
                <directionalLight position={[-3, -2, -1]} intensity={0.25} />
                <Suspense fallback={null}>
                  <UnfoldFoldScene elapsed={stageElapsed / 1000} />
                </Suspense>
              </Canvas>
            )}

            {/* Subtle left-side narration — micro-captions explaining the
                current phase of the animation. Visible during stages 2-4
                (point-cloud → reconstruct → unfold/fold). Each line fades
                in/out as its phase becomes active. */}
            {stage >= 2 && stage <= 4 && (
              <PhaseNarration stage={stage} elapsed={stageElapsed / 1000} />
            )}

            {/* Stage 5 — split: vertical depth column on the left, mesh shifts right */}
            {stage === 5 && (
              <div className="absolute inset-0 flex min-h-0 flex-col md:flex-row animate-fade-in">
                {/* LEFT: vertical column of orthographic depth placeholders */}
                <div className="flex w-full min-w-0 max-h-[38vh] items-center justify-center border-b border-border/40 px-4 py-3 sm:max-h-[40vh] md:w-[34%] md:min-h-0 md:max-h-none md:min-w-[240px] md:max-w-[340px] md:border-b-0 md:px-6 md:py-6">
                  <DepthColumn />
                </div>
                {/* RIGHT: rotating mesh slides into place + Next button */}
                <div className="mesh-shift-right relative flex min-h-0 flex-1 items-center justify-center max-md:min-h-[45vh]">
                  <Canvas
                    dpr={[1, 2]}
                    camera={{ position: [0, 2.6, 8.6], fov: 38 }}
                    gl={{ antialias: true, alpha: true }}
                    onCreated={({ camera }) => camera.lookAt(0, 0, 0)}
                  >
                    <ambientLight intensity={0.85} />
                    <directionalLight position={[4, 5, 3]} intensity={0.7} />
                    <directionalLight position={[-3, -2, -1]} intensity={0.25} />
                    <Suspense fallback={null}>
                      <RevealedMesh />
                    </Suspense>
                  </Canvas>
                  <button
                    onClick={() => navigate("/demo")}
                    className="absolute bottom-6 right-6 inline-flex items-center gap-2 rounded-full bg-foreground px-5 py-2.5 text-xs font-medium uppercase tracking-[0.2em] text-background shadow-md transition-all hover:shadow-lg"
                  >
                    Next
                    <ArrowRight className="h-3.5 w-3.5" strokeWidth={2} />
                  </button>
                </div>
              </div>
            )}


            {/* Bottom-left axis gizmo — visible across every 3D stage,
                rotates in lockstep with the hero shape. */}
            {stage >= 2 && <AxisGizmo />}
          </div>

          {/* Overall pipeline loading bar */}
          {stage !== 0 && (
            <div className="mt-4 flex-shrink-0 animate-fade-in">
              <div className="mb-2 flex items-center justify-between text-[10px] font-medium uppercase tracking-[0.3em] text-muted-foreground">
                <span>{STAGE_LABELS[stage]}</span>
                <span className="font-mono">
                  {String(Math.round(overallProgress)).padStart(3, "0")}%
                </span>
              </div>
              <div className="h-1 w-full overflow-hidden rounded-full bg-border">
                <div
                  className="h-full bg-foreground transition-[width] duration-200 ease-out"
                  style={{ width: `${overallProgress}%` }}
                />
              </div>
            </div>
          )}
        </div>
      </section>
    </main>
  );
};

export default Workflow;
