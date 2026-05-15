import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import * as THREE from "three";
import { Canvas, ThreeEvent, useFrame, useThree } from "@react-three/fiber";
import { ContactShadows, Environment, Html, OrbitControls } from "@react-three/drei";
import { Podium } from "./Podium";

const MM_TO_WORLD = 0.01;
const BASE_COLOR = "#8f97a3";
const HIGHLIGHT_COLOR = "#3b9bff";

const HOLE_1_OFFSET_X = 57.44290063889225;
const HOLE_1_OFFSET_Y = 0.0;
const HOLE_1_RADIUS_MM = 4.382203223295068;

const HOLE_2_OFFSET_X = -54.76796401363055;
const HOLE_2_OFFSET_Y = 0.0;
const HOLE_2_RADIUS_MM = 7.655201646000145;

type EditableSurfaceKey = "top";

export interface Sample128105Params {
  extrudeMm: number;
}

export const SAMPLE_128105_INITIAL_PARAMS: Sample128105Params = {
  extrudeMm: 38.25734937230923,
};

export function getSample128105TopY(params: Sample128105Params) {
  return params.extrudeMm * MM_TO_WORLD + 0.05;
}

type Segment =
  | {
      kind: "line";
      end: [number, number];
    }
  | {
      kind: "arc";
      mid: [number, number];
      end: [number, number];
    };

const PROFILE_START: [number, number] = [-73.31339532574692, 4.173822844523812];

const PROFILE_SEGMENTS: Segment[] = [
  {
    kind: "arc",
    mid: [-67.2171638901068, -14.530932361700883],
    end: [-47.693072714486476, -16.947907497638475],
  },
  { kind: "line", end: [-37.865742376587946, -11.23951848917491] },
  { kind: "line", end: [42.978469082083556, -11.395806196007005] },
  {
    kind: "arc",
    mid: [49.88663267757694, -15.607095054057963],
    end: [57.95546738039545, -15.875394088100446],
  },
  { kind: "line", end: [65.80970591615583, -13.477059339051506] },
  {
    kind: "arc",
    mid: [72.65400945295458, 3.5732316960801516],
    end: [60.51266566213822, 16.329938188888615],
  },
  { kind: "line", end: [52.746302964375275, 16.60531759989592] },
  { kind: "line", end: [42.28404499838954, 12.146816232046145] },
  { kind: "line", end: [-37.76895767819464, 12.046995121493119] },
  { kind: "line", end: [-46.97801596997432, 17.76571657757136] },
  {
    kind: "arc",
    mid: [-52.98608820218317, 19.337043596683674],
    end: [-59.19038560990996, 18.911456506081024],
  },
  { kind: "line", end: [-69.22297550442627, 13.687621767974216] },
  { kind: "line", end: [-73.31339532574692, 4.173822844523812] },
];

function sampleThreePointArc(
  start: [number, number],
  mid: [number, number],
  end: [number, number],
  segments = 30,
): THREE.Vector2[] {
  const ax = start[0];
  const ay = start[1];
  const bx = mid[0];
  const by = mid[1];
  const cx = end[0];
  const cy = end[1];

  const d = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by));
  const pts: THREE.Vector2[] = [];

  if (Math.abs(d) < 1e-9) {
    for (let i = 1; i <= segments; i++) {
      const t = i / segments;
      pts.push(new THREE.Vector2(ax + (cx - ax) * t, ay + (cy - ay) * t));
    }
    return pts;
  }

  const ax2 = ax * ax + ay * ay;
  const bx2 = bx * bx + by * by;
  const cx2 = cx * cx + cy * cy;
  const centerX = (ax2 * (by - cy) + bx2 * (cy - ay) + cx2 * (ay - by)) / d;
  const centerY = (ax2 * (cx - bx) + bx2 * (ax - cx) + cx2 * (bx - ax)) / d;

  const radius = Math.hypot(ax - centerX, ay - centerY);
  const startAngle = Math.atan2(ay - centerY, ax - centerX);
  const midAngle = Math.atan2(by - centerY, bx - centerX);
  const endAngle = Math.atan2(cy - centerY, cx - centerX);

  const norm = (a: number) => {
    let x = a;
    while (x < 0) x += Math.PI * 2;
    while (x >= Math.PI * 2) x -= Math.PI * 2;
    return x;
  };

  const sweepCCW = norm(endAngle - startAngle);
  const midCCW = norm(midAngle - startAngle);
  const goCCW = midCCW <= sweepCCW;
  const delta = goCCW ? sweepCCW : -(Math.PI * 2 - sweepCCW);

  for (let i = 1; i <= segments; i++) {
    const a = startAngle + (delta * i) / segments;
    pts.push(new THREE.Vector2(centerX + Math.cos(a) * radius, centerY + Math.sin(a) * radius));
  }
  return pts;
}

function getProfilePoints(): THREE.Vector2[] {
  const pts: THREE.Vector2[] = [new THREE.Vector2(PROFILE_START[0], PROFILE_START[1])];
  let cursor: [number, number] = PROFILE_START;
  for (const seg of PROFILE_SEGMENTS) {
    if (seg.kind === "line") {
      pts.push(new THREE.Vector2(seg.end[0], seg.end[1]));
      cursor = seg.end;
    } else {
      const arcPts = sampleThreePointArc(cursor, seg.mid, seg.end);
      arcPts.forEach((p) => pts.push(p));
      cursor = seg.end;
    }
  }
  return pts;
}

function getProfileBBoxCenter(): { x: number; y: number } {
  const pts = getProfilePoints();
  let minX = Infinity;
  let maxX = -Infinity;
  let minY = Infinity;
  let maxY = -Infinity;
  for (const p of pts) {
    if (p.x < minX) minX = p.x;
    if (p.x > maxX) maxX = p.x;
    if (p.y < minY) minY = p.y;
    if (p.y > maxY) maxY = p.y;
  }
  return { x: (minX + maxX) / 2, y: (minY + maxY) / 2 };
}

function circlePath(radiusMm: number, cxMm: number, cyMm: number): THREE.Path {
  const path = new THREE.Path();
  path.absellipse(
    cxMm * MM_TO_WORLD,
    cyMm * MM_TO_WORLD,
    radiusMm * MM_TO_WORLD,
    radiusMm * MM_TO_WORLD,
    0,
    Math.PI * 2,
    false,
    0,
  );
  return path;
}

function buildSample128105Shape(): THREE.Shape {
  const pts = getProfilePoints();
  const shape = new THREE.Shape();
  shape.moveTo(pts[0].x * MM_TO_WORLD, pts[0].y * MM_TO_WORLD);
  for (let i = 1; i < pts.length; i++) {
    shape.lineTo(pts[i].x * MM_TO_WORLD, pts[i].y * MM_TO_WORLD);
  }
  shape.closePath();

  const bbox = getProfileBBoxCenter();
  shape.holes.push(
    circlePath(HOLE_1_RADIUS_MM, bbox.x + HOLE_1_OFFSET_X, bbox.y + HOLE_1_OFFSET_Y),
  );
  shape.holes.push(
    circlePath(HOLE_2_RADIUS_MM, bbox.x + HOLE_2_OFFSET_X, bbox.y + HOLE_2_OFFSET_Y),
  );
  return shape;
}

export function buildSample128105Geometry(params: Sample128105Params): THREE.BufferGeometry {
  const shape = buildSample128105Shape();
  const geom = new THREE.ExtrudeGeometry(shape, {
    depth: params.extrudeMm * MM_TO_WORLD,
    bevelEnabled: false,
    curveSegments: 48,
  });
  geom.rotateX(-Math.PI / 2);
  geom.computeVertexNormals();
  return geom;
}

function SurfaceShape({
  y,
  shape,
  hovered,
  active,
  onPointerDown,
  onPointerOver,
  onPointerOut,
}: {
  y: number;
  shape: THREE.Shape;
  hovered: boolean;
  active: boolean;
  onPointerDown: (e: ThreeEvent<PointerEvent>) => void;
  onPointerOver: (e: ThreeEvent<PointerEvent>) => void;
  onPointerOut: (e: ThreeEvent<PointerEvent>) => void;
}) {
  const geom = useMemo(() => new THREE.ShapeGeometry(shape, 48), [shape]);
  return (
    <mesh
      geometry={geom}
      rotation={[-Math.PI / 2, 0, 0]}
      position={[0, y, 0]}
      onPointerDown={onPointerDown}
      onPointerOver={onPointerOver}
      onPointerOut={onPointerOut}
    >
      <meshBasicMaterial
        color={HIGHLIGHT_COLOR}
        transparent
        opacity={active ? 0.3 : hovered ? 0.18 : 0}
        depthWrite={false}
        side={THREE.DoubleSide}
      />
    </mesh>
  );
}

function SurfaceLabel({
  y,
  text,
}: {
  y: number;
  text: string;
}) {
  return (
    <Html position={[0, y, 0]} center distanceFactor={7} zIndexRange={[100, 0]} style={{ pointerEvents: "none" }}>
      <div className="floating-label whitespace-nowrap">{text}</div>
    </Html>
  );
}

function ParametricPart({
  params,
  activeSurface,
  liveValueMm,
}: {
  params: Sample128105Params;
  activeSurface: EditableSurfaceKey | null;
  liveValueMm: number | null;
}) {
  const geom = useMemo(() => buildSample128105Geometry(params), [params]);
  const material = useMemo(
    () =>
      new THREE.MeshStandardMaterial({
        color: BASE_COLOR,
        roughness: 0.28,
        metalness: 0.12,
      }),
    [],
  );

  useEffect(() => () => {
    geom.dispose();
    material.dispose();
  }, [geom, material]);

  const topY = params.extrudeMm * MM_TO_WORLD;

  return (
    <group position={[0, 0.05, 0]}>
      <mesh geometry={geom} material={material} castShadow receiveShadow />
      {activeSurface && liveValueMm !== null && (
        <SurfaceLabel y={topY + 0.18} text={`${Math.round(liveValueMm)} mm`} />
      )}
    </group>
  );
}

export function Sample128105Solid({
  params = SAMPLE_128105_INITIAL_PARAMS,
  color = BASE_COLOR,
  opacity = 1,
}: {
  params?: Sample128105Params;
  color?: string;
  opacity?: number;
}) {
  const geom = useMemo(() => buildSample128105Geometry(params), [params]);
  const material = useMemo(
    () =>
      new THREE.MeshStandardMaterial({
        color,
        roughness: 0.28,
        metalness: 0.12,
        transparent: opacity < 1,
        opacity,
      }),
    [color, opacity],
  );

  useEffect(
    () => () => {
      geom.dispose();
      material.dispose();
    },
    [geom, material],
  );

  return (
    <group position={[0, 0.05, 0]}>
      <mesh geometry={geom} material={material} castShadow receiveShadow />
    </group>
  );
}

function EditorObject({
  params,
  onChange,
  onDraggingChange,
  interactive = true,
}: {
  params: Sample128105Params;
  onChange: (next: Sample128105Params) => void;
  onDraggingChange?: (dragging: boolean) => void;
  interactive?: boolean;
}) {
  const groupRef = useRef<THREE.Group>(null);
  const { gl, camera } = useThree();
  const [hovered, setHovered] = useState<EditableSurfaceKey | null>(null);
  const [dragging, setDragging] = useState<EditableSurfaceKey | null>(null);
  const [liveParams, setLiveParams] = useState(params);
  const dragStateRef = useRef<{
    surface: EditableSurfaceKey;
    startY: number;
    pixelsPerUnit: number;
    startParams: Sample128105Params;
  } | null>(null);

  useEffect(() => {
    if (!dragging) setLiveParams(params);
  }, [params, dragging]);

  const shapeTop = useMemo(() => buildSample128105Shape(), []);

  const computePixelsPerUnit = useCallback((worldOrigin: THREE.Vector3) => {
    const a = worldOrigin.clone().project(camera);
    const b = worldOrigin.clone().add(new THREE.Vector3(0, 1, 0)).project(camera);
    const rect = gl.domElement.getBoundingClientRect();
    const dx = ((b.x - a.x) * rect.width) / 2;
    const dy = ((a.y - b.y) * rect.height) / 2;
    return Math.max(Math.hypot(dx, dy), 40);
  }, [camera, gl]);

  const beginDrag = useCallback((surface: EditableSurfaceKey, worldY: number) => (e: ThreeEvent<PointerEvent>) => {
    if (!interactive) return;
    e.stopPropagation();
    (e.target as Element)?.setPointerCapture?.(e.pointerId);
    const worldOrigin = new THREE.Vector3(0, worldY, 0);
    groupRef.current?.localToWorld(worldOrigin);
    dragStateRef.current = {
      surface,
      startY: e.clientY,
      pixelsPerUnit: computePixelsPerUnit(worldOrigin),
      startParams: params,
    };
    setDragging(surface);
    onDraggingChange?.(true);
    gl.domElement.style.cursor = "ns-resize";
  }, [computePixelsPerUnit, gl, interactive, onDraggingChange, params]);

  useEffect(() => {
    const handleMove = (ev: PointerEvent) => {
      const state = dragStateRef.current;
      if (!state) return;
      const deltaMm = ((state.startY - ev.clientY) / state.pixelsPerUnit) / MM_TO_WORLD;
      const next = { ...state.startParams };
      next.extrudeMm = THREE.MathUtils.clamp(state.startParams.extrudeMm + deltaMm, 8, 80);
      setLiveParams(next);
    };
    const handleUp = () => {
      if (!dragStateRef.current) return;
      onChange(liveParams);
      dragStateRef.current = null;
      setDragging(null);
      onDraggingChange?.(false);
      gl.domElement.style.cursor = "default";
    };
    window.addEventListener("pointermove", handleMove);
    window.addEventListener("pointerup", handleUp);
    window.addEventListener("pointercancel", handleUp);
    return () => {
      window.removeEventListener("pointermove", handleMove);
      window.removeEventListener("pointerup", handleUp);
      window.removeEventListener("pointercancel", handleUp);
    };
  }, [gl, liveParams, onChange, onDraggingChange]);

  const shown = dragging ? liveParams : params;
  const activeHover = interactive ? hovered : null;
  const topY = shown.extrudeMm * MM_TO_WORLD + 0.05;
  const liveValueMm = dragging === "top" ? shown.extrudeMm : null;

  return (
    <group ref={groupRef}>
      <ParametricPart params={shown} activeSurface={dragging} liveValueMm={liveValueMm} />
      <SurfaceShape
        y={topY}
        shape={shapeTop}
        hovered={activeHover === "top"}
        active={dragging === "top"}
        onPointerDown={beginDrag("top", topY)}
        onPointerOver={(e) => {
          e.stopPropagation();
          if (interactive) setHovered("top");
          if (interactive && !dragging) gl.domElement.style.cursor = "ns-resize";
        }}
        onPointerOut={() => {
          setHovered((prev) => (prev === "top" ? null : prev));
          if (!dragging) gl.domElement.style.cursor = "default";
        }}
      />
    </group>
  );
}

export function Sample128105EditableSolid({
  params,
  onChange,
  onDraggingChange,
  interactive = true,
}: {
  params: Sample128105Params;
  onChange: (next: Sample128105Params) => void;
  onDraggingChange?: (dragging: boolean) => void;
  interactive?: boolean;
}) {
  return (
    <EditorObject
      params={params}
      onChange={onChange}
      onDraggingChange={onDraggingChange}
      interactive={interactive}
    />
  );
}

function Stage({
  params,
  onChange,
  interactive = true,
}: {
  params: Sample128105Params;
  onChange: (next: Sample128105Params) => void;
  interactive?: boolean;
}) {
  return (
    <>
      <ambientLight intensity={0.7} />
      <directionalLight
        position={[5, 8, 4]}
        intensity={0.82}
        castShadow
        shadow-mapSize-width={1024}
        shadow-mapSize-height={1024}
        shadow-camera-near={0.4}
        shadow-camera-far={24}
        shadow-camera-left={-6}
        shadow-camera-right={6}
        shadow-camera-top={6}
        shadow-camera-bottom={-6}
        shadow-bias={-0.0004}
      />
      <directionalLight position={[-4, 2.5, -2]} intensity={0.24} />
      <Environment preset="studio" environmentIntensity={0.5} />
      <group>
        <Podium />
        <group scale={1.5}>
          <EditorObject
            params={params}
            onChange={onChange}
            interactive={interactive}
          />
        </group>
      </group>
      <mesh position={[0, -0.549, 0]} rotation={[-Math.PI / 2, 0, 0]} receiveShadow>
        <planeGeometry args={[18, 18]} />
        <shadowMaterial transparent opacity={0.18} />
      </mesh>
      <ContactShadows
        position={[0, -0.548, 0]}
        opacity={0.3}
        scale={4}
        blur={1.4}
        far={0.5}
        resolution={1024}
        color="#0f1626"
        frames={1}
      />
      <OrbitControls
        makeDefault
        enablePan={false}
        enableRotate={false}
        enableZoom
        zoomSpeed={0.6}
        minDistance={5}
        maxDistance={20}
        enableDamping
        dampingFactor={0.08}
        target={[0, 0.5, 0]}
      />
    </>
  );
}

export function Sample128105EditorScene({
  params,
  onChange,
  interactive = true,
}: {
  params: Sample128105Params;
  onChange: (next: Sample128105Params) => void;
  interactive?: boolean;
}) {
  return (
    <Canvas
      shadows="soft"
      dpr={[1, 2]}
      camera={{ position: [0, 2.5, 9], fov: 28 }}
      gl={{ antialias: true, alpha: true }}
      style={{ touchAction: "none" }}
    >
      <Stage params={params} onChange={onChange} interactive={interactive} />
    </Canvas>
  );
}
