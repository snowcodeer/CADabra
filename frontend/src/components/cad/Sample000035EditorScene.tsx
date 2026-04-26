import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import * as THREE from "three";
import { Canvas, ThreeEvent, useFrame, useThree } from "@react-three/fiber";
import { ContactShadows, Environment, Html, OrbitControls } from "@react-three/drei";
import { Podium } from "./Podium";

const MM_TO_WORLD = 0.01;
const BASE_COLOR = "#8f97a3";
const HIGHLIGHT_COLOR = "#3b9bff";
const OUTER_RADIUS_MM = 34.84795430908487;
const COUNTERBORE_RADIUS_MM = 48.402948402948404 / 2;
const THROUGH_RADIUS_MM = 38.19088158751952 / 2;

type EditableSurfaceKey = "baseTop" | "bossTop" | "counterboreFloor";

export interface Sample000035Params {
  baseHeightMm: number;
  bossHeightMm: number;
  counterboreDepthMm: number;
}

export const SAMPLE_000035_INITIAL_PARAMS: Sample000035Params = {
  baseHeightMm: 20.10387833172643,
  bossHeightMm: 25.758094112524493,
  counterboreDepthMm: 3.769477187198706,
};

export function getSample000035TopY(params: Sample000035Params) {
  return (params.baseHeightMm + params.bossHeightMm) * MM_TO_WORLD + 0.05;
}

type Segment =
  | {
      kind: "line";
      start: [number, number];
      end: [number, number];
    }
  | {
      kind: "arc";
      start: [number, number];
      end: [number, number];
      arcCenter: [number, number];
    };

const OUTER_PROFILE: Segment[] = [
  { kind: "line", start: [-7.4498095973601375, 54.286359473922666], end: [-29.672734134553515, 46.74793901610148] },
  { kind: "line", start: [-29.64495051505229, 46.35300171054931], end: [-44.160450622363925, 31.876588018850004] },
  { kind: "line", start: [-43.749093797580805, 31.449749784430868], end: [-44.227973081061464, -27.027011231201595] },
  {
    kind: "arc",
    start: [-43.93553109705668, -27.01250146242402],
    end: [-39.5497904284069, -34.48410212768042],
    arcCenter: [-34.581532049716344, -26.54480446734409],
  },
  { kind: "line", start: [-40.08453211020782, -35.23514855137973], end: [-26.26908069299105, -46.46819351136355] },
  { kind: "line", start: [-26.044524118709354, -46.192358143181316], end: [-13.248065011829516, -51.54988284578001] },
  { kind: "line", start: [-13.28383338059259, -51.84229646017237], end: [5.927209507708278, -53.0972180506291] },
  { kind: "line", start: [5.9873458982136585, -53.82082606662321], end: [26.07667798582525, -46.77558256653548] },
  { kind: "line", start: [26.533141593580062, -47.15873784681128], end: [44.223359614821064, -28.97672539654907] },
  { kind: "line", start: [42.268797542242055, -28.48971123349839], end: [42.728224841368736, 29.98333748707927] },
  { kind: "line", start: [43.80892942415965, 30.040345939283213], end: [34.166111914488155, 41.07497140061989] },
  { kind: "line", start: [34.28317972720693, 41.090208932431814], end: [19.04892382533255, 50.42544637480556] },
  {
    kind: "arc",
    start: [18.505887696005377, 49.55210163590824],
    end: [-7.4498095973601375, 54.286359473922666],
    arcCenter: [3.4598243076391855, 25.291592433792676],
  },
];

function circlePath(radiusMm: number): THREE.Path {
  const path = new THREE.Path();
  path.absellipse(0, 0, radiusMm * MM_TO_WORLD, radiusMm * MM_TO_WORLD, 0, Math.PI * 2, false, 0);
  return path;
}

function sampleArc(
  start: [number, number],
  end: [number, number],
  center: [number, number],
) {
  const startAngle = Math.atan2(start[1] - center[1], start[0] - center[0]);
  const endAngle = Math.atan2(end[1] - center[1], end[0] - center[0]);
  let delta = endAngle - startAngle;
  if (delta > Math.PI) delta -= Math.PI * 2;
  if (delta < -Math.PI) delta += Math.PI * 2;
  const steps = Math.max(10, Math.ceil(Math.abs(delta) / 0.12));
  const radius = Math.hypot(start[0] - center[0], start[1] - center[1]);
  const pts: THREE.Vector2[] = [];
  for (let i = 1; i <= steps; i++) {
    const a = startAngle + (delta * i) / steps;
    pts.push(new THREE.Vector2((center[0] + Math.cos(a) * radius) * MM_TO_WORLD, (center[1] + Math.sin(a) * radius) * MM_TO_WORLD));
  }
  return pts;
}

function buildOuterShape() {
  const points: THREE.Vector2[] = [];
  OUTER_PROFILE.forEach((segment, idx) => {
    if (idx === 0) {
      points.push(new THREE.Vector2(segment.start[0] * MM_TO_WORLD, segment.start[1] * MM_TO_WORLD));
    }
    if (segment.kind === "line") {
      points.push(new THREE.Vector2(segment.end[0] * MM_TO_WORLD, segment.end[1] * MM_TO_WORLD));
    } else {
      points.push(...sampleArc(segment.start, segment.end, segment.arcCenter));
    }
  });
  return new THREE.Shape(points);
}

function buildBaseSolidGeometry(params: Sample000035Params) {
  const shape = buildOuterShape();
  shape.holes.push(circlePath(THROUGH_RADIUS_MM));
  const geom = new THREE.ExtrudeGeometry(shape, {
    depth: params.baseHeightMm * MM_TO_WORLD,
    bevelEnabled: false,
    curveSegments: 48,
  });
  geom.rotateX(-Math.PI / 2);
  geom.computeVertexNormals();
  return geom;
}

function buildBossGeometry(params: Sample000035Params) {
  const outerR = OUTER_RADIUS_MM * MM_TO_WORLD;
  const cbR = COUNTERBORE_RADIUS_MM * MM_TO_WORLD;
  const innerR = THROUGH_RADIUS_MM * MM_TO_WORLD;
  const bossH = params.bossHeightMm * MM_TO_WORLD;
  const cbDepth = params.counterboreDepthMm * MM_TO_WORLD;
  const points = [
    new THREE.Vector2(innerR, 0),
    new THREE.Vector2(outerR, 0),
    new THREE.Vector2(outerR, bossH),
    new THREE.Vector2(cbR, bossH),
    new THREE.Vector2(cbR, bossH - cbDepth),
    new THREE.Vector2(innerR, bossH - cbDepth),
  ];
  const geom = new THREE.LatheGeometry(points, 96);
  geom.computeVertexNormals();
  return geom;
}

function buildSample000035Geometries(params: Sample000035Params) {
  return {
    base: buildBaseSolidGeometry(params),
    boss: buildBossGeometry(params),
  };
}

function buildRingShape(innerRadiusMm: number, outerRadiusMm: number) {
  const shape = new THREE.Shape();
  shape.absellipse(0, 0, outerRadiusMm * MM_TO_WORLD, outerRadiusMm * MM_TO_WORLD, 0, Math.PI * 2, false, 0);
  shape.holes.push(circlePath(innerRadiusMm));
  return shape;
}

function SurfaceRing({
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
  params: Sample000035Params;
  activeSurface: EditableSurfaceKey | null;
  liveValueMm: number | null;
}) {
  const { base: baseGeom, boss: bossGeom } = useMemo(
    () => buildSample000035Geometries(params),
    [params],
  );
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
    baseGeom.dispose();
    bossGeom.dispose();
    material.dispose();
  }, [baseGeom, bossGeom, material]);

  const baseTopY = params.baseHeightMm * MM_TO_WORLD;
  const bossTopY = (params.baseHeightMm + params.bossHeightMm) * MM_TO_WORLD;
  const counterboreFloorY = (params.baseHeightMm + params.bossHeightMm - params.counterboreDepthMm) * MM_TO_WORLD;

  return (
    <group position={[0, 0.05, 0]}>
      <mesh geometry={baseGeom} material={material} castShadow receiveShadow />
      <mesh geometry={bossGeom} material={material} castShadow receiveShadow position={[0, baseTopY, 0]} />
      {activeSurface && liveValueMm !== null && (
        <SurfaceLabel
          y={
            activeSurface === "baseTop"
              ? baseTopY + 0.18
              : activeSurface === "bossTop"
                ? bossTopY + 0.18
                : counterboreFloorY + 0.18
          }
          text={`${Math.round(liveValueMm)} mm`}
        />
      )}
    </group>
  );
}

export function Sample000035Solid({
  params = SAMPLE_000035_INITIAL_PARAMS,
  color = BASE_COLOR,
  opacity = 1,
}: {
  params?: Sample000035Params;
  color?: string;
  opacity?: number;
}) {
  const { base: baseGeom, boss: bossGeom } = useMemo(
    () => buildSample000035Geometries(params),
    [params],
  );
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
      baseGeom.dispose();
      bossGeom.dispose();
      material.dispose();
    },
    [baseGeom, bossGeom, material],
  );

  return (
    <group position={[0, 0.05, 0]}>
      <mesh geometry={baseGeom} material={material} castShadow receiveShadow />
      <mesh
        geometry={bossGeom}
        material={material}
        castShadow
        receiveShadow
        position={[0, params.baseHeightMm * MM_TO_WORLD, 0]}
      />
    </group>
  );
}

export function Sample000035PointCloud({
  params = SAMPLE_000035_INITIAL_PARAMS,
  count = 14000,
  color = "#6b7280",
}: {
  params?: Sample000035Params;
  count?: number;
  color?: string;
}) {
  const { positions, total } = useMemo(() => {
    const { base, boss } = buildSample000035Geometries(params);
    const baseAttr = base.getAttribute("position") as THREE.BufferAttribute;
    const bossAttr = boss.getAttribute("position") as THREE.BufferAttribute;
    const combined = new Float32Array(count * 3);
    for (let i = 0; i < count; i++) {
      const useBoss = i % 4 === 0;
      const attr = useBoss ? bossAttr : baseAttr;
      const idx = Math.floor(Math.random() * attr.count);
      const x = attr.getX(idx);
      const y = attr.getY(idx) + (useBoss ? params.baseHeightMm * MM_TO_WORLD : 0);
      const z = attr.getZ(idx);
      combined[i * 3] = x + (Math.random() - 0.5) * 0.004;
      combined[i * 3 + 1] = y + (Math.random() - 0.5) * 0.004;
      combined[i * 3 + 2] = z + (Math.random() - 0.5) * 0.004;
    }
    base.dispose();
    boss.dispose();
    return { positions: combined, total: count };
  }, [count, params]);

  return (
    <group position={[0, 0.05, 0]}>
      <points>
        <bufferGeometry>
          <bufferAttribute attach="attributes-position" array={positions} count={total} itemSize={3} />
        </bufferGeometry>
        <pointsMaterial size={0.024} color={color} sizeAttenuation transparent opacity={0.92} />
      </points>
    </group>
  );
}

function EditorObject({
  params,
  onChange,
  onDraggingChange,
  interactive = true,
}: {
  params: Sample000035Params;
  onChange: (next: Sample000035Params) => void;
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
    startParams: Sample000035Params;
  } | null>(null);

  useEffect(() => {
    if (!dragging) setLiveParams(params);
  }, [params, dragging]);

  const shapeBaseTop = useMemo(() => {
    const shape = buildOuterShape();
    shape.holes.push(circlePath(OUTER_RADIUS_MM));
    return shape;
  }, []);
  const shapeBossTop = useMemo(() => buildRingShape(COUNTERBORE_RADIUS_MM, OUTER_RADIUS_MM), []);
  const shapeCounterboreFloor = useMemo(() => buildRingShape(THROUGH_RADIUS_MM, COUNTERBORE_RADIUS_MM), []);

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
      if (state.surface === "baseTop") {
        next.baseHeightMm = THREE.MathUtils.clamp(state.startParams.baseHeightMm + deltaMm, 8, 40);
      } else if (state.surface === "bossTop") {
        next.bossHeightMm = THREE.MathUtils.clamp(
          state.startParams.bossHeightMm + deltaMm,
          Math.max(state.startParams.counterboreDepthMm + 2, 6),
          45,
        );
      } else {
        next.counterboreDepthMm = THREE.MathUtils.clamp(
          state.startParams.counterboreDepthMm - deltaMm,
          0,
          Math.max(0, state.startParams.bossHeightMm - 2),
        );
      }
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
  const baseTopY = shown.baseHeightMm * MM_TO_WORLD + 0.05;
  const bossTopY = (shown.baseHeightMm + shown.bossHeightMm) * MM_TO_WORLD + 0.05;
  const counterboreFloorY = (shown.baseHeightMm + shown.bossHeightMm - shown.counterboreDepthMm) * MM_TO_WORLD + 0.05;
  const liveValueMm =
    dragging === "baseTop"
      ? shown.baseHeightMm
      : dragging === "bossTop"
        ? shown.bossHeightMm
        : dragging === "counterboreFloor"
          ? shown.counterboreDepthMm
          : null;

  return (
    <group ref={groupRef}>
      <ParametricPart params={shown} activeSurface={dragging} liveValueMm={liveValueMm} />
      <SurfaceRing
        y={baseTopY}
        shape={shapeBaseTop}
        hovered={activeHover === "baseTop"}
        active={dragging === "baseTop"}
        onPointerDown={beginDrag("baseTop", baseTopY)}
        onPointerOver={(e) => {
          e.stopPropagation();
          if (interactive) setHovered("baseTop");
          if (interactive && !dragging) gl.domElement.style.cursor = "ns-resize";
        }}
        onPointerOut={() => {
          setHovered((prev) => (prev === "baseTop" ? null : prev));
          if (!dragging) gl.domElement.style.cursor = "default";
        }}
      />
      <SurfaceRing
        y={bossTopY}
        shape={shapeBossTop}
        hovered={activeHover === "bossTop"}
        active={dragging === "bossTop"}
        onPointerDown={beginDrag("bossTop", bossTopY)}
        onPointerOver={(e) => {
          e.stopPropagation();
          if (interactive) setHovered("bossTop");
          if (interactive && !dragging) gl.domElement.style.cursor = "ns-resize";
        }}
        onPointerOut={() => {
          setHovered((prev) => (prev === "bossTop" ? null : prev));
          if (!dragging) gl.domElement.style.cursor = "default";
        }}
      />
      <SurfaceRing
        y={counterboreFloorY}
        shape={shapeCounterboreFloor}
        hovered={activeHover === "counterboreFloor"}
        active={dragging === "counterboreFloor"}
        onPointerDown={beginDrag("counterboreFloor", counterboreFloorY)}
        onPointerOver={(e) => {
          e.stopPropagation();
          if (interactive) setHovered("counterboreFloor");
          if (interactive && !dragging) gl.domElement.style.cursor = "ns-resize";
        }}
        onPointerOut={() => {
          setHovered((prev) => (prev === "counterboreFloor" ? null : prev));
          if (!dragging) gl.domElement.style.cursor = "default";
        }}
      />
    </group>
  );
}

function Stage({
  params,
  onChange,
  interactive = true,
}: {
  params: Sample000035Params;
  onChange: (next: Sample000035Params) => void;
  interactive?: boolean;
}) {
  const spinRef = useRef<THREE.Group>(null);
  const [dragging, setDragging] = useState(false);
  useFrame((_, delta) => {
    if (!spinRef.current || dragging) return;
    spinRef.current.rotation.y += delta * 0.18;
  });
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
      <group ref={spinRef}>
        <Podium />
        <EditorObject
          params={params}
          onChange={onChange}
          onDraggingChange={setDragging}
          interactive={interactive}
        />
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
        enableZoom
        zoomSpeed={0.6}
        minDistance={4.5}
        maxDistance={14}
        enableDamping
        dampingFactor={0.08}
        minPolarAngle={0.15}
        maxPolarAngle={Math.PI / 2 - 0.05}
        target={[0, 0.35, 0]}
      />
    </>
  );
}

export function Sample000035EditorScene({
  params,
  onChange,
  interactive = true,
}: {
  params: Sample000035Params;
  onChange: (next: Sample000035Params) => void;
  interactive?: boolean;
}) {
  return (
    <Canvas
      shadows="soft"
      dpr={[1, 2]}
      camera={{ position: [0, 1.8, 6.8], fov: 28 }}
      gl={{ antialias: true, alpha: true }}
      style={{ touchAction: "none" }}
    >
      <Stage params={params} onChange={onChange} interactive={interactive} />
    </Canvas>
  );
}
