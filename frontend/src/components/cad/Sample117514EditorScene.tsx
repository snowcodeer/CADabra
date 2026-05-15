import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import * as THREE from "three";
import { Canvas, ThreeEvent, useThree } from "@react-three/fiber";
import { ContactShadows, Environment, Html, OrbitControls } from "@react-three/drei";
import { Podium } from "./Podium";

const MM_TO_WORLD = 0.01;
const BASE_COLOR = "#8f97a3";
const HIGHLIGHT_COLOR = "#3b9bff";

type EditableSurfaceKey = "tier1Top" | "tier2Top" | "tier3Top";

const TIER1_POLY: Array<[number, number]> = [
  [-3.8970348981067278, 27.36803998100899],
  [16.17630243585472, 21.565145079995656],
  [26.267029050724584, 3.267885795087068],
  [20.464134149711253, -16.805451538874372],
  [2.166874864802663, -26.896178153744245],
  [-17.906462469158775, -21.09328325273092],
  [-27.997189084028644, -2.796023967822332],
  [-22.194294183015327, 17.277313366139115],
];

const TIER2_POLY: Array<[number, number]> = [
  [-2.6045786760708305, 20.811208570677596],
  [-15.94971004485865, 13.451523156314474],
  [-20.182059467657936, -1.1889931942765959],
  [-12.822374053294816, -14.534124563064413],
  [1.8181422972962538, -18.7664739858637],
  [15.163273666084072, -11.406788571500584],
  [19.39562308888336, 3.2337277790904855],
  [12.035937674520243, 16.578859147878305],
];

const TIER3_POLY: Array<[number, number]> = [
  [-1.4272973093033916, 13.478257287281878],
  [-10.278059115091713, 8.597164642483186],
  [-13.08503909789459, -1.112722757793562],
  [-8.203946453095899, -9.963484563581883],
  [1.5059409471808491, -12.770464546384762],
  [10.35670275296917, -7.889371901586071],
  [13.16368273577205, 1.820515498690677],
  [8.28259009097336, 10.671277304478998],
];

export interface Sample117514Params {
  tier1Mm: number;
  tier2Mm: number;
  tier3Mm: number;
}

export const SAMPLE_117514_INITIAL_PARAMS: Sample117514Params = {
  tier1Mm: 27.33214837422117,
  tier2Mm: 28.383384850152748,
  tier3Mm: 27.332148374221187,
};

export function getSample117514TopY(params: Sample117514Params) {
  return (params.tier1Mm + params.tier2Mm + params.tier3Mm) * MM_TO_WORLD + 0.05;
}

function polylineToShape(poly: Array<[number, number]>) {
  const points = poly.map(([x, y]) => new THREE.Vector2(x * MM_TO_WORLD, y * MM_TO_WORLD));
  return new THREE.Shape(points);
}

function polylineToPath(poly: Array<[number, number]>) {
  const path = new THREE.Path();
  path.moveTo(poly[0][0] * MM_TO_WORLD, poly[0][1] * MM_TO_WORLD);
  for (let i = 1; i < poly.length; i++) {
    path.lineTo(poly[i][0] * MM_TO_WORLD, poly[i][1] * MM_TO_WORLD);
  }
  path.closePath();
  return path;
}

function buildTierGeometry(poly: Array<[number, number]>, heightMm: number) {
  const shape = polylineToShape(poly);
  const geom = new THREE.ExtrudeGeometry(shape, {
    depth: heightMm * MM_TO_WORLD,
    bevelEnabled: false,
    curveSegments: 1,
  });
  geom.rotateX(-Math.PI / 2);
  geom.computeVertexNormals();
  return geom;
}

export function buildSample117514Geometries(params: Sample117514Params) {
  return {
    tier1: buildTierGeometry(TIER1_POLY, params.tier1Mm),
    tier2: buildTierGeometry(TIER2_POLY, params.tier2Mm),
    tier3: buildTierGeometry(TIER3_POLY, params.tier3Mm),
  };
}

function buildAnnulusShape(outer: Array<[number, number]>, inner: Array<[number, number]>) {
  const shape = polylineToShape(outer);
  shape.holes.push(polylineToPath(inner));
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
  const geom = useMemo(() => new THREE.ShapeGeometry(shape, 1), [shape]);
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

function SurfaceLabel({ y, text }: { y: number; text: string }) {
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
  params: Sample117514Params;
  activeSurface: EditableSurfaceKey | null;
  liveValueMm: number | null;
}) {
  const { tier1, tier2, tier3 } = useMemo(
    () => buildSample117514Geometries(params),
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
    tier1.dispose();
    tier2.dispose();
    tier3.dispose();
    material.dispose();
  }, [tier1, tier2, tier3, material]);

  const tier1TopY = params.tier1Mm * MM_TO_WORLD;
  const tier2TopY = (params.tier1Mm + params.tier2Mm) * MM_TO_WORLD;
  const tier3TopY = (params.tier1Mm + params.tier2Mm + params.tier3Mm) * MM_TO_WORLD;

  return (
    <group position={[0, 0.05, 0]}>
      <mesh geometry={tier1} material={material} castShadow receiveShadow />
      <mesh geometry={tier2} material={material} castShadow receiveShadow position={[0, tier1TopY, 0]} />
      <mesh geometry={tier3} material={material} castShadow receiveShadow position={[0, tier2TopY, 0]} />
      {activeSurface && liveValueMm !== null && (
        <SurfaceLabel
          y={
            activeSurface === "tier1Top"
              ? tier1TopY + 0.18
              : activeSurface === "tier2Top"
                ? tier2TopY + 0.18
                : tier3TopY + 0.18
          }
          text={`${Math.round(liveValueMm)} mm`}
        />
      )}
    </group>
  );
}

export function Sample117514Solid({
  params = SAMPLE_117514_INITIAL_PARAMS,
  color = BASE_COLOR,
  opacity = 1,
}: {
  params?: Sample117514Params;
  color?: string;
  opacity?: number;
}) {
  const { tier1, tier2, tier3 } = useMemo(
    () => buildSample117514Geometries(params),
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
      tier1.dispose();
      tier2.dispose();
      tier3.dispose();
      material.dispose();
    },
    [tier1, tier2, tier3, material],
  );

  const tier1TopY = params.tier1Mm * MM_TO_WORLD;
  const tier2TopY = (params.tier1Mm + params.tier2Mm) * MM_TO_WORLD;

  return (
    <group position={[0, 0.05, 0]}>
      <mesh geometry={tier1} material={material} castShadow receiveShadow />
      <mesh geometry={tier2} material={material} castShadow receiveShadow position={[0, tier1TopY, 0]} />
      <mesh geometry={tier3} material={material} castShadow receiveShadow position={[0, tier2TopY, 0]} />
    </group>
  );
}

export function Sample117514PointCloud({
  params = SAMPLE_117514_INITIAL_PARAMS,
  count = 14000,
  color = "#6b7280",
}: {
  params?: Sample117514Params;
  count?: number;
  color?: string;
}) {
  const { positions, total } = useMemo(() => {
    const { tier1, tier2, tier3 } = buildSample117514Geometries(params);
    const tier1Attr = tier1.getAttribute("position") as THREE.BufferAttribute;
    const tier2Attr = tier2.getAttribute("position") as THREE.BufferAttribute;
    const tier3Attr = tier3.getAttribute("position") as THREE.BufferAttribute;
    const tier1OffsetY = 0;
    const tier2OffsetY = params.tier1Mm * MM_TO_WORLD;
    const tier3OffsetY = (params.tier1Mm + params.tier2Mm) * MM_TO_WORLD;
    const combined = new Float32Array(count * 3);
    for (let i = 0; i < count; i++) {
      const bucket = i % 3;
      const attr = bucket === 0 ? tier1Attr : bucket === 1 ? tier2Attr : tier3Attr;
      const offsetY = bucket === 0 ? tier1OffsetY : bucket === 1 ? tier2OffsetY : tier3OffsetY;
      const idx = Math.floor(Math.random() * attr.count);
      const x = attr.getX(idx);
      const y = attr.getY(idx) + offsetY;
      const z = attr.getZ(idx);
      combined[i * 3] = x + (Math.random() - 0.5) * 0.004;
      combined[i * 3 + 1] = y + (Math.random() - 0.5) * 0.004;
      combined[i * 3 + 2] = z + (Math.random() - 0.5) * 0.004;
    }
    tier1.dispose();
    tier2.dispose();
    tier3.dispose();
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
  params: Sample117514Params;
  onChange: (next: Sample117514Params) => void;
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
    startParams: Sample117514Params;
  } | null>(null);

  useEffect(() => {
    if (!dragging) setLiveParams(params);
  }, [params, dragging]);

  const shapeTier1Top = useMemo(() => buildAnnulusShape(TIER1_POLY, TIER2_POLY), []);
  const shapeTier2Top = useMemo(() => buildAnnulusShape(TIER2_POLY, TIER3_POLY), []);
  const shapeTier3Top = useMemo(() => polylineToShape(TIER3_POLY), []);

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
      if (state.surface === "tier1Top") {
        next.tier1Mm = THREE.MathUtils.clamp(state.startParams.tier1Mm + deltaMm, 5, 80);
      } else if (state.surface === "tier2Top") {
        next.tier2Mm = THREE.MathUtils.clamp(state.startParams.tier2Mm + deltaMm, 5, 80);
      } else {
        next.tier3Mm = THREE.MathUtils.clamp(state.startParams.tier3Mm + deltaMm, 5, 80);
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
  const tier1TopY = shown.tier1Mm * MM_TO_WORLD + 0.05;
  const tier2TopY = (shown.tier1Mm + shown.tier2Mm) * MM_TO_WORLD + 0.05;
  const tier3TopY = (shown.tier1Mm + shown.tier2Mm + shown.tier3Mm) * MM_TO_WORLD + 0.05;
  const liveValueMm =
    dragging === "tier1Top"
      ? shown.tier1Mm
      : dragging === "tier2Top"
        ? shown.tier2Mm
        : dragging === "tier3Top"
          ? shown.tier3Mm
          : null;

  return (
    <group ref={groupRef}>
      <ParametricPart params={shown} activeSurface={dragging} liveValueMm={liveValueMm} />
      <SurfaceRing
        y={tier1TopY}
        shape={shapeTier1Top}
        hovered={activeHover === "tier1Top"}
        active={dragging === "tier1Top"}
        onPointerDown={beginDrag("tier1Top", tier1TopY)}
        onPointerOver={(e) => {
          e.stopPropagation();
          if (interactive) setHovered("tier1Top");
          if (interactive && !dragging) gl.domElement.style.cursor = "ns-resize";
        }}
        onPointerOut={() => {
          setHovered((prev) => (prev === "tier1Top" ? null : prev));
          if (!dragging) gl.domElement.style.cursor = "default";
        }}
      />
      <SurfaceRing
        y={tier2TopY}
        shape={shapeTier2Top}
        hovered={activeHover === "tier2Top"}
        active={dragging === "tier2Top"}
        onPointerDown={beginDrag("tier2Top", tier2TopY)}
        onPointerOver={(e) => {
          e.stopPropagation();
          if (interactive) setHovered("tier2Top");
          if (interactive && !dragging) gl.domElement.style.cursor = "ns-resize";
        }}
        onPointerOut={() => {
          setHovered((prev) => (prev === "tier2Top" ? null : prev));
          if (!dragging) gl.domElement.style.cursor = "default";
        }}
      />
      <SurfaceRing
        y={tier3TopY}
        shape={shapeTier3Top}
        hovered={activeHover === "tier3Top"}
        active={dragging === "tier3Top"}
        onPointerDown={beginDrag("tier3Top", tier3TopY)}
        onPointerOver={(e) => {
          e.stopPropagation();
          if (interactive) setHovered("tier3Top");
          if (interactive && !dragging) gl.domElement.style.cursor = "ns-resize";
        }}
        onPointerOut={() => {
          setHovered((prev) => (prev === "tier3Top" ? null : prev));
          if (!dragging) gl.domElement.style.cursor = "default";
        }}
      />
    </group>
  );
}

export function Sample117514EditableSolid({
  params,
  onChange,
  onDraggingChange,
  interactive = true,
}: {
  params: Sample117514Params;
  onChange: (next: Sample117514Params) => void;
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
  params: Sample117514Params;
  onChange: (next: Sample117514Params) => void;
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
        minDistance={4}
        maxDistance={14}
        enableDamping
        dampingFactor={0.08}
        target={[0, 0.5, 0]}
      />
    </>
  );
}

export function Sample117514EditorScene({
  params,
  onChange,
  interactive = true,
}: {
  params: Sample117514Params;
  onChange: (next: Sample117514Params) => void;
  interactive?: boolean;
}) {
  return (
    <Canvas
      shadows="soft"
      dpr={[1, 2]}
      camera={{ position: [0, 2.5, 7.5], fov: 28 }}
      gl={{ antialias: true, alpha: true }}
      style={{ touchAction: "none" }}
    >
      <Stage params={params} onChange={onChange} interactive={interactive} />
    </Canvas>
  );
}
