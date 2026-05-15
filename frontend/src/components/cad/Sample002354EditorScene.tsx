import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import * as THREE from "three";
import { Canvas, ThreeEvent, useThree } from "@react-three/fiber";
import { ContactShadows, Environment, Html, OrbitControls } from "@react-three/drei";
import { Podium } from "./Podium";

const MM_TO_WORLD = 0.01;
const BASE_COLOR = "#8f97a3";
const HIGHLIGHT_COLOR = "#3b9bff";
const HOLE_RADIUS_MM = 8.353316376867248;

type EditableSurfaceKey = "topFace";

export interface Sample002354Params {
  extrudeMm: number;
}

export const SAMPLE_002354_INITIAL_PARAMS: Sample002354Params = {
  extrudeMm: 45.48475955233508,
};

const POLYLINE_MM: ReadonlyArray<readonly [number, number]> = [
  [-41.23951532745047, 24.258538427912043],
  [-42.04813327504754, 22.236993558919373],
  [-42.04813327504754, 19.811139716128167],
  [-42.04813327504754, 17.385285873336965],
  [-42.04813327504754, 14.95943203054576],
  [-42.04813327504754, 12.533578187754555],
  [-42.04813327504754, 10.107724344963351],
  [-42.04813327504754, 7.6818705021721465],
  [-42.04813327504754, 5.2560166593809425],
  [-42.04813327504754, 2.830162816589738],
  [-42.04813327504754, 0.40430897379853403],
  [-42.04813327504754, -2.0215448689926703],
  [-42.04813327504754, -4.447398711783874],
  [-42.04813327504754, -6.873252554575078],
  [-42.04813327504754, -9.703415371164816],
  [-42.04813327504754, -12.129269213956022],
  [-42.04813327504754, -14.555123056747226],
  [-42.04813327504754, -16.98097689953843],
  [-42.04813327504754, -19.406830742329632],
  [-42.04813327504754, -21.832684585120838],
  [-41.23951532745047, -23.854229454113508],
  [-38.813661484659264, -23.854229454113508],
  [-36.387807641868065, -23.854229454113508],
  [-33.96195379907686, -23.854229454113508],
  [-31.536099956285653, -23.854229454113508],
  [-29.11024611349445, -23.854229454113508],
  [-26.684392270703245, -23.854229454113508],
  [-24.258538427912043, -23.854229454113508],
  [-21.832684585120838, -23.854229454113508],
  [-19.406830742329632, -23.854229454113508],
  [-16.98097689953843, -23.854229454113508],
  [-14.555123056747226, -23.854229454113508],
  [-12.129269213956022, -23.854229454113508],
  [-9.703415371164816, -23.854229454113508],
  [-7.277561528373613, -23.854229454113508],
  [-4.851707685582408, -23.854229454113508],
  [-2.425853842791204, -23.854229454113508],
  [0.0, -23.854229454113508],
  [2.425853842791204, -23.854229454113508],
  [4.851707685582408, -23.854229454113508],
  [7.277561528373613, -23.854229454113508],
  [9.703415371164816, -23.854229454113508],
  [12.129269213956022, -23.854229454113508],
  [14.555123056747226, -23.854229454113508],
  [16.98097689953843, -23.854229454113508],
  [19.406830742329632, -23.854229454113508],
  [21.832684585120838, -23.854229454113508],
  [24.258538427912043, -23.854229454113508],
  [26.684392270703245, -23.854229454113508],
  [29.514555087292983, -23.854229454113508],
  [31.94040893008419, -23.854229454113508],
  [34.366262772875395, -23.854229454113508],
  [34.77057174667393, -21.832684585120838],
  [34.77057174667393, -19.406830742329632],
  [34.77057174667393, -16.98097689953843],
  [34.77057174667393, -14.555123056747226],
  [34.77057174667393, -12.129269213956022],
  [34.77057174667393, -9.703415371164816],
  [34.77057174667393, -7.277561528373613],
  [34.77057174667393, -4.851707685582408],
  [34.77057174667393, -2.425853842791204],
  [34.77057174667393, -0.0],
  [34.77057174667393, 2.425853842791204],
  [34.77057174667393, 4.851707685582408],
  [34.77057174667393, 7.277561528373613],
  [36.7921166156666, 8.894797423567748],
  [39.2179704584578, 8.894797423567748],
  [41.643824301249005, 9.299106397366282],
  [41.643824301249005, 11.320651266358952],
  [41.643824301249005, 14.150814082948692],
  [41.643824301249005, 16.576667925739894],
  [41.643824301249005, 19.0025217685311],
  [41.643824301249005, 21.428375611322302],
  [41.643824301249005, 23.854229454113508],
  [39.622279432256335, 24.258538427912043],
  [37.19642558946513, 24.258538427912043],
  [34.77057174667393, 24.258538427912043],
  [32.344717903882724, 24.258538427912043],
  [29.91886406109152, 24.258538427912043],
  [27.493010218300313, 24.258538427912043],
  [25.06715637550911, 24.258538427912043],
  [22.641302532717905, 24.258538427912043],
  [20.215448689926703, 24.258538427912043],
  [17.789594847135497, 24.258538427912043],
  [15.363741004344293, 24.258538427912043],
  [12.937887161553089, 24.258538427912043],
  [10.512033318761885, 24.258538427912043],
  [7.6818705021721465, 24.258538427912043],
  [5.2560166593809425, 24.258538427912043],
  [2.830162816589738, 24.258538427912043],
  [0.40430897379853403, 24.258538427912043],
  [-2.0215448689926703, 24.258538427912043],
  [-4.447398711783874, 24.258538427912043],
  [-6.873252554575078, 24.258538427912043],
  [-9.299106397366282, 24.258538427912043],
  [-11.724960240157486, 24.258538427912043],
  [-14.150814082948692, 24.258538427912043],
  [-16.576667925739894, 24.258538427912043],
  [-19.0025217685311, 24.258538427912043],
  [-21.428375611322302, 24.258538427912043],
  [-23.854229454113508, 24.258538427912043],
  [-26.280083296904714, 24.258538427912043],
  [-28.705937139695916, 24.258538427912043],
  [-31.13179098248712, 24.258538427912043],
  [-33.557644825278324, 24.258538427912043],
  [-35.98349866806953, 24.258538427912043],
  [-38.409352510860735, 24.258538427912043],
];

const POLYGON_BBOX_CENTER_MM: readonly [number, number] = (() => {
  let minX = Infinity;
  let maxX = -Infinity;
  let minY = Infinity;
  let maxY = -Infinity;
  for (const [x, y] of POLYLINE_MM) {
    if (x < minX) minX = x;
    if (x > maxX) maxX = x;
    if (y < minY) minY = y;
    if (y > maxY) maxY = y;
  }
  return [(minX + maxX) / 2, (minY + maxY) / 2];
})();

export function getSample002354TopY(params: Sample002354Params) {
  return params.extrudeMm * MM_TO_WORLD + 0.05;
}

function circlePath(radiusMm: number, centerXMm = 0, centerYMm = 0): THREE.Path {
  const path = new THREE.Path();
  path.absellipse(
    centerXMm * MM_TO_WORLD,
    centerYMm * MM_TO_WORLD,
    radiusMm * MM_TO_WORLD,
    radiusMm * MM_TO_WORLD,
    0,
    Math.PI * 2,
    false,
    0,
  );
  return path;
}

function buildOuterShape() {
  const shape = new THREE.Shape();
  POLYLINE_MM.forEach(([x, y], idx) => {
    const wx = x * MM_TO_WORLD;
    const wy = y * MM_TO_WORLD;
    if (idx === 0) shape.moveTo(wx, wy);
    else shape.lineTo(wx, wy);
  });
  shape.closePath();
  return shape;
}

function buildTopFaceShape() {
  const shape = buildOuterShape();
  shape.holes.push(circlePath(HOLE_RADIUS_MM, POLYGON_BBOX_CENTER_MM[0], POLYGON_BBOX_CENTER_MM[1]));
  return shape;
}

export function buildSample002354Geometry(params: Sample002354Params): THREE.BufferGeometry {
  const shape = buildTopFaceShape();
  const geom = new THREE.ExtrudeGeometry(shape, {
    depth: params.extrudeMm * MM_TO_WORLD,
    bevelEnabled: false,
    curveSegments: 48,
  });
  geom.rotateX(-Math.PI / 2);
  geom.computeVertexNormals();
  return geom;
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
  params: Sample002354Params;
  activeSurface: EditableSurfaceKey | null;
  liveValueMm: number | null;
}) {
  const geom = useMemo(() => buildSample002354Geometry(params), [params]);
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

export function Sample002354Solid({
  params = SAMPLE_002354_INITIAL_PARAMS,
  color = BASE_COLOR,
  opacity = 1,
}: {
  params?: Sample002354Params;
  color?: string;
  opacity?: number;
}) {
  const geom = useMemo(() => buildSample002354Geometry(params), [params]);
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
  params: Sample002354Params;
  onChange: (next: Sample002354Params) => void;
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
    startParams: Sample002354Params;
  } | null>(null);

  useEffect(() => {
    if (!dragging) setLiveParams(params);
  }, [params, dragging]);

  const shapeTopFace = useMemo(() => buildTopFaceShape(), []);

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
      next.extrudeMm = THREE.MathUtils.clamp(state.startParams.extrudeMm + deltaMm, 10, 100);
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
  const liveValueMm = dragging === "topFace" ? shown.extrudeMm : null;

  return (
    <group ref={groupRef}>
      <ParametricPart params={shown} activeSurface={dragging} liveValueMm={liveValueMm} />
      <SurfaceRing
        y={topY}
        shape={shapeTopFace}
        hovered={activeHover === "topFace"}
        active={dragging === "topFace"}
        onPointerDown={beginDrag("topFace", topY)}
        onPointerOver={(e) => {
          e.stopPropagation();
          if (interactive) setHovered("topFace");
          if (interactive && !dragging) gl.domElement.style.cursor = "ns-resize";
        }}
        onPointerOut={() => {
          setHovered((prev) => (prev === "topFace" ? null : prev));
          if (!dragging) gl.domElement.style.cursor = "default";
        }}
      />
    </group>
  );
}

export function Sample002354EditableSolid({
  params,
  onChange,
  onDraggingChange,
  interactive = true,
}: {
  params: Sample002354Params;
  onChange: (next: Sample002354Params) => void;
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
  params: Sample002354Params;
  onChange: (next: Sample002354Params) => void;
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
        minDistance={4.5}
        maxDistance={16}
        enableDamping
        dampingFactor={0.08}
        target={[0, 0.5, 0]}
      />
    </>
  );
}

export function Sample002354EditorScene({
  params,
  onChange,
  interactive = true,
}: {
  params: Sample002354Params;
  onChange: (next: Sample002354Params) => void;
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
