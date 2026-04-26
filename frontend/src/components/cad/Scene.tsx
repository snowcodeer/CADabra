import { Suspense, useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import * as THREE from "three";
import { Canvas, ThreeEvent, useFrame, useThree } from "@react-three/fiber";
import { Environment, ContactShadows, OrbitControls, Html } from "@react-three/drei";
import { STLLoader } from "three/addons/loaders/STLLoader.js";
import { Podium } from "./Podium";
import { API_BASE } from "@/lib/api";
import {
  SAMPLE_000035_INITIAL_PARAMS,
  Sample000035Solid,
  type Sample000035Params,
  getSample000035TopY,
} from "./Sample000035EditorScene";

type FaceKey = "px" | "nx" | "py" | "ny" | "pz" | "nz";
type ActiveDrag = { face: FaceKey; delta: number; pin: THREE.Vector3 } | null;

interface SpinningStageProps {
  /** Pause auto-rotation (e.g. while user drags this object) */
  paused: boolean;
  /** Shared rotation ref so every stage spins in unison */
  rotationRef: React.MutableRefObject<number>;
  children: React.ReactNode;
}

/**
 * Group whose Y rotation is driven by a shared external value, so all
 * stages on screen turn at exactly the same speed and angle.
 */
function SpinningStage({ paused, rotationRef, children }: SpinningStageProps) {
  const ref = useRef<THREE.Group>(null);
  useFrame((_, delta) => {
    if (!ref.current) return;
    if (!paused) rotationRef.current += delta * 0.22;
    ref.current.rotation.y = rotationRef.current;
  });
  return <group ref={ref}>{children}</group>;
}

/** Smooth easing — symmetric ease-in-out for natural acceleration & settle. */
const easeInOutCubic = (t: number) =>
  t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;

/** Duration (seconds) for the Compare-mode glide. */
const TRANSITION_DURATION = 2.35;

/**
 * Animates a group's position, scale and material opacity from its current
 * values toward target values using a duration-based eased tween. When any
 * target changes, the tween restarts from the current pose so the motion
 * always feels deliberate (no asymptotic drift).
 */
interface AnimatedStageProps {
  targetPosition: [number, number, number];
  targetScale: number;
  targetOpacity: number;
  children: React.ReactNode;
}
function AnimatedStage({
  targetPosition,
  targetScale,
  targetOpacity,
  children,
}: AnimatedStageProps) {
  const ref = useRef<THREE.Group>(null);
  const tweenRef = useRef<{
    fromPos: [number, number, number];
    fromScale: number;
    fromOpacity: number;
    toPos: [number, number, number];
    toScale: number;
    toOpacity: number;
    elapsed: number;
  } | null>(null);
  const opacityRef = useRef(targetOpacity);
  const initializedRef = useRef(false);

  // Restart the tween whenever a target changes. We snap to the first
  // target on mount so meshes appear at their resting pose immediately.
  useLayoutEffect(() => {
    if (!ref.current) return;
    if (!initializedRef.current) {
      ref.current.position.set(...targetPosition);
      ref.current.scale.setScalar(targetScale);
      ref.current.visible = true;
      opacityRef.current = targetOpacity;
      initializedRef.current = true;
      return;
    }
    tweenRef.current = {
      fromPos: [
        ref.current.position.x,
        ref.current.position.y,
        ref.current.position.z,
      ],
      fromScale: ref.current.scale.x,
      fromOpacity: opacityRef.current,
      toPos: targetPosition,
      toScale: targetScale,
      toOpacity: targetOpacity,
      elapsed: 0,
    };
    ref.current.visible = true;
  }, [targetPosition[0], targetPosition[1], targetPosition[2], targetScale, targetOpacity]);

  useFrame((_, delta) => {
    if (!ref.current) return;
    const tween = tweenRef.current;
    if (!tween) return;

    tween.elapsed += delta;
    const t = Math.min(1, tween.elapsed / TRANSITION_DURATION);
    const e = easeInOutCubic(t);
    ref.current.position.x = tween.fromPos[0] + (tween.toPos[0] - tween.fromPos[0]) * e;
    ref.current.position.y = tween.fromPos[1] + (tween.toPos[1] - tween.fromPos[1]) * e;
    ref.current.position.z = tween.fromPos[2] + (tween.toPos[2] - tween.fromPos[2]) * e;
    const s = tween.fromScale + (tween.toScale - tween.fromScale) * e;
    ref.current.scale.set(s, s, s);
    opacityRef.current = tween.fromOpacity + (tween.toOpacity - tween.fromOpacity) * e;

    // Keep the group renderable at all times. The point cloud exits compare
    // mode by moving/scaling out of frame instead of toggling visibility,
    // because drei Html labels can remain visible even when WebGL children
    // are hidden, making the podiums/objects appear to vanish.
    ref.current.visible = true;

    if (t >= 1) tweenRef.current = null;
  });
  return <group ref={ref}>{children}</group>;
}

/**
 * Drives the camera with a duration-based eased tween while `active` is
 * true. The tween captures the camera's CURRENT pose (from the orbit ref
 * that's continuously synced from OrbitControls during free orbit) and
 * arcs along a spherical path around the destination look target until it
 * settles at the requested view. While active it also keeps the camera
 * looking at the eased focal point via `camera.lookAt()`. OrbitControls is
 * unmounted by the parent during this period so nothing fights the camera.
 */
function CameraTransitionController({
  compareMode,
  cameraTarget,
  lookTarget,
  currentLookRef,
}: {
  compareMode: boolean;
  cameraTarget: [number, number, number];
  lookTarget: [number, number, number];
  /** Live ref to the user's current look target. */
  currentLookRef: React.MutableRefObject<THREE.Vector3>;
}) {
  const camera = useThree((state) => state.camera);
  const controls = useThree((state) => state.controls) as unknown as
    | { target: THREE.Vector3; enabled: boolean; update: () => void }
    | undefined;
  const tweenRef = useRef<{
    fromLook: THREE.Vector3;
    toLook: THREE.Vector3;
    fromPosition: THREE.Vector3;
    toPosition: THREE.Vector3;
    elapsed: number;
  } | null>(null);
  const lastFreePoseRef = useRef({
    initialized: false,
    position: new THREE.Vector3(0, 4.2, 10.5),
    look: new THREE.Vector3(0, 0.1, 0),
  });
  const initialModeRef = useRef(compareMode);
  const didInitializeCameraRef = useRef(false);

  useLayoutEffect(() => {
    if (didInitializeCameraRef.current || !controls) return;
    controls.target.copy(currentLookRef.current);
    controls.update();
    camera.lookAt(currentLookRef.current);
    lastFreePoseRef.current.initialized = true;
    lastFreePoseRef.current.position.copy(camera.position);
    lastFreePoseRef.current.look.copy(currentLookRef.current);
    didInitializeCameraRef.current = true;
  }, [camera, controls, currentLookRef]);

  // Build the tween before paint on mode changes, using the last free-orbit
  // pose captured on the previous frame. This avoids any reset/snap caused by
  // React re-rendering, Canvas props, or OrbitControls receiving new props.
  useLayoutEffect(() => {
    if (initialModeRef.current === compareMode) {
      return;
    }
    initialModeRef.current = compareMode;
    const saved = lastFreePoseRef.current;
    const fromLook = saved.initialized ? saved.look.clone() : currentLookRef.current.clone();
    const toLook = new THREE.Vector3(...lookTarget);
    const fromPosition = saved.initialized ? saved.position.clone() : camera.position.clone();
    const toPosition = new THREE.Vector3(...cameraTarget);
    tweenRef.current = { fromLook, toLook, fromPosition, toPosition, elapsed: 0 };
    if (controls) controls.enabled = false;
  }, [camera, cameraTarget, compareMode, controls, currentLookRef, lookTarget]);

  useFrame((_, delta) => {
    const tween = tweenRef.current;
    if (!tween) {
      const look = controls?.target ?? currentLookRef.current;
      controls?.update();
      currentLookRef.current.copy(look);
      lastFreePoseRef.current.initialized = true;
      lastFreePoseRef.current.position.copy(camera.position);
      lastFreePoseRef.current.look.copy(look);
      if (controls) controls.enabled = true;
      return;
    }

    if (controls) controls.enabled = false;
    tween.elapsed += delta;
    const t = Math.min(1, tween.elapsed / TRANSITION_DURATION);
    const e = easeInOutCubic(t);

    const look = tween.fromLook.clone().lerp(tween.toLook, e);
    camera.position.lerpVectors(tween.fromPosition, tween.toPosition, e);
    camera.lookAt(look);
    if (controls) {
      controls.target.copy(look);
      controls.update();
    }
    currentLookRef.current.copy(look);
    if (t >= 1) {
      tweenRef.current = null;
      lastFreePoseRef.current.initialized = true;
      lastFreePoseRef.current.position.copy(tween.toPosition);
      lastFreePoseRef.current.look.copy(tween.toLook);
      if (controls) controls.enabled = true;
    }
  });
  return null;
}






interface SceneProps {
  compareMode?: boolean;
  onAnalysis?: (a: CompareAnalysis) => void;
  generatedParams?: Sample000035Params;
}

const DISPLAY_FRAME_TARGET = 2.7;
const PODIUM_DECK_Y = 0.05;
const PODIUM_CLEARANCE_Y = 0.006;
const FACE_KEYS: FaceKey[] = ["px", "nx", "py", "ny", "pz", "nz"];
const FACE_LABELS: Record<FaceKey, string> = {
  px: "Right (+X)",
  nx: "Left (−X)",
  py: "Top (+Y)",
  ny: "Bottom (−Y)",
  pz: "Front (+Z)",
  nz: "Back (−Z)",
};

function deformPoint(
  point: THREE.Vector3,
  bounds: THREE.Box3,
  activeDrag: ActiveDrag,
  mode: "solid" | "cloud",
) {
  if (!activeDrag || activeDrag.delta === 0) return point.clone();
  const out = point.clone();
  const { face, delta, pin } = activeDrag;
  const size = new THREE.Vector3();
  bounds.getSize(size);
  const spanX = Math.max(size.x, 1e-4);
  const spanY = Math.max(size.y, 1e-4);
  const spanZ = Math.max(size.z, 1e-4);

  let axisAmount = 0;
  let radial = 1;
  switch (face) {
    case "px": {
      axisAmount = THREE.MathUtils.clamp((point.x - bounds.min.x) / spanX, 0, 1);
      const dy = point.y - pin.y;
      const dz = point.z - pin.z;
      radial = 1 - Math.min(1, Math.hypot(dy / spanY, dz / spanZ) / 0.85);
      out.x += delta * (mode === "cloud" ? Math.max(0, radial) : Math.pow(axisAmount, 1.15));
      break;
    }
    case "nx": {
      axisAmount = THREE.MathUtils.clamp((bounds.max.x - point.x) / spanX, 0, 1);
      const dy = point.y - pin.y;
      const dz = point.z - pin.z;
      radial = 1 - Math.min(1, Math.hypot(dy / spanY, dz / spanZ) / 0.85);
      out.x -= delta * (mode === "cloud" ? Math.max(0, radial) : Math.pow(axisAmount, 1.15));
      break;
    }
    case "py": {
      axisAmount = THREE.MathUtils.clamp((point.y - bounds.min.y) / spanY, 0, 1);
      const dx = point.x - pin.x;
      const dz = point.z - pin.z;
      radial = 1 - Math.min(1, Math.hypot(dx / spanX, dz / spanZ) / 0.85);
      out.y += delta * (mode === "cloud" ? Math.max(0, radial) : Math.pow(axisAmount, 1.15));
      break;
    }
    case "ny": {
      axisAmount = THREE.MathUtils.clamp((bounds.max.y - point.y) / spanY, 0, 1);
      const dx = point.x - pin.x;
      const dz = point.z - pin.z;
      radial = 1 - Math.min(1, Math.hypot(dx / spanX, dz / spanZ) / 0.85);
      out.y -= delta * (mode === "cloud" ? Math.max(0, radial) : Math.pow(axisAmount, 1.15));
      break;
    }
    case "pz": {
      axisAmount = THREE.MathUtils.clamp((point.z - bounds.min.z) / spanZ, 0, 1);
      const dx = point.x - pin.x;
      const dy = point.y - pin.y;
      radial = 1 - Math.min(1, Math.hypot(dx / spanX, dy / spanY) / 0.85);
      out.z += delta * (mode === "cloud" ? Math.max(0, radial) : Math.pow(axisAmount, 1.15));
      break;
    }
    case "nz": {
      axisAmount = THREE.MathUtils.clamp((bounds.max.z - point.z) / spanZ, 0, 1);
      const dx = point.x - pin.x;
      const dy = point.y - pin.y;
      radial = 1 - Math.min(1, Math.hypot(dx / spanX, dy / spanY) / 0.85);
      out.z -= delta * (mode === "cloud" ? Math.max(0, radial) : Math.pow(axisAmount, 1.15));
      break;
    }
  }
  return out;
}

function computeBoundsFromGeometry(geometry: THREE.BufferGeometry) {
  const attr = geometry.getAttribute("position") as THREE.BufferAttribute;
  const box = new THREE.Box3();
  const p = new THREE.Vector3();
  for (let i = 0; i < attr.count; i++) {
    p.set(attr.getX(i), attr.getY(i), attr.getZ(i));
    box.expandByPoint(p);
  }
  return box;
}

function normalizeGeometryForDisplay(g: THREE.BufferGeometry): THREE.BufferGeometry {
  const out = g.clone();
  out.rotateX(-Math.PI / 2);
  out.computeBoundingBox();
  const bb = out.boundingBox!;
  const size = new THREE.Vector3();
  const center = new THREE.Vector3();
  bb.getSize(size);
  bb.getCenter(center);
  const longest = Math.max(size.x, size.y, size.z) || 1;
  const scale = DISPLAY_FRAME_TARGET / longest;
  // Align every asset to the same presentation frame:
  // - centred in X/Z
  // - resting on the podium at minY = 0
  out.translate(-center.x, -bb.min.y, -center.z);
  out.scale(scale, scale, scale);
  out.computeVertexNormals();
  return out;
}

function useNormalizedStl(url: string | null): THREE.BufferGeometry | null {
  const [geom, setGeom] = useState<THREE.BufferGeometry | null>(null);
  useEffect(() => {
    if (!url) {
      setGeom(null);
      return;
    }
    let cancelled = false;
    const loader = new STLLoader();
    loader.load(
      url,
      (g) => {
        if (cancelled) return;
        setGeom(normalizeGeometryForDisplay(g));
      },
      undefined,
      () => {
        if (!cancelled) setGeom(null);
      },
    );
    return () => {
      cancelled = true;
    };
  }, [url]);
  return geom;
}

function StlSolid({
  geometry,
  color = "#8f97a3",
  opacity = 1,
  activeDrag = null,
}: {
  geometry: THREE.BufferGeometry | null;
  color?: string;
  opacity?: number;
  activeDrag?: ActiveDrag;
}) {
  if (!geometry) return null;
  const displayGeometry = useMemo(() => {
    const attr = geometry.getAttribute("position") as THREE.BufferAttribute;
    const bounds = computeBoundsFromGeometry(geometry);
    const clone = geometry.clone();
    const pos = new Float32Array(attr.count * 3);
    const p = new THREE.Vector3();
    for (let i = 0; i < attr.count; i++) {
      p.set(attr.getX(i), attr.getY(i), attr.getZ(i));
      const next = deformPoint(p, bounds, activeDrag, "solid");
      pos[i * 3] = next.x;
      pos[i * 3 + 1] = next.y;
      pos[i * 3 + 2] = next.z;
    }
    clone.setAttribute("position", new THREE.BufferAttribute(pos, 3));
    clone.computeVertexNormals();
    return clone;
  }, [geometry, activeDrag]);

  useEffect(() => () => displayGeometry.dispose(), [displayGeometry]);
  return (
    <group position={[0, PODIUM_DECK_Y + PODIUM_CLEARANCE_Y, 0]}>
      <mesh geometry={displayGeometry} castShadow receiveShadow>
        <meshStandardMaterial
          color={color}
          roughness={0.28}
          metalness={0.12}
          transparent={opacity < 1}
          opacity={opacity}
        />
      </mesh>
    </group>
  );
}

function StlPointCloud({
  geometry,
  count = 14000,
  color = "#60a5fa",
  activeDrag = null,
}: {
  geometry: THREE.BufferGeometry | null;
  count?: number;
  color?: string;
  activeDrag?: ActiveDrag;
}) {
  const sampled = useMemo(() => {
    if (!geometry) return null;
    const attr = geometry.getAttribute("position") as THREE.BufferAttribute;
    const bounds = computeBoundsFromGeometry(geometry);
    const pts = new Float32Array(count * 3);
    for (let i = 0; i < count; i++) {
      const idx = Math.floor(Math.random() * attr.count);
      pts[i * 3] = attr.getX(idx) + (Math.random() - 0.5) * 0.004;
      pts[i * 3 + 1] = attr.getY(idx) + Math.random() * 0.004;
      pts[i * 3 + 2] = attr.getZ(idx) + (Math.random() - 0.5) * 0.004;
    }
    return { pts, bounds };
  }, [geometry, count]);

  const positions = useMemo(() => {
    if (!sampled) return null;
    const out = sampled.pts.slice();
    if (!activeDrag) return out;
    const p = new THREE.Vector3();
    for (let i = 0; i < count; i++) {
      p.set(sampled.pts[i * 3], sampled.pts[i * 3 + 1], sampled.pts[i * 3 + 2]);
      const next = deformPoint(p, sampled.bounds, activeDrag, "cloud");
      out[i * 3] = next.x;
      out[i * 3 + 1] = next.y;
      out[i * 3 + 2] = next.z;
    }
    return out;
  }, [sampled, activeDrag, count]);

  if (!positions || !sampled) return null;
  return (
    <group position={[0, PODIUM_DECK_Y + PODIUM_CLEARANCE_Y, 0]}>
      <points>
        <bufferGeometry>
          <bufferAttribute attach="attributes-position" array={positions} count={count} itemSize={3} />
        </bufferGeometry>
        <pointsMaterial size={0.03} color={color} sizeAttenuation transparent opacity={0.95} />
      </points>
    </group>
  );
}

function FaceDragLayer({
  geometry,
  activeDrag,
  onDragChange,
}: {
  geometry: THREE.BufferGeometry | null;
  activeDrag: ActiveDrag;
  onDragChange: (drag: ActiveDrag) => void;
}) {
  const { gl, camera } = useThree();
  const groupRef = useRef<THREE.Group>(null);
  const [hovered, setHovered] = useState<FaceKey | null>(null);
  const dragStateRef = useRef<{
    face: FaceKey;
    startScreen: THREE.Vector2;
    axisScreen: THREE.Vector2;
    pixelsPerUnit: number;
    pin: THREE.Vector3;
  } | null>(null);

  const bounds = useMemo(
    () => (geometry ? computeBoundsFromGeometry(geometry) : null),
    [geometry],
  );

  const worldDirToScreen = useMemo(
    () => (origin: THREE.Vector3, dir: THREE.Vector3) => {
      const a = origin.clone().project(camera);
      const b = origin.clone().add(dir).project(camera);
      const rect = gl.domElement.getBoundingClientRect();
      return new THREE.Vector2(((b.x - a.x) * rect.width) / 2, ((a.y - b.y) * rect.height) / 2);
    },
    [camera, gl],
  );

  useEffect(() => {
    const handleMove = (ev: PointerEvent) => {
      const state = dragStateRef.current;
      if (!state) return;
      const dx = ev.clientX - state.startScreen.x;
      const dy = ev.clientY - state.startScreen.y;
      const projected = dx * state.axisScreen.x + dy * state.axisScreen.y;
      const delta = projected / state.pixelsPerUnit;
      onDragChange({ face: state.face, delta, pin: state.pin.clone() });
    };
    const handleUp = () => {
      if (!dragStateRef.current) return;
      dragStateRef.current = null;
      onDragChange(null);
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
  }, [gl, onDragChange]);

  if (!bounds) return null;
  const size = new THREE.Vector3();
  const center = new THREE.Vector3();
  bounds.getSize(size);
  bounds.getCenter(center);

  const faceDefs: Array<{
    key: FaceKey;
    rotation: [number, number, number];
    position: [number, number, number];
    plane: [number, number];
    normal: THREE.Vector3;
  }> = [
    {
      key: "px",
      rotation: [0, Math.PI / 2, 0],
      position: [bounds.max.x + 0.002, center.y, center.z],
      plane: [size.z, size.y],
      normal: new THREE.Vector3(1, 0, 0),
    },
    {
      key: "nx",
      rotation: [0, -Math.PI / 2, 0],
      position: [bounds.min.x - 0.002, center.y, center.z],
      plane: [size.z, size.y],
      normal: new THREE.Vector3(-1, 0, 0),
    },
    {
      key: "py",
      rotation: [-Math.PI / 2, 0, 0],
      position: [center.x, bounds.max.y + 0.002, center.z],
      plane: [size.x, size.z],
      normal: new THREE.Vector3(0, 1, 0),
    },
    {
      key: "ny",
      rotation: [Math.PI / 2, 0, 0],
      position: [center.x, bounds.min.y - 0.002, center.z],
      plane: [size.x, size.z],
      normal: new THREE.Vector3(0, -1, 0),
    },
    {
      key: "pz",
      rotation: [0, 0, 0],
      position: [center.x, center.y, bounds.max.z + 0.002],
      plane: [size.x, size.y],
      normal: new THREE.Vector3(0, 0, 1),
    },
    {
      key: "nz",
      rotation: [0, Math.PI, 0],
      position: [center.x, center.y, bounds.min.z - 0.002],
      plane: [size.x, size.y],
      normal: new THREE.Vector3(0, 0, -1),
    },
  ];

  return (
    <group ref={groupRef} position={[0, PODIUM_DECK_Y + PODIUM_CLEARANCE_Y, 0]}>
      {faceDefs.map((face) => {
        const lit = activeDrag?.face === face.key || hovered === face.key;
        return (
          <mesh
            key={face.key}
            position={face.position}
            rotation={face.rotation}
            onPointerOver={(e) => {
              e.stopPropagation();
              setHovered(face.key);
              if (!dragStateRef.current) gl.domElement.style.cursor = "grab";
            }}
            onPointerOut={() => {
              setHovered((prev) => (prev === face.key ? null : prev));
              if (!dragStateRef.current) gl.domElement.style.cursor = "default";
            }}
            onPointerDown={(e: ThreeEvent<PointerEvent>) => {
              e.stopPropagation();
              (e.target as Element)?.setPointerCapture?.(e.pointerId);
              const pin = e.point.clone();
              groupRef.current?.worldToLocal(pin);
              const worldOrigin = e.point.clone();
              const worldNormal = face.normal.clone();
              const axis = worldDirToScreen(worldOrigin, worldNormal).normalize();
              const pixelsPerUnit = Math.max(worldDirToScreen(worldOrigin, worldNormal).length(), 30);
              dragStateRef.current = {
                face: face.key,
                startScreen: new THREE.Vector2(e.clientX, e.clientY),
                axisScreen: axis,
                pixelsPerUnit,
                pin,
              };
              onDragChange({ face: face.key, delta: 0, pin });
              gl.domElement.style.cursor = "grabbing";
            }}
          >
            <planeGeometry args={face.plane} />
            <meshBasicMaterial color="#3b9bff" transparent opacity={lit ? 0.18 : 0} depthWrite={false} side={THREE.DoubleSide} />
          </mesh>
        );
      })}
      {activeDrag && (
        <Html position={[activeDrag.pin.x, activeDrag.pin.y + 0.15, activeDrag.pin.z]} center distanceFactor={7} zIndexRange={[100, 0]} style={{ pointerEvents: "none" }}>
          <div className="floating-label whitespace-nowrap">
            {activeDrag.delta >= 0 ? "+" : ""}
            {Math.round(activeDrag.delta * 100)}mm
          </div>
        </Html>
      )}
    </group>
  );
}

/** Per-Compare-mode summary surfaced to the side panel. */
export interface CompareAnalysis {
  /** Number of faces (out of 6) where Generated CAD matches Ground Truth. */
  matchedFaces: number;
  /** Number of faces with any miss (Generated didn't reach as far as Truth). */
  missedFaces: number;
  /** Sum of all matched extrusion lengths in millimetres. */
  matchedMm: number;
  /** Sum of all missed extrusion lengths in millimetres. */
  missedMm: number;
  /** Coverage % = matchedMm / (matchedMm + missedMm) — 100% if no extrusions. */
  coverage: number;
  /** Per-face breakdown for the table in the side panel. */
  faces: Array<{ key: FaceKey; label: string; matchMm: number; missMm: number }>;
}

export function Scene({
  compareMode = false,
  onAnalysis,
  generatedParams = SAMPLE_000035_INITIAL_PARAMS,
}: SceneProps) {
  const pointCloudGeometry = useNormalizedStl("/demos/deepcadimg_000035_recon_noisy.stl");
  const groundTruthGeometry = useNormalizedStl("/demos/deepcadimg_000035.stl");
  const [activeDrag, setActiveDrag] = useState<ActiveDrag>(null);
  // Shared rotation value — all three stages read from this so they spin
  // in perfect unison (same speed, same angle).
  const sharedRotation = useRef(0);

  // Live-tracked OrbitControls target. Lets the tween read the user's
  // current look point at the moment they hit Compare, even though
  // OrbitControls is briefly unmounted during the tween.
  const currentLookRef = useRef(new THREE.Vector3(0, 0.1, 0));

  const compareAnalysis = useMemo<CompareAnalysis>(() => {
    const faces = FACE_KEYS.map((key) => ({
      key,
      label: FACE_LABELS[key],
      matchMm: 0,
      missMm: 0,
    }));
    const deltas = [
      Math.abs(generatedParams.baseHeightMm - SAMPLE_000035_INITIAL_PARAMS.baseHeightMm),
      Math.abs(generatedParams.bossHeightMm - SAMPLE_000035_INITIAL_PARAMS.bossHeightMm),
      Math.abs(
        generatedParams.counterboreDepthMm - SAMPLE_000035_INITIAL_PARAMS.counterboreDepthMm,
      ),
    ];
    const gtTotalMm =
      SAMPLE_000035_INITIAL_PARAMS.baseHeightMm +
      SAMPLE_000035_INITIAL_PARAMS.bossHeightMm +
      SAMPLE_000035_INITIAL_PARAMS.counterboreDepthMm;
    const missedMm = Math.round(deltas.reduce((sum, v) => sum + v, 0));
    const matchedMm = Math.max(Math.round(gtTotalMm - missedMm), 0);
    const editedFeatures = deltas.filter((v) => v > 1e-6).length;
    const coverage = matchedMm + missedMm === 0 ? 100 : (matchedMm / (matchedMm + missedMm)) * 100;

    return {
      matchedFaces: 3 - editedFeatures,
      missedFaces: editedFeatures,
      matchedMm,
      missedMm,
      coverage,
      faces,
    };
  }, [generatedParams]);

  useEffect(() => {
    if (!onAnalysis) return;
    onAnalysis(compareAnalysis);
  }, [compareAnalysis, onAnalysis]);


  // Compare-mode layout: hide point cloud, slide the remaining two podiums
  // closer together and recenter the camera between them.
  const cloudTarget = compareMode
    ? { pos: [-7, 0, 0] as [number, number, number], scale: 0.4, opacity: 0 }
    : { pos: [-3.8, 0, 0] as [number, number, number], scale: 0.96, opacity: 1 };
  const genTarget = compareMode
    ? { pos: [-1.6, 0, 0] as [number, number, number], scale: 1.05, opacity: 1 }
    : { pos: [0, 0, 0] as [number, number, number], scale: 0.96, opacity: 1 };
  const truthTarget = compareMode
    ? { pos: [1.6, 0, 0] as [number, number, number], scale: 1.05, opacity: 1 }
    : { pos: [3.8, 0, 0] as [number, number, number], scale: 0.96, opacity: 1 };
  // Camera glide targets. In normal view the camera sits back to frame all
  // three podiums; in compare mode it moves lower + farther back and aims
  // nearer the podiums' mesh band so both objects stay fully in frame (tall
  // brick + labels). Look Y stays in a range that still clears the on-page
  // analysis panel by keeping the frustum a bit more level than looking
  // straight down.
  const lookTarget: [number, number, number] = compareMode ? [0, 0.85, 0] : [0, 0.1, 0];
  const cameraTarget: [number, number, number] = compareMode
    ? [0, 2.15, 14.6]
    : [0, 4.2, 10.5];

  return (
    <Canvas
      shadows="soft"
      dpr={[1, 2]}
      camera={{ position: [0, 4.2, 10.5], fov: 30 }}
      gl={{ antialias: true, alpha: true }}
      style={{ touchAction: "none" }}
    >
      {/* Transparent canvas — page background shows through */}

        <OrbitControls
        makeDefault
        enablePan={false}
        enableZoom
        zoomSpeed={0.6}
        minDistance={5}
        maxDistance={22}
        enableRotate={!activeDrag}
        enableDamping
        dampingFactor={0.08}
        minPolarAngle={0.15}
        maxPolarAngle={Math.PI / 2 - 0.05}
        target={[
          currentLookRef.current.x,
          currentLookRef.current.y,
          currentLookRef.current.z,
        ]}
      />
      <CameraTransitionController
        compareMode={compareMode}
        cameraTarget={cameraTarget}
        lookTarget={lookTarget}
        currentLookRef={currentLookRef}
      />







      {/* Static lighting rig — only the key directional light casts shadows
          (gives realistic, soft drop shadows under the podiums). The fill
          and ambient stay flat so apparent lighting on the objects doesn't
          shift when geometry is extruded. */}
      <ambientLight intensity={0.65} />
      <directionalLight
        position={[5, 8, 4]}
        intensity={0.85}
        castShadow
        shadow-mapSize-width={2048}
        shadow-mapSize-height={2048}
        shadow-camera-near={0.5}
        shadow-camera-far={28}
        shadow-camera-left={-10}
        shadow-camera-right={10}
        shadow-camera-top={8}
        shadow-camera-bottom={-8}
        shadow-bias={-0.0004}
        shadow-radius={6}
      />
      <directionalLight position={[-5, 3, -2]} intensity={0.3} />

      <Suspense fallback={null}>
        <Environment preset="studio" environmentIntensity={0.55} />

        {/* All three podiums share the same Y, Z and scale — only the X
            position differs — so the objects sit at identical heights and
            depths and rotate in perfect unison via the shared rotation. */}

        {/* LEFT — Point Cloud (cloth-pull interaction) */}
        <AnimatedStage
          targetPosition={cloudTarget.pos}
          targetScale={cloudTarget.scale}
          targetOpacity={cloudTarget.opacity}
        >
          <Podium />
          <CaptionLabel
            title="Point Cloud"
            subtitle="Raw Scan"
            muted
            topY={getSample000035TopY(SAMPLE_000035_INITIAL_PARAMS)}
          />
          <SpinningStage paused={!!activeDrag} rotationRef={sharedRotation}>
            <StlPointCloud geometry={pointCloudGeometry} activeDrag={activeDrag} />
            <FaceDragLayer
              geometry={pointCloudGeometry}
              activeDrag={activeDrag}
              onDragChange={setActiveDrag}
            />
          </SpinningStage>
        </AnimatedStage>

        {/* CENTER — Generated CAD */}
        <AnimatedStage
          targetPosition={genTarget.pos}
          targetScale={genTarget.scale}
          targetOpacity={genTarget.opacity}
        >
          <Podium />
          <CaptionLabel
            title="Generated CAD"
            subtitle="Our Result"
            topY={getSample000035TopY(generatedParams)}
          />
          <SpinningStage paused={!!activeDrag} rotationRef={sharedRotation}>
            <Sample000035Solid
              params={generatedParams}
              color={compareMode ? "#ef4444" : undefined}
              opacity={compareMode ? 0.85 : 1}
            />
          </SpinningStage>
        </AnimatedStage>

        {/* RIGHT — Ground Truth */}
        <AnimatedStage
          targetPosition={truthTarget.pos}
          targetScale={truthTarget.scale}
          targetOpacity={truthTarget.opacity}
        >
          <Podium />
          <CaptionLabel
            title="Ground Truth"
            subtitle="Input STL"
            muted
            topY={getSample000035TopY(SAMPLE_000035_INITIAL_PARAMS)}
          />
          <SpinningStage paused={!!activeDrag} rotationRef={sharedRotation}>
            <StlSolid
              geometry={groundTruthGeometry}
              opacity={compareMode ? 0.92 : 0.92}
              color={compareMode ? "#22c55e" : undefined}
            />
          </SpinningStage>
        </AnimatedStage>


        {/* Invisible floor plane — receives the real cast shadows from
            the directional light via shadowMaterial. The plane itself is
            transparent so the page's uniform white background shows
            through; only the actual shadow under each podium is rendered. */}
        <mesh
          position={[0, -0.549, 0]}
          rotation={[-Math.PI / 2, 0, 0]}
          receiveShadow
        >
          <planeGeometry args={[60, 60]} />
          <shadowMaterial transparent opacity={0.32} color="#0a1020" />
        </mesh>

        {/* Subtle additional contact darkening directly under each podium
            for that close, soft "ambient occlusion" pool. */}
        <ContactShadows
          position={[0, -0.548, 0]}
          opacity={0.35}
          scale={5}
          blur={1.4}
          far={0.5}
          resolution={1024}
          color="#0f1626"
          frames={1}
        />
      </Suspense>
    </Canvas>
  );
}

/**
 * Screen-space caption anchored above each podium in 3D world space.
 * Uses drei's <Html> so the label tracks the podium as the camera moves,
 * always staying directly above the object.
 */
interface CaptionLabelProps {
  title: string;
  subtitle: string;
  muted?: boolean;
  /** Live extrusion of the +Y face (object grows upward). Caption rises
   *  with it so it always sits a fixed distance above the actual top. */
  topY?: number;
}

function CaptionLabel({ title, subtitle, muted, topY = 0 }: CaptionLabelProps) {
  // Local-space top of the brick + studs is roughly y ≈ 0.55. Add the
  // live upward extrusion plus a generous breathing offset so the label
  // sits well above the highest geometry at all times.
  const y = 0.55 + topY + 1.0;
  return (
    <Html
      position={[0, y, 0]}
      center
      distanceFactor={8}
      zIndexRange={[100, 0]}
      style={{ pointerEvents: "none" }}
    >
      <div
        className={`select-none whitespace-nowrap text-center leading-none transition-opacity duration-300 ${
          muted ? "opacity-55" : "opacity-100"
        }`}
      >
        <div className="font-wordmark text-base font-semibold tracking-tight text-foreground">
          {title}
        </div>
        <div className="mt-1 text-[10px] font-medium uppercase tracking-[0.22em] text-muted-foreground">
          {subtitle}
        </div>
      </div>
    </Html>
  );
}
