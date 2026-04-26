import { Suspense, forwardRef, useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import * as THREE from "three";
import { Canvas, useFrame, useThree } from "@react-three/fiber";
import { Environment, ContactShadows, OrbitControls, Html } from "@react-three/drei";
import { Podium } from "./Podium";
import { PointCloudObject } from "./PointCloudObject";
import { GroundTruthObject } from "./GroundTruthObject";
import { FaceInteractionLayer } from "./ExtrudableObject";
import {
  BASE_SIZE,
  computeBody,
  Extrusion,
  FaceKey,
  ZERO_EXTRUSION,
} from "./extrusion";

type ObjectId = "cloud" | "generated" | "truth";

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

export function Scene({ compareMode = false, onAnalysis }: SceneProps) {
  const [wallExtrusion, setWallExtrusion] = useState<Extrusion>(ZERO_EXTRUSION);
  const [cloudExtrusion, setCloudExtrusion] = useState<Extrusion>(ZERO_EXTRUSION);
  const [activeDrag, setActiveDrag] = useState<{
    owner: ObjectId;
    face: FaceKey;
    delta: number;
  } | null>(null);

  // Shared rotation value — all three stages read from this so they spin
  // in perfect unison (same speed, same angle).
  const sharedRotation = useRef(0);

  // Live-tracked OrbitControls target. Lets the tween read the user's
  // current look point at the moment they hit Compare, even though
  // OrbitControls is briefly unmounted during the tween.
  const currentLookRef = useRef(new THREE.Vector3(0, 0.1, 0));


  const extrusion = useMemo<Extrusion>(() => ({
    px: Math.max(wallExtrusion.px, cloudExtrusion.px),
    nx: Math.max(wallExtrusion.nx, cloudExtrusion.nx),
    py: Math.max(wallExtrusion.py, cloudExtrusion.py),
    ny: Math.max(wallExtrusion.ny, cloudExtrusion.ny),
    pz: Math.max(wallExtrusion.pz, cloudExtrusion.pz),
    nz: Math.max(wallExtrusion.nz, cloudExtrusion.nz),
  }), [wallExtrusion, cloudExtrusion]);

  const handleDragStart = useCallback(
    (owner: ObjectId) => (face: FaceKey) => {
      setActiveDrag({ owner, face, delta: 0 });
    },
    [],
  );

  const handleDragMove = useCallback(
    (owner: ObjectId) => (face: FaceKey, delta: number) => {
      setActiveDrag((prev) =>
        prev && prev.owner === owner ? { ...prev, face, delta } : prev,
      );
    },
    [],
  );

  const handleDragEnd = useCallback(
    (owner: ObjectId) => (face: FaceKey, delta: number) => {
      if (owner === "cloud") {
        setCloudExtrusion((prev) => ({
          ...prev,
          [face]: Math.max(prev[face], Math.max(0, delta)),
        }));
      } else {
        // Drag started visually at extrusion[face] (the current max across
        // sources). The new absolute reach is extrusion[face] + delta. Keep
        // the longest extrusion ever recorded for this face so additional
        // pulls never shrink the other objects' walls.
        const candidate = Math.max(0, extrusion[face] + delta);
        setWallExtrusion((prev) => ({
          ...prev,
          [face]: Math.max(prev[face], candidate),
        }));
      }
      setActiveDrag((prev) => (prev && prev.owner === owner ? null : prev));
    },
    [extrusion],
  );

  const isDragging = !!activeDrag;

  // Visual sync drag for non-owning objects (owner gets it via its own layer).
  // For a cloud drag, only push the other walls outward if the live spike
  // length exceeds the current longest extrusion for that face — otherwise
  // the other objects stay put.
  const syncDrag = activeDrag
    ? {
        face: activeDrag.face,
        delta:
          activeDrag.owner === "cloud"
            ? Math.max(0, activeDrag.delta - extrusion[activeDrag.face])
            : activeDrag.delta,
      }
    : null;

  // Live top extension (object grows upward when the +Y face is pulled).
  // Combines persisted extrusion with the in-flight drag delta so labels
  // float just above the current top of each object in real time.
  const liveTopY = useMemo(() => {
    let top = extrusion.py;
    if (activeDrag && activeDrag.face === "py") {
      const candidate =
        activeDrag.owner === "cloud"
          ? activeDrag.delta
          : extrusion.py + activeDrag.delta;
      top = Math.max(top, candidate);
    }
    return top;
  }, [extrusion.py, activeDrag]);

  // Per-face diff between Generated CAD (wallExtrusion) and Ground Truth
  // (the shared `extrusion`, which is max of cloud + wall). Anything the
  // point cloud reached but the wall didn't = "missed" (red). The shared
  // length both agree on = "matched" (green).
  const faceDiff = useMemo(() => {
    const faces: FaceKey[] = ["px", "nx", "py", "ny", "pz", "nz"];
    const out: Record<FaceKey, { match: number; miss: number }> = {
      px: { match: 0, miss: 0 },
      nx: { match: 0, miss: 0 },
      py: { match: 0, miss: 0 },
      ny: { match: 0, miss: 0 },
      pz: { match: 0, miss: 0 },
      nz: { match: 0, miss: 0 },
    };
    for (const f of faces) {
      const truth = extrusion[f];
      const gen = wallExtrusion[f];
      out[f] = {
        match: Math.min(truth, gen),
        miss: Math.max(0, truth - gen),
      };
    }
    return out;
  }, [extrusion, wallExtrusion]);

  // Surface a high-level summary up to the page so the side panel can show
  // a quick coverage analysis. Recomputed whenever the diff changes.
  useEffect(() => {
    if (!onAnalysis) return;
    const FACE_LABELS: Record<FaceKey, string> = {
      px: "Right (+X)",
      nx: "Left (−X)",
      py: "Top (+Y)",
      ny: "Bottom (−Y)",
      pz: "Front (+Z)",
      nz: "Back (−Z)",
    };
    const faces = (Object.keys(faceDiff) as FaceKey[]).map((k) => ({
      key: k,
      label: FACE_LABELS[k],
      matchMm: Math.round(faceDiff[k].match * 100),
      missMm: Math.round(faceDiff[k].miss * 100),
    }));
    const matchedMm = faces.reduce((s, f) => s + f.matchMm, 0);
    const missedMm = faces.reduce((s, f) => s + f.missMm, 0);
    const matchedFaces = faces.filter((f) => f.matchMm > 0 && f.missMm === 0).length;
    const missedFaces = faces.filter((f) => f.missMm > 0).length;
    const total = matchedMm + missedMm;
    const coverage = total === 0 ? 100 : (matchedMm / total) * 100;
    onAnalysis({ matchedFaces, missedFaces, matchedMm, missedMm, coverage, faces });
  }, [faceDiff, onAnalysis]);


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
        enableRotate={!isDragging}
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
            topY={liveTopY}
          />
          <SpinningStage paused={isDragging} rotationRef={sharedRotation}>
            <PointCloudObject
              extrusion={extrusion}
              activeDrag={syncDrag}
              onDragStart={handleDragStart("cloud")}
              onDragMove={handleDragMove("cloud")}
              onDragEnd={handleDragEnd("cloud")}
              isOwningDrag={activeDrag?.owner === "cloud"}
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
            topY={liveTopY}
          />
          <SpinningStage paused={isDragging} rotationRef={sharedRotation}>
            <GroundTruthObjectVisual
              extrusion={extrusion}
              activeDrag={syncDrag}
              color={compareMode ? "#ef4444" : undefined}
              opacity={compareMode ? 0.85 : 1}
            />
            <FaceInteractionLayer
              extrusion={extrusion}
              activeDrag={syncDrag}
              onDragStart={handleDragStart("generated")}
              onDragMove={handleDragMove("generated")}
              onDragEnd={handleDragEnd("generated")}
              showLabel={activeDrag?.owner === "generated"}
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
            subtitle="DeepCAD"
            muted
            topY={liveTopY}
          />
          <SpinningStage paused={isDragging} rotationRef={sharedRotation}>
            <GroundTruthObject
              opacity={compareMode ? 0.92 : 0.92}
              extrusion={extrusion}
              activeDrag={syncDrag}
              color={compareMode ? "#22c55e" : undefined}
            />
            {/* Per-face diff overlay omitted: in compare mode the entire
                Ground Truth body is tinted green and the Generated CAD red,
                acting as the GitHub-style additions/deletions diff. */}
            <FaceInteractionLayer
              extrusion={extrusion}
              activeDrag={syncDrag}
              onDragStart={handleDragStart("truth")}
              onDragMove={handleDragMove("truth")}
              onDragEnd={handleDragEnd("truth")}
              showLabel={activeDrag?.owner === "truth"}
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

/** Generated CAD uses the same brick visual but rendered fully opaque. */
function GroundTruthObjectVisual(
  props: React.ComponentProps<typeof GroundTruthObject>,
) {
  return <GroundTruthObject {...props} />;
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

/**
 * Compare-mode visual diff: paints colored overlays on the Ground Truth's
 * faces. Green = generated CAD agrees, red = generated CAD missed (the
 * point cloud reached further than the wall did on that face).
 */
interface DiffOverlayProps {
  extrusion: Extrusion;
  faceDiff: Record<FaceKey, { match: number; miss: number }>;
}
const DiffOverlay = forwardRef<THREE.Group, DiffOverlayProps>(function DiffOverlay(
  { extrusion, faceDiff },
  ref,
) {
  const body = useMemo(() => computeBody(BASE_SIZE, extrusion), [extrusion]);
  const [sx, sy, sz] = body.size;
  const [ox, oy, oz] = body.offset;

  // Tiny outward offset so the overlay sits just above the brick face
  // (avoids z-fighting with the white body).
  const eps = 0.005;

  const faces: Array<{
    key: FaceKey;
    size: [number, number];
    position: [number, number, number];
    rotation: [number, number, number];
  }> = [
    { key: "px", size: [sz, sy], position: [ox + sx / 2 + eps, oy, oz], rotation: [0, Math.PI / 2, 0] },
    { key: "nx", size: [sz, sy], position: [ox - sx / 2 - eps, oy, oz], rotation: [0, -Math.PI / 2, 0] },
    { key: "py", size: [sx, sz], position: [ox, oy + sy / 2 + eps, oz], rotation: [-Math.PI / 2, 0, 0] },
    { key: "ny", size: [sx, sz], position: [ox, oy - sy / 2 - eps, oz], rotation: [Math.PI / 2, 0, 0] },
    { key: "pz", size: [sx, sy], position: [ox, oy, oz + sz / 2 + eps], rotation: [0, 0, 0] },
    { key: "nz", size: [sx, sy], position: [ox, oy, oz - sz / 2 - eps], rotation: [0, Math.PI, 0] },
  ];

  return (
    <group ref={ref} position={[0, 0.35, 0]}>
      {faces.map((f) => {
        const diff = faceDiff[f.key];
        let color: string | null = null;
        if (diff.miss > 0.001) color = "#ef4444"; // red — generated missed this region
        else if (diff.match > 0.001) color = "#22c55e"; // green — agreed
        if (!color) return null;
        return (
          <mesh key={f.key} position={f.position} rotation={f.rotation}>
            <planeGeometry args={f.size} />
            <meshBasicMaterial
              color={color}
              transparent
              opacity={0.55}
              depthWrite={false}
              side={THREE.DoubleSide}
            />
          </mesh>
        );
      })}
    </group>
  );
});
