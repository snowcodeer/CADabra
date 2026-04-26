import { useMemo, useRef, useState, useCallback, useEffect } from "react";
import * as THREE from "three";
import { Html } from "@react-three/drei";
import { ThreeEvent, useThree } from "@react-three/fiber";
import {
  BASE_SIZE,
  Extrusion,
  FaceKey,
  computeBody,
} from "./extrusion";

interface FaceDef {
  key: FaceKey;
  /** Local outward normal (pre-rotation) */
  normal: THREE.Vector3;
  /** Plane orientation to face outward */
  rotation: [number, number, number];
}

const FACES: FaceDef[] = [
  { key: "px", normal: new THREE.Vector3(1, 0, 0), rotation: [0, Math.PI / 2, 0] },
  { key: "nx", normal: new THREE.Vector3(-1, 0, 0), rotation: [0, -Math.PI / 2, 0] },
  { key: "py", normal: new THREE.Vector3(0, 1, 0), rotation: [-Math.PI / 2, 0, 0] },
  { key: "ny", normal: new THREE.Vector3(0, -1, 0), rotation: [Math.PI / 2, 0, 0] },
  { key: "pz", normal: new THREE.Vector3(0, 0, 1), rotation: [0, 0, 0] },
  { key: "nz", normal: new THREE.Vector3(0, 0, -1), rotation: [0, Math.PI, 0] },
];

interface FaceInteractionLayerProps {
  /** Persisted extrusion (shared) */
  extrusion: Extrusion;
  /** In-progress drag (shared so siblings can mirror it). */
  activeDrag: { face: FaceKey; delta: number } | null;
  /** Called when this object starts a drag — parent should pause this object's rotation. */
  onDragStart: (face: FaceKey) => void;
  onDragMove: (face: FaceKey, delta: number) => void;
  onDragEnd: (face: FaceKey, delta: number) => void;
  /** Whether to show the floating distance label (only the active object should). */
  showLabel?: boolean;
}

/**
 * Invisible-but-clickable face planes wrapped around the brick body.
 * Reports drag events upward so all 3 objects (point cloud, generated, ground
 * truth) can stay in sync while geometry is being pulled.
 *
 * World-space normal is computed at drag-start so rotation works from any
 * angle. The parent should freeze the wrapping group's rotation while a drag
 * is active so the face stays under the cursor.
 */
export function FaceInteractionLayer({
  extrusion,
  activeDrag,
  onDragStart,
  onDragMove,
  onDragEnd,
  showLabel = true,
}: FaceInteractionLayerProps) {
  const { gl, camera } = useThree();
  const groupRef = useRef<THREE.Group>(null);

  const [hoveredFace, setHoveredFace] = useState<FaceKey | null>(null);
  const dragStateRef = useRef<{
    startScreen: THREE.Vector2;
    axisScreen: THREE.Vector2;
    pixelsPerUnit: number;
    face: FaceKey;
  } | null>(null);
  const [dragDelta, setDragDelta] = useState(0);

  // Live extrusion (persisted + active drag)
  const liveExt = useMemo<Extrusion>(() => {
    if (!activeDrag) return extrusion;
    return {
      ...extrusion,
      [activeDrag.face]: Math.max(
        0,
        extrusion[activeDrag.face] + activeDrag.delta,
      ),
    };
  }, [extrusion, activeDrag]);

  const body = useMemo(() => computeBody(BASE_SIZE, liveExt), [liveExt]);

  /** World-space pixel direction for a given world origin + world direction. */
  const worldDirToScreen = useCallback(
    (origin: THREE.Vector3, dir: THREE.Vector3) => {
      const a = origin.clone().project(camera);
      const b = origin.clone().add(dir).project(camera);
      const rect = gl.domElement.getBoundingClientRect();
      const dx = ((b.x - a.x) * rect.width) / 2;
      const dy = ((a.y - b.y) * rect.height) / 2;
      return new THREE.Vector2(dx, dy);
    },
    [camera, gl],
  );

  const onFacePointerDown = useCallback(
    (face: FaceDef) => (e: ThreeEvent<PointerEvent>) => {
      e.stopPropagation();
      (e.target as Element)?.setPointerCapture?.(e.pointerId);

      // Convert local face normal -> world (group has rotation from parent spin).
      const worldNormal = face.normal.clone();
      if (groupRef.current) {
        const q = new THREE.Quaternion();
        groupRef.current.getWorldQuaternion(q);
        worldNormal.applyQuaternion(q).normalize();
      }

      // Face center in world space (apply current scale + offset, then matrix)
      const localCenter = face.normal
        .clone()
        .multiply(
          new THREE.Vector3(
            body.size[0] / 2,
            body.size[1] / 2,
            body.size[2] / 2,
          ),
        )
        .add(new THREE.Vector3(...body.offset));
      const worldCenter = localCenter.clone();
      groupRef.current?.localToWorld(worldCenter);

      const screenAxis = worldDirToScreen(worldCenter, worldNormal);
      const pixelsPerUnit = Math.max(screenAxis.length(), 30);
      const axisDir = screenAxis.clone().normalize();

      dragStateRef.current = {
        startScreen: new THREE.Vector2(e.clientX, e.clientY),
        axisScreen: axisDir,
        pixelsPerUnit,
        face: face.key,
      };
      setDragDelta(0);
      onDragStart(face.key);
      gl.domElement.style.cursor = "grabbing";
    },
    [body, worldDirToScreen, onDragStart, gl],
  );

  // Global pointer events keep drag alive when cursor leaves the face plane.
  useEffect(() => {
    const handleMove = (ev: PointerEvent) => {
      const state = dragStateRef.current;
      if (!state) return;
      const dx = ev.clientX - state.startScreen.x;
      const dy = ev.clientY - state.startScreen.y;
      const projected = dx * state.axisScreen.x + dy * state.axisScreen.y;
      const units = projected / state.pixelsPerUnit;
      setDragDelta(units);
      onDragMove(state.face, units);
    };
    const handleUp = () => {
      const state = dragStateRef.current;
      if (!state) return;
      // Allow negative deltas so the user can drag a face back inward and
      // shrink an existing extrusion. The persisted value is clamped at 0
      // so geometry never inverts.
      onDragEnd(state.face, dragDelta);
      setDragDelta(0);
      dragStateRef.current = null;
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
  }, [dragDelta, onDragMove, onDragEnd, gl]);

  const isOwningDrag = !!dragStateRef.current;
  const litFace = isOwningDrag ? dragStateRef.current!.face : hoveredFace;

  // Floating label position (only on the owning object)
  const labelInfo = useMemo(() => {
    if (!isOwningDrag || !showLabel) return null;
    const faceKey = dragStateRef.current!.face;
    const face = FACES.find((f) => f.key === faceKey)!;
    const sizeAlongNormal =
      faceKey === "px" || faceKey === "nx"
        ? body.size[0]
        : faceKey === "py" || faceKey === "ny"
          ? body.size[1]
          : body.size[2];
    const center = new THREE.Vector3(...body.offset).add(
      face.normal.clone().multiplyScalar(sizeAlongNormal / 2 + 0.18),
    );
    const totalUnits = liveExt[faceKey];
    const mm = Math.round(totalUnits * 100);
    return {
      position: center.toArray() as [number, number, number],
      mm,
      sign: mm >= 0 ? "+" : "",
    };
  }, [isOwningDrag, showLabel, body, liveExt]);

  return (
    <group ref={groupRef} position={[0, 0.35, 0]}>
      {FACES.map((face) => {
        const isLit = litFace === face.key;
        const [sx, sy, sz] = body.size;
        const [ox, oy, oz] = body.offset;

        let planeSize: [number, number] = [1, 1];
        let center: [number, number, number] = [0, 0, 0];
        switch (face.key) {
          case "px":
            planeSize = [sz, sy];
            center = [ox + sx / 2 + 0.001, oy, oz];
            break;
          case "nx":
            planeSize = [sz, sy];
            center = [ox - sx / 2 - 0.001, oy, oz];
            break;
          case "py":
            planeSize = [sx, sz];
            center = [ox, oy + sy / 2 + 0.001, oz];
            break;
          case "ny":
            planeSize = [sx, sz];
            center = [ox, oy - sy / 2 - 0.001, oz];
            break;
          case "pz":
            planeSize = [sx, sy];
            center = [ox, oy, oz + sz / 2 + 0.001];
            break;
          case "nz":
            planeSize = [sx, sy];
            center = [ox, oy, oz - sz / 2 - 0.001];
            break;
        }

        return (
          <mesh
            key={face.key}
            position={center}
            rotation={face.rotation}
            onPointerOver={(e) => {
              e.stopPropagation();
              setHoveredFace(face.key);
              if (!dragStateRef.current) gl.domElement.style.cursor = "grab";
            }}
            onPointerOut={() => {
              setHoveredFace((prev) => (prev === face.key ? null : prev));
              if (!dragStateRef.current) gl.domElement.style.cursor = "default";
            }}
            onPointerDown={onFacePointerDown(face)}
          >
            <planeGeometry args={planeSize} />
            <meshBasicMaterial
              color="#3b9bff"
              transparent
              opacity={isLit ? 0.18 : 0}
              depthWrite={false}
              side={THREE.DoubleSide}
            />
          </mesh>
        );
      })}

      {labelInfo && (
        <Html
          position={labelInfo.position}
          center
          distanceFactor={6}
          zIndexRange={[100, 0]}
          style={{ pointerEvents: "none" }}
        >
          <div className="floating-label whitespace-nowrap">
            {labelInfo.sign}{labelInfo.mm}mm
          </div>
        </Html>
      )}
    </group>
  );
}
