import { useMemo, useRef, useEffect, useState, useCallback } from "react";
import * as THREE from "three";
import { Html } from "@react-three/drei";
import { ThreeEvent, useThree, useFrame } from "@react-three/fiber";
import {
  BASE_SIZE,
  Extrusion,
  FaceKey,
  ZERO_EXTRUSION,
  computeBody,
} from "./extrusion";

interface PointCloudObjectProps {
  /** Approx point count (will be slightly higher due to layered structure) */
  count?: number;

  extrusion?: Extrusion;
  /** Live drag coming from a sibling object (uniform face extrusion). */
  activeDrag?: { face: FaceKey; delta: number } | null;

  /* --- cloth-pull integration --- */
  onDragStart: (face: FaceKey) => void;
  onDragMove: (face: FaceKey, delta: number) => void;
  onDragEnd: (face: FaceKey, delta: number) => void;
  /** True when this object owns the active drag (drives the floating label). */
  isOwningDrag: boolean;
}

/** Face metadata for cloth-pull picking. */
const FACES: Array<{
  key: FaceKey;
  normal: THREE.Vector3;
  rotation: [number, number, number];
}> = [
  { key: "px", normal: new THREE.Vector3(1, 0, 0), rotation: [0, Math.PI / 2, 0] },
  { key: "nx", normal: new THREE.Vector3(-1, 0, 0), rotation: [0, -Math.PI / 2, 0] },
  { key: "py", normal: new THREE.Vector3(0, 1, 0), rotation: [-Math.PI / 2, 0, 0] },
  { key: "ny", normal: new THREE.Vector3(0, -1, 0), rotation: [Math.PI / 2, 0, 0] },
  { key: "pz", normal: new THREE.Vector3(0, 0, 1), rotation: [0, 0, 0] },
  { key: "nz", normal: new THREE.Vector3(0, 0, -1), rotation: [0, Math.PI, 0] },
];

/**
 * Dense, structured point cloud sampled inside the (possibly extruded) brick
 * volume. Three layers:
 *   - Surface shell:   crisp dots biased to the box surface
 *   - Interior fog:    sparser dots in the volume
 *   - Stud caps:       cylindrical samples on the 4x2 studs
 *
 * Interaction model: cloth-pull. The user grabs the nearest dot on a face;
 * that dot becomes the "pin" and rises along the face normal. Surrounding
 * dots within a wide radius rise too, with a smooth cosine falloff so the
 * cloud tents up like a piece of fabric being lifted from a table.
 *
 * The reported drag delta to the parent equals the PEAK displacement, so
 * the sibling CAD bricks extrude their whole face by the same amount.
 */
export function PointCloudObject({
  count = 28000,
  extrusion = ZERO_EXTRUSION,
  activeDrag = null,
  onDragStart,
  onDragMove,
  onDragEnd,
  isOwningDrag,
}: PointCloudObjectProps) {
  const { gl, camera } = useThree();
  const groupRef = useRef<THREE.Group>(null);
  const pointsRef = useRef<THREE.Points>(null);

  // Tracks how much of the shared `extrusion` prop was contributed by THIS
  // cloud's own cloth pulls (so we don't double-count when rendering).
  const ownExtrusionContribRef = useRef<Extrusion>({
    px: 0,
    nx: 0,
    py: 0,
    ny: 0,
    pz: 0,
    nz: 0,
  });

  // Effective extrusion — shared face extrusions from the bricks DO grow the
  // cloud's whole face. We subtract our own contribution (extrusion saved
  // from cloth pulls) so cloth pulls don't double-count: the spike already
  // represents that height locally. Brick-initiated extrudes (which we did
  // NOT contribute to) still grow the cloud uniformly.
  const liveExt = useMemo<Extrusion>(() => {
    const own = ownExtrusionContribRef.current;
    const base: Extrusion = {
      px: Math.max(0, extrusion.px - own.px),
      nx: Math.max(0, extrusion.nx - own.nx),
      py: Math.max(0, extrusion.py - own.py),
      ny: Math.max(0, extrusion.ny - own.ny),
      pz: Math.max(0, extrusion.pz - own.pz),
      nz: Math.max(0, extrusion.nz - own.nz),
    };
    if (!activeDrag || isOwningDrag) return base;
    return {
      ...base,
      [activeDrag.face]: Math.max(
        0,
        base[activeDrag.face] + activeDrag.delta,
      ),
    };
  }, [extrusion, activeDrag, isOwningDrag]);




  // Build the base point cloud whenever the resting volume changes.
  const { basePositions, geometry } = useMemo(() => {
    const body = computeBody(BASE_SIZE, liveExt);
    const [w, h, d] = body.size;
    const [ox, oy, oz] = body.offset;

    const shellCount = Math.floor(count * 0.55);
    const interiorCount = Math.floor(count * 0.25);
    const studCount = count - shellCount - interiorCount;
    const total = shellCount + interiorCount + studCount;

    const positions = new Float32Array(total * 3);

    // ---- Surface shell: pick a random face, then a random point on it ----
    const faceAreas = [h * d, h * d, w * d, w * d, w * h, w * h];
    const totalArea = faceAreas.reduce((a, b) => a + b, 0);
    for (let i = 0; i < shellCount; i++) {
      const r = Math.random() * totalArea;
      let acc = 0;
      let f = 0;
      for (; f < 6; f++) {
        acc += faceAreas[f];
        if (r <= acc) break;
      }
      let x = 0,
        y = 0,
        z = 0;
      const jitter = 0.012; // tiny depth jitter keeps surfaces from looking like decals
      switch (f) {
        case 0: // +X
          x = w / 2 + (Math.random() - 0.5) * jitter;
          y = (Math.random() - 0.5) * h;
          z = (Math.random() - 0.5) * d;
          break;
        case 1: // -X
          x = -w / 2 + (Math.random() - 0.5) * jitter;
          y = (Math.random() - 0.5) * h;
          z = (Math.random() - 0.5) * d;
          break;
        case 2: // +Y
          x = (Math.random() - 0.5) * w;
          y = h / 2 + (Math.random() - 0.5) * jitter;
          z = (Math.random() - 0.5) * d;
          break;
        case 3: // -Y
          x = (Math.random() - 0.5) * w;
          y = -h / 2 + (Math.random() - 0.5) * jitter;
          z = (Math.random() - 0.5) * d;
          break;
        case 4: // +Z
          x = (Math.random() - 0.5) * w;
          y = (Math.random() - 0.5) * h;
          z = d / 2 + (Math.random() - 0.5) * jitter;
          break;
        case 5: // -Z
          x = (Math.random() - 0.5) * w;
          y = (Math.random() - 0.5) * h;
          z = -d / 2 + (Math.random() - 0.5) * jitter;
          break;
      }
      positions[i * 3] = ox + x;
      positions[i * 3 + 1] = oy + y;
      positions[i * 3 + 2] = oz + z;
    }

    // ---- Interior fog ----
    for (let i = 0; i < interiorCount; i++) {
      const idx = (shellCount + i) * 3;
      positions[idx] = ox + (Math.random() - 0.5) * w * 0.94;
      positions[idx + 1] = oy + (Math.random() - 0.5) * h * 0.94;
      positions[idx + 2] = oz + (Math.random() - 0.5) * d * 0.94;
    }

    // ---- Studs (4x2 cylinders pinned to the base footprint) ----
    const [bw, , bd] = BASE_SIZE;
    const cols = 4;
    const rows = 2;
    const stepX = bw / cols;
    const stepZ = bd / rows;
    const studRadius = Math.min(stepX, stepZ) / 2 - 0.05;
    const studHeight = 0.18;
    const topY = oy + h / 2;
    for (let i = 0; i < studCount; i++) {
      const c = Math.floor(Math.random() * cols);
      const r = Math.floor(Math.random() * rows);
      const cx = -bw / 2 + stepX * (c + 0.5);
      const cz = -bd / 2 + stepZ * (r + 0.5);
      // Mostly on the side wall + cap of the cylinder
      const onCap = Math.random() < 0.35;
      const angle = Math.random() * Math.PI * 2;
      const rad = onCap
        ? Math.sqrt(Math.random()) * studRadius
        : studRadius + (Math.random() - 0.5) * 0.01;
      const y = onCap
        ? topY + studHeight + (Math.random() - 0.5) * 0.012
        : topY + Math.random() * studHeight;
      const idx = (shellCount + interiorCount + i) * 3;
      positions[idx] = cx + Math.cos(angle) * rad;
      positions[idx + 1] = y;
      positions[idx + 2] = cz + Math.sin(angle) * rad;
    }

    const geo = new THREE.BufferGeometry();
    // Clone for the live attribute so the base array remains pristine.
    geo.setAttribute(
      "position",
      new THREE.BufferAttribute(positions.slice(), 3),
    );
    return { basePositions: positions, geometry: geo };
    // count is intentionally a dependency: changing it rebuilds the cloud.
  }, [count, liveExt]);

  // Cloth-pull state (only when WE own the drag)
  const dragRef = useRef<{
    face: FaceKey;
    pin: THREE.Vector3; // pin point in local cloud space
    normal: THREE.Vector3; // world-space outward normal at drag-start
    axisScreen: THREE.Vector2;
    pixelsPerUnit: number;
    startScreen: THREE.Vector2;
    radius: number;
    /** Existing extrusion amount when this drag started (>0 if re-picking a frozen spike). */
    initialAmount: number;
  } | null>(null);
  const [pullAmount, setPullAmount] = useState(0);

  /**
   * Persisted "frozen" pulls. When the user releases a cloth-pull, we snapshot
   * the deformation here so it stays in the cloud forever (until reset).
   * Each frame we apply ALL frozen pulls plus any active drag, summed.
   */
  const frozenPullsRef = useRef<
    Array<{ face: FaceKey; pin: THREE.Vector3; radius: number; amount: number }>
  >([]);

  // (ownExtrusionContribRef declared earlier — see top of component.)

  /**
   * Re-project a stored pin onto the CURRENT face plane (after the brick
   * walls may have grown via shared extrusion). This keeps frozen spikes
   * anchored to the moving surface so a wall extrude carries them outward.
   */
  const projectPinToCurrentFace = useCallback(
    (face: FaceKey, basePin: THREE.Vector3): THREE.Vector3 => {
      const body = computeBody(BASE_SIZE, liveExt);
      const [sx, sy, sz] = body.size;
      const [ox, oy, oz] = body.offset;
      const out = basePin.clone();
      switch (face) {
        case "px":
          out.x = ox + sx / 2;
          break;
        case "nx":
          out.x = ox - sx / 2;
          break;
        case "py":
          out.y = oy + sy / 2;
          break;
        case "ny":
          out.y = oy - sy / 2;
          break;
        case "pz":
          out.z = oz + sz / 2;
          break;
        case "nz":
          out.z = oz - sz / 2;
          break;
      }
      return out;
    },
    [liveExt],
  );


  /** Convert a world-space direction into a pixel-space vector. */
  const worldDirToScreen = useCallback(
    (origin: THREE.Vector3, dir: THREE.Vector3) => {
      const a = origin.clone().project(camera);
      const b = origin.clone().add(dir).project(camera);
      const rect = gl.domElement.getBoundingClientRect();
      return new THREE.Vector2(
        ((b.x - a.x) * rect.width) / 2,
        ((a.y - b.y) * rect.height) / 2,
      );
    },
    [camera, gl],
  );

  // Apply ALL deformations (frozen + active) every frame.
  // The cost is O(points * pulls); pulls is small (1 active + N frozen).
  useFrame(() => {
    if (!pointsRef.current) return;
    const attr = pointsRef.current.geometry.attributes
      .position as THREE.BufferAttribute;
    const arr = attr.array as Float32Array;

    // Build the active pull list for this frame, with pins re-projected
    // onto the CURRENT (possibly extruded) face plane.
    const pulls: Array<{
      face: FaceKey;
      pin: THREE.Vector3;
      radius: number;
      amount: number;
    }> = frozenPullsRef.current.map((p) => ({
      ...p,
      pin: projectPinToCurrentFace(p.face, p.pin),
    }));
    if (dragRef.current && isOwningDrag) {
      pulls.push({
        face: dragRef.current.face,
        pin: projectPinToCurrentFace(dragRef.current.face, dragRef.current.pin),
        radius: dragRef.current.radius,
        amount: pullAmount,
      });
    }

    // Fast path: nothing to apply -> snap to base
    if (pulls.length === 0) {
      arr.set(basePositions);
      attr.needsUpdate = true;
      return;
    }

    for (let i = 0; i < arr.length; i += 3) {
      let x = basePositions[i];
      let y = basePositions[i + 1];
      let z = basePositions[i + 2];

      for (let p = 0; p < pulls.length; p++) {
        const pull = pulls[p];
        const n = FACES.find((f) => f.key === pull.face)!.normal;
        const ox = basePositions[i] - pull.pin.x;
        const oy = basePositions[i + 1] - pull.pin.y;
        const oz = basePositions[i + 2] - pull.pin.z;
        const along = ox * n.x + oy * n.y + oz * n.z;
        const tx = ox - along * n.x;
        const ty = oy - along * n.y;
        const tz = oz - along * n.z;
        const tangentialDist = Math.sqrt(tx * tx + ty * ty + tz * tz);

        if (tangentialDist < pull.radius) {
          const t = tangentialDist / pull.radius;
          const influence = 0.5 * (1 + Math.cos(Math.PI * t));
          // Only lift points on the outward side of the pin's face
          const sameSide = along >= -0.02;
          if (sameSide) {
            const lift = influence * pull.amount;
            x += n.x * lift;
            y += n.y * lift;
            z += n.z * lift;
          }
        }
      }

      arr[i] = x;
      arr[i + 1] = y;
      arr[i + 2] = z;
    }
    attr.needsUpdate = true;
  });

  /**
   * Pointer-down on the cloud:
   *   1. If the click lands near the TIP of any frozen spike, re-pick that
   *      pull (drag it further out, or back in to flatten/remove).
   *   2. Otherwise, start a fresh cloth pull on the closest face.
   */
  const handlePointerDown = useCallback(
    (e: ThreeEvent<PointerEvent>) => {
      e.stopPropagation();
      (e.target as Element)?.setPointerCapture?.(e.pointerId);

      const rect = gl.domElement.getBoundingClientRect();
      const clickPx = new THREE.Vector2(e.clientX, e.clientY);

      // ----- (1) Try to re-pick a frozen spike's tip -----
      const REPICK_RADIUS_PX = 42;
      let bestRepick: {
        index: number;
        distPx: number;
      } | null = null;

      const groupQuat = new THREE.Quaternion();
      groupRef.current?.getWorldQuaternion(groupQuat);

      frozenPullsRef.current.forEach((pull, idx) => {
        const localNormal = FACES.find((f) => f.key === pull.face)!.normal;
        // Spike tip in local space = pin + normal * amount
        const tipLocal = pull.pin
          .clone()
          .add(localNormal.clone().multiplyScalar(pull.amount));
        // Project to world, then to screen pixels
        const tipWorld = tipLocal.clone();
        groupRef.current?.localToWorld(tipWorld);
        const ndc = tipWorld.clone().project(camera);
        const tipPx = new THREE.Vector2(
          ((ndc.x + 1) / 2) * rect.width + rect.left,
          ((1 - ndc.y) / 2) * rect.height + rect.top,
        );
        const distPx = tipPx.distanceTo(clickPx);
        if (distPx < REPICK_RADIUS_PX && (!bestRepick || distPx < bestRepick.distPx)) {
          bestRepick = { index: idx, distPx };
        }
      });

      if (bestRepick !== null) {
        // Re-pick: lift this pull out of the frozen list and resume editing it.
        const repickIndex: number = (bestRepick as { index: number; distPx: number }).index;
        const pull = frozenPullsRef.current[repickIndex];
        frozenPullsRef.current = frozenPullsRef.current.filter(
          (_, i) => i !== repickIndex,
        );

        const localNormal = FACES.find((f) => f.key === pull.face)!.normal;
        const worldNormal = localNormal.clone().applyQuaternion(groupQuat).normalize();

        // Anchor the screen-axis math at the spike's CURRENT tip position
        const tipLocal = pull.pin
          .clone()
          .add(localNormal.clone().multiplyScalar(pull.amount));
        const worldOrigin = tipLocal.clone();
        groupRef.current?.localToWorld(worldOrigin);
        const screenAxis = worldDirToScreen(worldOrigin, worldNormal);
        const pixelsPerUnit = Math.max(screenAxis.length(), 30);

        dragRef.current = {
          face: pull.face,
          pin: pull.pin.clone(),
          normal: worldNormal,
          axisScreen: screenAxis.clone().normalize(),
          pixelsPerUnit,
          startScreen: clickPx.clone(),
          radius: pull.radius,
          initialAmount: pull.amount,
        };
        setPullAmount(pull.amount);
        onDragStart(pull.face);
        gl.domElement.style.cursor = "grabbing";
        return;
      }

      // ----- (2) Fresh pull on the closest face -----
      const localHit = e.point.clone();
      groupRef.current?.worldToLocal(localHit);

      const body = computeBody(BASE_SIZE, liveExt);
      const [sx, sy, sz] = body.size;
      const [ox, oy, oz] = body.offset;

      const dists: Record<FaceKey, number> = {
        px: Math.abs(localHit.x - (ox + sx / 2)),
        nx: Math.abs(localHit.x - (ox - sx / 2)),
        py: Math.abs(localHit.y - (oy + sy / 2)),
        ny: Math.abs(localHit.y - (oy - sy / 2)),
        pz: Math.abs(localHit.z - (oz + sz / 2)),
        nz: Math.abs(localHit.z - (oz - sz / 2)),
      };
      const faceKey = (Object.keys(dists) as FaceKey[]).reduce((a, b) =>
        dists[a] < dists[b] ? a : b,
      );
      const face = FACES.find((f) => f.key === faceKey)!;

      const pin = localHit.clone();
      switch (faceKey) {
        case "px":
          pin.x = ox + sx / 2;
          break;
        case "nx":
          pin.x = ox - sx / 2;
          break;
        case "py":
          pin.y = oy + sy / 2;
          break;
        case "ny":
          pin.y = oy - sy / 2;
          break;
        case "pz":
          pin.z = oz + sz / 2;
          break;
        case "nz":
          pin.z = oz - sz / 2;
          break;
      }

      const worldNormal = face.normal.clone().applyQuaternion(groupQuat).normalize();
      const worldOrigin = pin.clone();
      groupRef.current?.localToWorld(worldOrigin);
      const screenAxis = worldDirToScreen(worldOrigin, worldNormal);
      const pixelsPerUnit = Math.max(screenAxis.length(), 30);

      const inPlaneA =
        faceKey === "px" || faceKey === "nx"
          ? sz
          : sx;
      const inPlaneB =
        faceKey === "px" || faceKey === "nx"
          ? sy
          : faceKey === "py" || faceKey === "ny"
            ? sz
            : sy;
      // Tighter pinch — only the local region around the pin tents up
      const radius = Math.max(inPlaneA, inPlaneB) * 0.22;

      dragRef.current = {
        face: faceKey,
        pin,
        normal: worldNormal,
        axisScreen: screenAxis.clone().normalize(),
        pixelsPerUnit,
        startScreen: clickPx.clone(),
        radius,
        initialAmount: 0,
      };
      setPullAmount(0);
      onDragStart(faceKey);
      gl.domElement.style.cursor = "grabbing";
    },
    [liveExt, worldDirToScreen, onDragStart, gl],
  );

  // Global move/up so dragging keeps working off the points
  useEffect(() => {
    const move = (ev: PointerEvent) => {
      const s = dragRef.current;
      if (!s) return;
      const dx = ev.clientX - s.startScreen.x;
      const dy = ev.clientY - s.startScreen.y;
      const projected = dx * s.axisScreen.x + dy * s.axisScreen.y;
      // Total pull = whatever the spike already had + the screen-projected delta.
      // This lets the user drag the tip further out OR back in to flatten it.
      const units = s.initialAmount + projected / s.pixelsPerUnit;
      setPullAmount(units);
      // Report current peak to parent so CAD bricks mirror the pull live.
      onDragMove(s.face, units);
    };
    const up = () => {
      const s = dragRef.current;
      if (!s) return;
      // Freeze the spike (or drop it if dragged back to ~flat).
      if (Math.abs(pullAmount) > 0.005) {
        frozenPullsRef.current = [
          ...frozenPullsRef.current,
          {
            face: s.face,
            pin: s.pin.clone(),
            radius: s.radius,
            amount: pullAmount,
          },
        ];
      }
      // The CAD bricks save the absolute peak for this face. Repeated pulls on
      // the same wall should keep the furthest reach, not add each pull.
      const brickDelta = pullAmount;
      // Track our own contribution so the cloud doesn't ALSO grow its wall
      // when this brick extrusion comes back through the shared `extrusion` prop.
      // Keep the furthest cloud pull per face; shorter later pulls must not
      // reduce this subtraction or the saved shared wall reappears as a second
      // rectangular extrusion after pointer-up.
      ownExtrusionContribRef.current = {
        ...ownExtrusionContribRef.current,
        [s.face]: Math.max(ownExtrusionContribRef.current[s.face], brickDelta, 0),
      };
      onDragEnd(s.face, brickDelta);
      dragRef.current = null;
      setPullAmount(0);
      gl.domElement.style.cursor = "default";
    };
    window.addEventListener("pointermove", move);
    window.addEventListener("pointerup", up);
    window.addEventListener("pointercancel", up);
    return () => {
      window.removeEventListener("pointermove", move);
      window.removeEventListener("pointerup", up);
      window.removeEventListener("pointercancel", up);
    };
  }, [pullAmount, onDragMove, onDragEnd, gl]);

  const material = useMemo(
    () =>
      new THREE.PointsMaterial({
        color: "#1a1f2c",
        size: 0.017,
        sizeAttenuation: true,
        transparent: true,
        opacity: 0.85,
      }),
    [],
  );

  /* ---- Spike-fill: extra dots that grow with the pull ---- */

  /** Max fill dots we'll ever need — generous so we don't have to resize. */
  const MAX_FILL = 32000;
  /** Dots generated per world-unit of pull amount per pull. */
  const FILL_DENSITY = 6000;

  const fillGeometry = useMemo(() => {
    const buf = new Float32Array(MAX_FILL * 3);
    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.BufferAttribute(buf, 3));
    geo.setDrawRange(0, 0);
    return geo;
  }, []);

  /**
   * A small deterministic PRNG (mulberry32) so each pull's fill dots stay
   * stable frame-to-frame instead of shimmering with Math.random().
   */
  const fillPointsRef = useRef<THREE.Points>(null);
  useFrame(() => {
    if (!fillPointsRef.current) return;
    const attr = fillPointsRef.current.geometry.attributes
      .position as THREE.BufferAttribute;
    const arr = attr.array as Float32Array;

    const pulls: Array<{
      face: FaceKey;
      pin: THREE.Vector3;
      radius: number;
      amount: number;
      seed: number;
    }> = frozenPullsRef.current.map((p, i) => ({
      ...p,
      pin: projectPinToCurrentFace(p.face, p.pin),
      seed: 0x9e37 + i * 1013,
    }));
    if (dragRef.current && isOwningDrag) {
      pulls.push({
        face: dragRef.current.face,
        pin: projectPinToCurrentFace(dragRef.current.face, dragRef.current.pin),
        radius: dragRef.current.radius,
        amount: pullAmount,
        seed: 0xa11ce, // stable seed for the active pull
      });
    }

    let cursor = 0;
    for (let p = 0; p < pulls.length; p++) {
      const pull = pulls[p];
      if (Math.abs(pull.amount) < 0.005) continue;
      const n = FACES.find((f) => f.key === pull.face)!.normal;
      const need = Math.min(
        Math.floor(FILL_DENSITY * Math.abs(pull.amount)),
        MAX_FILL - cursor,
      );
      if (need <= 0) continue;

      // Build an orthonormal frame {n, u, v}
      const u = new THREE.Vector3();
      const v = new THREE.Vector3();
      // Pick any axis not parallel to n
      const helper =
        Math.abs(n.x) < 0.9
          ? new THREE.Vector3(1, 0, 0)
          : new THREE.Vector3(0, 1, 0);
      u.crossVectors(n, helper).normalize();
      v.crossVectors(n, u).normalize();

      let s = pull.seed >>> 0;
      const rand = () => {
        s = (s + 0x6d2b79f5) >>> 0;
        let t = s;
        t = Math.imul(t ^ (t >>> 15), t | 1);
        t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
        return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
      };

      for (let i = 0; i < need; i++) {
        // Sample a position ALONG the spike first (favoring the tip), then
        // pick a tangential offset whose max width follows the cone shape of
        // the spike at that height. This avoids the flat disc/circle artifact
        // that happens when many dots end up at along≈0 with full radius.
        const alongT = Math.pow(rand(), 0.7); // bias toward larger values (tip)
        const along = alongT * pull.amount;
        // Cone radius at this height: 0 at the tip, full radius at the base
        const heightFromBase = 1 - alongT;
        const localRadius = pull.radius * heightFromBase;
        const r = Math.sqrt(rand()) * localRadius;
        const theta = rand() * Math.PI * 2;
        const tu = Math.cos(theta) * r;
        const tv = Math.sin(theta) * r;

        const px = pull.pin.x + u.x * tu + v.x * tv + n.x * along;
        const py = pull.pin.y + u.y * tu + v.y * tv + n.y * along;
        const pz = pull.pin.z + u.z * tu + v.z * tv + n.z * along;
        const idx = (cursor + i) * 3;
        arr[idx] = px;
        arr[idx + 1] = py;
        arr[idx + 2] = pz;
      }
      cursor += need;
    }

    fillPointsRef.current.geometry.setDrawRange(0, cursor);
    attr.needsUpdate = true;
  });

  // Floating "+Nmm" label at the pin (only while we own the drag)
  const labelInfo = useMemo(() => {
    if (!isOwningDrag || !dragRef.current) return null;
    const { pin, face } = dragRef.current;
    const localNormal = FACES.find((f) => f.key === face)!.normal;
    const pos = pin.clone().add(localNormal.clone().multiplyScalar(pullAmount + 0.15));
    const mm = Math.round(pullAmount * 100);
    return {
      position: pos.toArray() as [number, number, number],
      mm,
      sign: mm >= 0 ? "+" : "",
    };
  }, [isOwningDrag, pullAmount]);

  // Invisible hit volume slightly larger than the brick so the user can
  // grab the cloud anywhere on its silhouette.
  const hitBody = useMemo(() => computeBody(BASE_SIZE, liveExt), [liveExt]);

  return (
    <group ref={groupRef} position={[0, 0.35, 0]}>
      <points ref={pointsRef} geometry={geometry} material={material} />
      {/* Spike-fill dots: generated per-frame from active + frozen pulls */}
      <points ref={fillPointsRef} geometry={fillGeometry} material={material} />

      {/* Invisible pickable proxy — covers the full brick volume */}
      <mesh
        position={hitBody.offset}
        scale={hitBody.scale}
        onPointerDown={handlePointerDown}
        onPointerOver={(e) => {
          e.stopPropagation();
          if (!dragRef.current) gl.domElement.style.cursor = "grab";
        }}
        onPointerOut={() => {
          if (!dragRef.current) gl.domElement.style.cursor = "default";
        }}
      >
        <boxGeometry args={BASE_SIZE} />
        <meshBasicMaterial transparent opacity={0} depthWrite={false} />
      </mesh>

      {labelInfo && (
        <Html
          position={labelInfo.position}
          center
          distanceFactor={6}
          zIndexRange={[100, 0]}
          style={{ pointerEvents: "none" }}
        >
          <div className="floating-label whitespace-nowrap">
            {labelInfo.sign}
            {labelInfo.mm}mm
          </div>
        </Html>
      )}
    </group>
  );
}
