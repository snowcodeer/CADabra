import { useMemo } from "react";
import * as THREE from "three";
import {
  BASE_SIZE,
  Extrusion,
  FaceKey,
  ZERO_EXTRUSION,
  computeBody,
} from "./extrusion";

interface BrickProps {
  color?: string;
  opacity?: number;
  extrusion?: Extrusion;
  /** Live drag for visual sync */
  activeDrag?: { face: FaceKey; delta: number } | null;
}

/**
 * Clean white CAD-style 4x2 brick. Body scales/translates with extrusion;
 * studs stay anchored to the base footprint and ride the top face.
 */
export function GroundTruthObject({
  color = "#7f8897",
  opacity = 1,
  extrusion = ZERO_EXTRUSION,
  activeDrag = null,
}: BrickProps) {
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

  const material = useMemo(
    () =>
      new THREE.MeshStandardMaterial({
        color,
        roughness: 0.35,
        metalness: 0.08,
        transparent: opacity < 1,
        opacity,
      }),
    [color, opacity],
  );

  const studs = useMemo(() => {
    const [bw, , bd] = BASE_SIZE;
    const cols = 4;
    const rows = 2;
    const stepX = bw / cols;
    const stepZ = bd / rows;
    const r = Math.min(stepX, stepZ) / 2 - 0.05;
    const positions: [number, number, number][] = [];
    for (let c = 0; c < cols; c++) {
      for (let row = 0; row < rows; row++) {
        positions.push([
          -bw / 2 + stepX * (c + 0.5),
          0,
          -bd / 2 + stepZ * (row + 0.5),
        ]);
      }
    }
    return { positions, radius: r };
  }, []);

  const topY = body.offset[1] + body.size[1] / 2;

  return (
    <group position={[0, 0.35, 0]}>
      {/* Body */}
      <group position={body.offset} scale={body.scale}>
        <mesh castShadow receiveShadow material={material}>
          <boxGeometry args={BASE_SIZE} />
        </mesh>
      </group>

      {/* Studs ride the top face */}
      {studs.positions.map((p, i) => (
        <mesh
          key={i}
          position={[p[0], topY + 0.09, p[2]]}
          castShadow
          receiveShadow
          material={material}
        >
          <cylinderGeometry args={[studs.radius, studs.radius, 0.18, 48]} />
        </mesh>
      ))}
    </group>
  );
}
