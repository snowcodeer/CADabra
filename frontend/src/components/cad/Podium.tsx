import { useMemo } from "react";
import * as THREE from "three";

interface PodiumProps {
  position?: [number, number, number];
  scale?: number;
}

/**
 * Clean, minimal white podium platform. Two stacked discs with a thin
 * accent ring between them — soft satin finish (not glossy plastic) so the
 * geometry reads cleanly without busy reflections.
 */
export function Podium({ position = [0, 0, 0], scale = 1 }: PodiumProps) {
  // Soft satin white — slightly off-white so it doesn't blow out against
  // the page background, low specular for a clean look.
  const baseMat = useMemo(
    () =>
      new THREE.MeshStandardMaterial({
        color: "#d9dee7",
        roughness: 0.55,
        metalness: 0.0,
        envMapIntensity: 0.6,
      }),
    [],
  );

  const topMat = useMemo(
    () =>
      new THREE.MeshStandardMaterial({
        color: "#edf1f6",
        roughness: 0.45,
        metalness: 0.0,
        envMapIntensity: 0.7,
      }),
    [],
  );

  // Thin neutral accent ring (no emissive glow — keeps the look clean).
  const ringMat = useMemo(
    () =>
      new THREE.MeshStandardMaterial({
        color: "#8f9aaa",
        roughness: 0.6,
        metalness: 0.0,
      }),
    [],
  );

  return (
    <group position={position} scale={scale}>
      {/* Lower disc */}
      <mesh position={[0, -0.42, 0]} receiveShadow castShadow material={baseMat}>
        <cylinderGeometry args={[1.55, 1.55, 0.25, 128]} />
      </mesh>

      {/* Thin separator ring */}
      <mesh position={[0, -0.27, 0]} castShadow receiveShadow material={ringMat}>
        <cylinderGeometry args={[1.52, 1.52, 0.035, 128]} />
      </mesh>

      {/* Upper disc */}
      <mesh position={[0, -0.1, 0]} receiveShadow castShadow material={topMat}>
        <cylinderGeometry args={[1.4, 1.4, 0.3, 128]} />
      </mesh>
    </group>
  );
}
