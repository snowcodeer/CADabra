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
  // Soft white with a subtle clearcoat highlight so motion/rotation reads
  // clearly under the studio environment.
  const baseMat = useMemo(
    () =>
      new THREE.MeshPhysicalMaterial({
        color: "#d9dee7",
        roughness: 0.34,
        metalness: 0.03,
        clearcoat: 0.35,
        clearcoatRoughness: 0.22,
        envMapIntensity: 0.85,
      }),
    [],
  );

  const topMat = useMemo(
    () =>
      new THREE.MeshPhysicalMaterial({
        color: "#edf1f6",
        roughness: 0.26,
        metalness: 0.04,
        clearcoat: 0.5,
        clearcoatRoughness: 0.16,
        envMapIntensity: 1.0,
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
