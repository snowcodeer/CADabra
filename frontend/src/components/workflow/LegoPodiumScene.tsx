import { Suspense, useMemo, useRef } from "react";
import { Canvas, useFrame } from "@react-three/fiber";
import { Environment } from "@react-three/drei";
import * as THREE from "three";
import { Podium } from "@/components/cad/Podium";
import { BASE_SIZE } from "@/components/cad/extrusion";

/**
 * Auto-rotating point-cloud brick on a podium — a literal, sampled
 * point cloud of a 4×2 lego-style brick body + 8 studs, rendered as
 * dark dots. Sits on the same Podium used in /demo so the visual
 * language is consistent.
 */

const BRICK_W = BASE_SIZE[0];
const BRICK_H = BASE_SIZE[1];
const BRICK_D = BASE_SIZE[2];
const STUD_RADIUS = 0.115;
const STUD_HEIGHT = 0.18;
const STUD_COLS = 4;
const STUD_ROWS = 2;

/** Sample N points uniformly on the surface of a box. */
function sampleBoxSurface(
  w: number,
  h: number,
  d: number,
  count: number,
  rand: () => number,
) {
  const areas = [h * d, h * d, w * d, w * d, w * h, w * h]; // ±x, ±y, ±z
  const total = areas.reduce((a, b) => a + b, 0);
  const cdf = areas.map((a, i) => areas.slice(0, i + 1).reduce((s, x) => s + x, 0) / total);
  const out: number[] = [];
  for (let i = 0; i < count; i++) {
    const r = rand();
    let face = 0;
    while (face < 5 && r > cdf[face]) face++;
    const u = rand() - 0.5;
    const v = rand() - 0.5;
    let x = 0, y = 0, z = 0;
    switch (face) {
      case 0: x =  w / 2; y = u * h; z = v * d; break;
      case 1: x = -w / 2; y = u * h; z = v * d; break;
      case 2: y =  h / 2; x = u * w; z = v * d; break;
      case 3: y = -h / 2; x = u * w; z = v * d; break;
      case 4: z =  d / 2; x = u * w; y = v * h; break;
      case 5: z = -d / 2; x = u * w; y = v * h; break;
    }
    out.push(x, y, z);
  }
  return out;
}

/** Sample N points on the surface of an upright cylinder centered at origin. */
function sampleCylinderSurface(
  radius: number,
  height: number,
  count: number,
  rand: () => number,
  cx = 0,
  cy = 0,
  cz = 0,
) {
  const sideArea = 2 * Math.PI * radius * height;
  const capArea = Math.PI * radius * radius;
  const total = sideArea + 2 * capArea;
  const out: number[] = [];
  for (let i = 0; i < count; i++) {
    const r = rand() * total;
    if (r < sideArea) {
      const theta = rand() * Math.PI * 2;
      const y = (rand() - 0.5) * height;
      out.push(cx + Math.cos(theta) * radius, cy + y, cz + Math.sin(theta) * radius);
    } else {
      // Cap (top or bottom)
      const top = r < sideArea + capArea;
      const rho = Math.sqrt(rand()) * radius;
      const theta = rand() * Math.PI * 2;
      const y = top ? height / 2 : -height / 2;
      out.push(cx + Math.cos(theta) * rho, cy + y, cz + Math.sin(theta) * rho);
    }
  }
  return out;
}

function PointCloudBrick() {
  const ref = useRef<THREE.Group>(null);
  useFrame((_, dt) => {
    if (ref.current) ref.current.rotation.y += dt * 0.45;
  });

  const geometry = useMemo(() => {
    // Deterministic PRNG so the cloud is identical between renders.
    let s = 0xC0FFEE;
    const rand = () => {
      s = (s * 1664525 + 1013904223) >>> 0;
      return s / 0xffffffff;
    };

    const bodyPts = sampleBoxSurface(BRICK_W, BRICK_H, BRICK_D, 4500, rand);

    // Stud positions match the demo brick: 4 along x, 2 along z.
    const studXSpacing = BRICK_W / STUD_COLS;
    const studZSpacing = BRICK_D / STUD_ROWS;
    const studPts: number[] = [];
    for (let cx = 0; cx < STUD_COLS; cx++) {
      for (let cz = 0; cz < STUD_ROWS; cz++) {
        const x = -BRICK_W / 2 + studXSpacing * (cx + 0.5);
        const z = -BRICK_D / 2 + studZSpacing * (cz + 0.5);
        const y = BRICK_H / 2 + STUD_HEIGHT / 2;
        studPts.push(
          ...sampleCylinderSurface(STUD_RADIUS, STUD_HEIGHT, 220, rand, x, y, z),
        );
      }
    }

    const all = new Float32Array(bodyPts.length + studPts.length);
    all.set(bodyPts, 0);
    all.set(studPts, bodyPts.length);

    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.BufferAttribute(all, 3));
    return geo;
  }, []);

  const material = useMemo(
    () =>
      new THREE.PointsMaterial({
        color: "#1f2430",
        size: 0.022,
        sizeAttenuation: true,
        transparent: true,
        opacity: 0.95,
        depthWrite: false,
      }),
    [],
  );

  return (
    <group ref={ref} position={[0, 0.45, 0]}>
      <points geometry={geometry} material={material} />
    </group>
  );
}

export function LegoPodiumScene() {
  return (
    <Canvas
      shadows
      dpr={[1, 2]}
      camera={{ position: [0, 1.3, 6.7], fov: 28 }}
      gl={{ antialias: true, alpha: true }}
    >
      <ambientLight intensity={0.65} />
      <directionalLight
        position={[5, 8, 4]}
        intensity={0.85}
        castShadow
        shadow-mapSize-width={1024}
        shadow-mapSize-height={1024}
        shadow-camera-near={0.5}
        shadow-camera-far={20}
        shadow-camera-left={-6}
        shadow-camera-right={6}
        shadow-camera-top={6}
        shadow-camera-bottom={-6}
        shadow-bias={-0.0004}
        shadow-radius={6}
      />
      <directionalLight position={[-5, 3, -2]} intensity={0.3} />
      <Suspense fallback={null}>
        <Environment preset="studio" environmentIntensity={0.55} />
        <Podium />
        <PointCloudBrick />
        {/* Soft contact shadow plane under the podium */}
        <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, -0.55, 0]} receiveShadow>
          <planeGeometry args={[12, 12]} />
          <shadowMaterial transparent opacity={0.18} />
        </mesh>
      </Suspense>
    </Canvas>
  );
}
