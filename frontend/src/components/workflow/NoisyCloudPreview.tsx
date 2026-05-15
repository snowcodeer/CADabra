import { Suspense, useEffect, useMemo, useRef, useState } from "react";
import * as THREE from "three";
import { Canvas, useFrame } from "@react-three/fiber";
import { STLLoader } from "three/addons/loaders/STLLoader.js";

/**
 * Tiny R3F preview that loads a noisy reconstructed STL and renders
 * it as a slowly-rotating point cloud — exactly the "messy scan"
 * input that the CADabra pipeline starts from. Dropped into the demo
 * cards on /workflow.
 *
 * Why points and not the mesh: the point-cloud read tells the user
 * "this is a noisy scan, not a finished CAD part" at a glance, which
 * is the whole framing of the demo. The triangulated mesh would look
 * too clean.
 */

const POINT_COUNT = 2400;
const FRAME_TARGET = 1.6; // world-units the cloud is normalised to

/** Centre + uniform-scale the geometry so the point cloud fits in a
 *  predictable box regardless of the source STL's units. */
function normaliseGeometry(g: THREE.BufferGeometry): THREE.BufferGeometry {
  const out = g.clone();
  out.computeBoundingBox();
  const bb = out.boundingBox!;
  const center = new THREE.Vector3();
  bb.getCenter(center);
  const size = new THREE.Vector3();
  bb.getSize(size);
  const longest = Math.max(size.x, size.y, size.z) || 1;
  const scale = FRAME_TARGET / longest;
  out.translate(-center.x, -center.y, -center.z);
  out.scale(scale, scale, scale);
  return out;
}

/** Sample N points from triangle vertices (cheap, no area weighting). */
function buildPointPositions(
  geom: THREE.BufferGeometry,
  count: number,
): Float32Array {
  const posAttr = geom.getAttribute("position") as THREE.BufferAttribute;
  if (!posAttr) return new Float32Array(0);
  const total = posAttr.count;
  const out = new Float32Array(count * 3);
  for (let i = 0; i < count; i++) {
    const idx = Math.floor(Math.random() * total);
    out[i * 3 + 0] = posAttr.getX(idx);
    out[i * 3 + 1] = posAttr.getY(idx);
    out[i * 3 + 2] = posAttr.getZ(idx);
  }
  return out;
}

function buildFallbackCloud(count: number): Float32Array {
  const out = new Float32Array(count * 3);
  for (let i = 0; i < count; i++) {
    const phi = Math.acos(1 - (2 * (i + 0.5)) / count);
    const theta = Math.PI * (1 + Math.sqrt(5)) * i;
    const r = 0.72 + Math.sin(theta * 3.1) * 0.06;
    out[i * 3 + 0] = Math.cos(theta) * Math.sin(phi) * r;
    out[i * 3 + 1] = Math.sin(theta) * Math.sin(phi) * r;
    out[i * 3 + 2] = Math.cos(phi) * r;
  }
  return out;
}

function PointCloudFromStl({ src }: { src: string }) {
  const [loaded, setLoaded] = useState<THREE.BufferGeometry | null>(null);
  const [failed, setFailed] = useState(false);

  useEffect(() => {
    let cancelled = false;
    setLoaded(null);
    setFailed(false);
    const loader = new STLLoader();
    loader.load(
      src,
      (geometry) => {
        if (!cancelled) setLoaded(geometry);
      },
      undefined,
      (err) => {
        console.warn(`Preview STL failed for ${src}`, err);
        if (!cancelled) setFailed(true);
      },
    );
    return () => {
      cancelled = true;
    };
  }, [src]);

  const positions = useMemo(() => {
    if (!loaded) return failed ? buildFallbackCloud(POINT_COUNT) : null;
    const normed = normaliseGeometry(loaded);
    return buildPointPositions(normed, POINT_COUNT);
  }, [loaded, failed]);

  const geometry = useMemo(() => {
    if (!positions) return null;
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.BufferAttribute(positions, 3));
    return g;
  }, [positions]);

  useEffect(() => () => geometry?.dispose(), [geometry]);

  const material = useMemo(
    () =>
      new THREE.PointsMaterial({
        color: 0x60a5fa,
        size: 0.02,
        sizeAttenuation: true,
        transparent: true,
        opacity: 0.92,
      }),
    [],
  );

  const ref = useRef<THREE.Points>(null);
  useFrame((_, dt) => {
    if (ref.current) ref.current.rotation.y += dt * 0.4;
  });

  if (!geometry) return null;
  return <points ref={ref} geometry={geometry} material={material} />;
}

export function NoisyCloudPreview({ src }: { src: string }) {
  return (
    <Canvas
      dpr={[1, 1.5]}
      camera={{ position: [0, 0.4, 3], fov: 35 }}
      gl={{ antialias: true, alpha: true, preserveDrawingBuffer: false }}
      frameloop="always"
      style={{ background: "transparent", pointerEvents: "none" }}
    >
      <ambientLight intensity={0.9} />
      <Suspense fallback={null}>
        <PointCloudFromStl src={src} />
      </Suspense>
    </Canvas>
  );
}
