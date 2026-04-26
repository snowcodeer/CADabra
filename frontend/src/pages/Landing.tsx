import { Suspense, useMemo, useRef } from "react";
import { Link } from "react-router-dom";
import { ArrowRight } from "lucide-react";
import * as THREE from "three";
import { Canvas, useFrame } from "@react-three/fiber";
import { Edges } from "@react-three/drei";

/* ============================================================
   BACKGROUND SCENE
   A fixed, full-viewport R3F canvas behind every section. It
   contains a slowly drifting field of rotating wireframe cubes
   and several point-cloud clusters scattered through 3D space.
   The canvas is pointer-events-none so it never blocks the UI.
   ============================================================ */

interface FloatingCubeProps {
  position: [number, number, number];
  size: number;
  speed: number;
  color: string;
}

/**
 * A single wireframe cube that perpetually rotates on two axes and
 * gently bobs in place. Used dozens of times to fill the background
 * with a sterile, technical "CAD swarm" of primitives.
 */
function FloatingCube({ position, size, speed, color }: FloatingCubeProps) {
  const ref = useRef<THREE.Mesh>(null);
  const startY = position[1];
  useFrame((state, delta) => {
    if (!ref.current) return;
    ref.current.rotation.x += delta * speed;
    ref.current.rotation.y += delta * speed * 0.8;
    // Subtle vertical bob, phased by the cube's X position so the
    // field doesn't move in lockstep.
    ref.current.position.y =
      startY + Math.sin(state.clock.elapsedTime * 0.6 + position[0]) * 0.2;
  });
  return (
    <mesh ref={ref} position={position}>
      <boxGeometry args={[size, size, size]} />
      <meshBasicMaterial transparent opacity={0} depthWrite={false} />
      <Edges threshold={15} color={color} />
    </mesh>
  );
}

interface PointCloudClusterProps {
  position: [number, number, number];
  radius: number;
  count: number;
  color: string;
  size?: number;
  /** When false the cluster is fully static — no per-frame rotation. */
  animate?: boolean;
}

/**
 * A spherical cluster of points (Fibonacci-distributed for an even
 * surface look). Optionally rotates slowly. Multiple of these scattered
 * through the background read as floating "scans" mid-capture.
 *
 * Points are rendered fully opaque with depth-write enabled so they
 * stop flickering when a dense cluster overlaps itself — alpha-blended
 * point sprites cause z-sort glitches that read as visual noise.
 */
function PointCloudCluster({
  position,
  radius,
  count,
  color,
  size = 0.035,
  animate = true,
}: PointCloudClusterProps) {
  const ref = useRef<THREE.Points>(null);
  const positions = useMemo(() => {
    const arr = new Float32Array(count * 3);
    for (let i = 0; i < count; i++) {
      const phi = Math.acos(1 - (2 * (i + 0.5)) / count);
      const theta = Math.PI * (1 + Math.sqrt(5)) * i;
      // Slight radial jitter so the surface looks like a real scan.
      const r = radius * (0.92 + (Math.sin(i * 12.9898) * 0.5 + 0.5) * 0.16);
      arr[i * 3] = r * Math.cos(theta) * Math.sin(phi);
      arr[i * 3 + 1] = r * Math.sin(theta) * Math.sin(phi);
      arr[i * 3 + 2] = r * Math.cos(phi);
    }
    return arr;
  }, [count, radius]);
  useFrame((_, delta) => {
    if (!animate || !ref.current) return;
    ref.current.rotation.y += delta * 0.12;
    ref.current.rotation.x += delta * 0.04;
  });
  return (
    <points ref={ref} position={position}>
      <bufferGeometry>
        <bufferAttribute
          attach="attributes-position"
          count={positions.length / 3}
          array={positions}
          itemSize={3}
        />
      </bufferGeometry>
      <pointsMaterial
        size={size}
        color={color}
        sizeAttenuation
      />
    </points>
  );
}

/** A whole field of cubes + clusters, parented to a slowly drifting group. */
function BackgroundField() {
  const groupRef = useRef<THREE.Group>(null);
  useFrame((state) => {
    if (!groupRef.current) return;
    // Very slow parallax drift — gives the entire field life without
    // distracting from foreground content.
    groupRef.current.rotation.y = state.clock.elapsedTime * 0.02;
  });

  // Deterministic pseudo-random scatter so the field is identical on
  // every render (no hydration jitter, no SSR mismatch).
  const cubes = useMemo<FloatingCubeProps[]>(() => {
    const out: FloatingCubeProps[] = [];
    const seed = (i: number, k: number) =>
      (Math.sin(i * 374.761 + k * 91.123) * 43758.5453) % 1;
    for (let i = 0; i < 28; i++) {
      const x = (seed(i, 1) * 2 - 1) * 14;
      const y = (seed(i, 2) * 2 - 1) * 7;
      const z = -2 - Math.abs(seed(i, 3)) * 14;
      const size = 0.35 + Math.abs(seed(i, 4)) * 0.7;
      const speed = 0.15 + Math.abs(seed(i, 5)) * 0.45;
      // Mostly slate ink, occasional blue accent, very rare green.
      const r = Math.abs(seed(i, 6));
      const color = r > 0.92 ? "#22c55e" : r > 0.7 ? "#2563eb" : "#1f2937";
      out.push({ position: [x, y, z], size, speed, color });
    }
    return out;
  }, []);

  const clusters = useMemo<PointCloudClusterProps[]>(() => {
    const seed = (i: number, k: number) =>
      (Math.sin(i * 921.41 + k * 17.77) * 43758.5453) % 1;
    const out: PointCloudClusterProps[] = [];
    for (let i = 0; i < 6; i++) {
      const x = (seed(i, 1) * 2 - 1) * 12;
      const y = (seed(i, 2) * 2 - 1) * 5;
      const z = -3 - Math.abs(seed(i, 3)) * 10;
      const radius = 0.7 + Math.abs(seed(i, 4)) * 0.9;
      const count = 220;
      const color = i % 3 === 0 ? "#2563eb" : "#475569";
      out.push({ position: [x, y, z], radius, count, color });
    }
    return out;
  }, []);

  return (
    <group ref={groupRef}>
      {cubes.map((c, i) => (
        <FloatingCube key={`c-${i}`} {...c} />
      ))}
      {clusters.map((c, i) => (
        <PointCloudCluster key={`p-${i}`} {...c} />
      ))}
    </group>
  );
}

/** Fixed-position background canvas mounted once for the whole page. */
function BackgroundScene() {
  return (
    <div
      aria-hidden
      className="pointer-events-none fixed inset-0 -z-10 opacity-[0.55]"
    >
      <Canvas
        dpr={[1, 1.75]}
        camera={{ position: [0, 0, 10], fov: 45 }}
        gl={{ antialias: true, alpha: true, powerPreference: "low-power" }}
      >
        <ambientLight intensity={0.8} />
        <Suspense fallback={null}>
          <BackgroundField />
        </Suspense>
      </Canvas>
    </div>
  );
}

/* ============================================================
   HERO SCENE — three primary primitives in their own canvas
   ============================================================ */

/**
 * Hero point cloud companion. A single dense Fibonacci sphere with two
 * subtle satellites — fully static so dots never flicker against each
 * other. The animated background carries the motion; this stays still.
 */
function HeroPointClouds() {
  return (
    <group>
      <PointCloudCluster
        position={[0, 0.1, 0]}
        radius={1.55}
        count={2000}
        color="#0f172a"
        size={0.07}
        animate={false}
      />
      <PointCloudCluster
        position={[1.5, 0.8, -0.6]}
        radius={0.6}
        count={700}
        color="#2563eb"
        size={0.055}
        animate={false}
      />
      <PointCloudCluster
        position={[-1.3, -0.85, -0.3]}
        radius={0.45}
        count={500}
        color="#475569"
        size={0.05}
        animate={false}
      />
    </group>
  );
}

/* ============================================================
   PAGE — single, minimalist, full-viewport hero
   ============================================================ */

const Landing = () => {
  return (
    <main className="relative min-h-screen w-full stage-bg text-foreground">
      {/* Faint blueprint grid — kept whisper thin */}
      <div
        aria-hidden
        className="pointer-events-none fixed inset-0 -z-20 opacity-[0.15]"
        style={{
          backgroundImage:
            "linear-gradient(hsl(220 14% 88% / 0.55) 1px, transparent 1px), linear-gradient(90deg, hsl(220 14% 88% / 0.55) 1px, transparent 1px)",
          backgroundSize: "80px 80px",
          maskImage:
            "radial-gradient(ellipse at 50% 50%, black 0%, transparent 75%)",
          WebkitMaskImage:
            "radial-gradient(ellipse at 50% 50%, black 0%, transparent 75%)",
        }}
      />

      {/* Animated 3D background */}
      <BackgroundScene />

      {/* Top navigation — barely there */}
      <header className="absolute top-0 left-0 right-0 z-30">
        <div className="mx-auto flex max-w-6xl items-center justify-between px-8 py-6">
          <Link to="/" className="animate-fade-in">
            <span className="font-wordmark text-[30px] font-bold tracking-[-0.02em] text-foreground">
              CAD
            </span>
            <span className="font-wordmark text-[30px] font-light italic tracking-[-0.02em] text-foreground/70">
              abra
            </span>
          </Link>
          <Link
            to="/workflow"
            className="text-xs font-medium uppercase tracking-[0.25em] text-muted-foreground transition-colors hover:text-foreground"
          >
            Begin →
          </Link>
        </div>
      </header>

      {/* HERO — single full-viewport composition, perfectly centered */}
      <section className="relative flex min-h-screen items-center justify-center px-6">
        <div className="mx-auto grid w-full max-w-5xl items-center gap-4 lg:grid-cols-2 lg:gap-0">
          {/* LEFT — copy */}
          <div className="z-10 animate-fade-in-up text-left lg:justify-self-end lg:pr-4 lg:text-right">
            <div className="mb-6 inline-flex items-center gap-2 text-[10px] font-medium uppercase tracking-[0.3em] text-muted-foreground">
              Scan → Point Cloud → CAD
            </div>
            <h1 className="font-wordmark text-5xl font-light leading-[1.02] tracking-[-0.02em] text-foreground sm:text-6xl lg:text-7xl">
              Reconstruct
              <br />
              <span className="font-semibold">geometry,</span>
              <br />
              <span className="bg-gradient-to-br from-foreground/90 to-foreground/40 bg-clip-text font-light italic text-transparent">
                to the millimetre.
              </span>
            </h1>
            <div className="mt-10 flex items-center gap-5 lg:justify-end">
              <Link
                to="/workflow"
                className="group inline-flex items-center gap-2 text-sm font-medium text-foreground"
              >
                <span className="relative">
                  Begin
                  <span className="absolute -bottom-1 left-0 h-px w-full bg-foreground/40 transition-all duration-300 group-hover:bg-foreground" />
                </span>
                <ArrowRight
                  className="h-4 w-4 transition-transform duration-300 group-hover:translate-x-1"
                  strokeWidth={1.5}
                />
              </Link>
              <Link
                to="/demo"
                className="text-sm font-light text-muted-foreground transition-colors hover:text-foreground"
              >
                Live demo
              </Link>
            </div>
          </div>

          {/* RIGHT — point clouds */}
          <div className="relative h-[360px] w-full animate-fade-in sm:h-[460px] lg:h-[520px] lg:justify-self-start">
            <Canvas
              dpr={[1.5, 2.5]}
              camera={{ position: [0, 0, 6.5], fov: 38 }}
              gl={{ antialias: true, alpha: true }}
            >
              <ambientLight intensity={0.9} />
              <Suspense fallback={null}>
                <HeroPointClouds />
              </Suspense>
            </Canvas>
          </div>
        </div>

        {/* Scroll cue */}
        <div className="absolute bottom-10 left-1/2 -translate-x-1/2 animate-fade-in">
          <div className="flex flex-col items-center gap-2 text-[10px] font-medium uppercase tracking-[0.3em] text-muted-foreground">
            <span>Scroll</span>
            <span className="h-8 w-px animate-pulse bg-gradient-to-b from-foreground/40 to-transparent" />
          </div>
        </div>
      </section>

      {/* USE CASE — second viewport */}
      <section className="relative flex min-h-screen items-center justify-center px-6 py-24">
        <div className="mx-auto grid w-full max-w-5xl gap-16 lg:grid-cols-12 lg:gap-12">
          {/* Left rail — eyebrow + heading */}
          <div className="lg:col-span-5">
            <div className="mb-6 text-[10px] font-medium uppercase tracking-[0.3em] text-muted-foreground">
              Built for the field
            </div>
            <h2 className="font-wordmark text-4xl font-light leading-[1.05] tracking-[-0.02em] text-foreground sm:text-5xl">
              From a noisy{" "}
              <span className="font-semibold">LiDAR scan</span> to a clean{" "}
              <span className="italic text-foreground/70">parametric model</span>
              , in minutes.
            </h2>
          </div>

          {/* Right rail — explanation + use case */}
          <div className="space-y-10 lg:col-span-7">
            <p className="text-base font-light leading-relaxed text-muted-foreground sm:text-lg">
              CADabra ingests raw point clouds from any 3D scanner and
              reconstructs them into editable, dimensionally accurate CAD
              geometry. No manual remeshing, no painful retopology. Just
              measured surfaces, ready to drop into your engineering pipeline.
            </p>

            {/* Use case card */}
            <div className="rounded-2xl border border-border bg-surface/60 p-8 backdrop-blur-sm">
              <div className="mb-4 flex items-center gap-2 text-[10px] font-medium uppercase tracking-[0.3em] text-muted-foreground">
                <span className="h-1.5 w-1.5 rounded-full bg-highlight" />
                Use case · Industrial retrofit
              </div>
              <p className="text-base font-light leading-relaxed text-foreground/85">
                A factory team scans a legacy hydraulic press with a handheld
                LiDAR. CADabra turns the resulting cloud into a parametric
                solid model overnight, letting engineers design a new safety
                guard that fits the machine to the millimetre, without ever
                taking it offline for measurement.
              </p>
            </div>

            {/* Three quick value props */}
            <div className="grid gap-6 sm:grid-cols-3">
              <ValueProp
                k="±1mm"
                v="Reconstruction tolerance verified against ground truth."
              />
              <ValueProp
                k="Any scanner"
                v="LiDAR, photogrammetry, structured light. All welcome."
              />
              <ValueProp
                k="STEP / IGES"
                v="Export straight into SolidWorks, Fusion, or Onshape."
              />
            </div>

            <div className="pt-2">
              <Link
                to="/workflow"
                className="group inline-flex items-center gap-2 text-sm font-medium text-foreground"
              >
                <span className="relative">
                  Start a reconstruction
                  <span className="absolute -bottom-1 left-0 h-px w-full bg-foreground/40 transition-all duration-300 group-hover:bg-foreground" />
                </span>
                <ArrowRight
                  className="h-4 w-4 transition-transform duration-300 group-hover:translate-x-1"
                  strokeWidth={1.5}
                />
              </Link>
            </div>
          </div>
        </div>
      </section>

      {/* Footer — minimal */}
      <footer className="relative z-20 border-t border-border/50">
        <div className="mx-auto flex max-w-6xl items-center justify-between px-8 py-6 text-[10px] font-medium uppercase tracking-[0.3em] text-muted-foreground">
          <span>© CADabra</span>
          <span>v1.0 · 2026</span>
        </div>
      </footer>
    </main>
  );
};

/** Small key / value block for the use-case section. */
function ValueProp({ k, v }: { k: string; v: string }) {
  return (
    <div>
      <div className="font-wordmark text-xl font-semibold tracking-tight text-foreground">
        {k}
      </div>
      <p className="mt-2 text-xs font-light leading-relaxed text-muted-foreground">
        {v}
      </p>
    </div>
  );
}

export default Landing;
