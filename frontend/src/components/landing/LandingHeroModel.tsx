import { Suspense, useLayoutEffect, useMemo, useRef } from "react";
import { OrbitControls } from "@react-three/drei";
import { Canvas, useFrame, useLoader } from "@react-three/fiber";
import * as THREE from "three";
import { STLLoader } from "three/addons/loaders/STLLoader.js";

const STL_PATH = "/models/deepcadimg_000035.stl";

/** Full solid at the start of each loop (and after each return from cloud). */
const SOLID_SECONDS = 3;
/** Solid → point cloud (and point cloud → solid): ease crossfade in seconds. */
const SOLID_TO_CLOUD = 1.4;
const CLOUD_TO_SOLID = SOLID_TO_CLOUD;
/** Hold in point-cloud mode before crossfading back to solid. */
const HOLD_CLOUD = 2.8;
/** Hold full solid after cloud→solid before the next dissolve. */
const HOLD_SOLID_AFTER = 2;
const CLOUD_OPACITY = 0.85;
/** Match /demo lego point density feel (~28k); shell + “volume” split below. */
const MAX_CLOUD_POINTS = 28_000;
const SHELL_FRACTION = 0.58;

const HERO_LOOK_DIR = new THREE.Vector3(1, 0.78, 0.95).normalize();
const HERO_CAM_DIST = 2.72;
const MODEL_YAW = 0.5;

const HERO_CAM_INITIAL: [number, number, number] = (() => {
  const p = HERO_LOOK_DIR.clone().multiplyScalar(HERO_CAM_DIST);
  return [p.x, p.y, p.z];
})();

function frameStl(geometry: THREE.BufferGeometry) {
  const g = geometry.clone();
  g.computeBoundingBox();
  const box = g.boundingBox;
  if (!box) {
    g.computeVertexNormals();
    return g;
  }
  const c = new THREE.Vector3();
  const size = new THREE.Vector3();
  box.getCenter(c);
  box.getSize(size);
  const maxD = Math.max(size.x, size.y, size.z, 1e-9);
  const s = 1.75 / maxD;
  g.translate(-c.x, -c.y, -c.z);
  g.scale(s, s, s);
  g.computeBoundingBox();
  g.computeVertexNormals();
  return g;
}

const _va = new THREE.Vector3();
const _vb = new THREE.Vector3();
const _vc = new THREE.Vector3();
const _ab = new THREE.Vector3();
const _ac = new THREE.Vector3();
const _n = new THREE.Vector3();
const _p = new THREE.Vector3();
const _tangent = new THREE.Vector3();
const _gram = new THREE.Vector3();
const _bitangent = new THREE.Vector3();

/**
 * Point cloud like /demo: dense **surface** (area-weighted barycentric samples per triangle, not
 * mesh vertices) plus **sub-surface** scatter along inwards normals so it reads as a solid object,
 * not edge loops.
 */
function buildDemoStylePointCloud(geometry: THREE.BufferGeometry, total: number) {
  const pos = geometry.getAttribute("position");
  if (!pos || pos.count < 3) {
    return new THREE.BufferGeometry();
  }

  const index = geometry.index;
  const triCount = index ? index.count / 3 : pos.count / 3;
  if (triCount < 1) {
    return new THREE.BufferGeometry();
  }

  const areas: number[] = new Array(triCount);
  let totalArea = 0;
  for (let t = 0; t < triCount; t++) {
    const i0 = index ? index.getX(t * 3) : t * 3;
    const i1 = index ? index.getX(t * 3 + 1) : t * 3 + 1;
    const i2 = index ? index.getX(t * 3 + 2) : t * 3 + 2;
    _va.set(pos.getX(i0), pos.getY(i0), pos.getZ(i0));
    _vb.set(pos.getX(i1), pos.getY(i1), pos.getZ(i1));
    _vc.set(pos.getX(i2), pos.getY(i2), pos.getZ(i2));
    _ab.subVectors(_vb, _va);
    _ac.subVectors(_vc, _va);
    const a = 0.5 * _ac.cross(_ab).length();
    const area = a > 1e-20 ? a : 0;
    areas[t] = area;
    totalArea += area;
  }
  if (totalArea < 1e-20) {
    return new THREE.BufferGeometry();
  }

  const cum: number[] = new Array(triCount);
  let acc = 0;
  for (let t = 0; t < triCount; t++) {
    acc += areas[t] / totalArea;
    cum[t] = acc;
  }

  geometry.computeBoundingBox();
  const size = new THREE.Vector3(1, 1, 1);
  geometry.boundingBox?.getSize(size);
  const maxDim = Math.max(size.x, size.y, size.z, 1e-6);
  /** Inward nudge for “interior” dots — same idea as demo’s volume fog, mesh-aware. */
  const depthJitter = maxDim * 0.12;

  const pickTriangle = (r: number) => {
    let lo = 0;
    let hi = triCount - 1;
    while (lo < hi) {
      const mid = (lo + hi) >> 1;
      if (cum[mid] < r) {
        lo = mid + 1;
      } else {
        hi = mid;
      }
    }
    return lo;
  };

  const sampleBaryAndNormal = (tIdx: number, out: THREE.Vector3, normal: THREE.Vector3) => {
    const i0 = index ? index.getX(tIdx * 3) : tIdx * 3;
    const i1 = index ? index.getX(tIdx * 3 + 1) : tIdx * 3 + 1;
    const i2 = index ? index.getX(tIdx * 3 + 2) : tIdx * 3 + 2;
    _va.set(pos.getX(i0), pos.getY(i0), pos.getZ(i0));
    _vb.set(pos.getX(i1), pos.getY(i1), pos.getZ(i1));
    _vc.set(pos.getX(i2), pos.getY(i2), pos.getZ(i2));
    const r1 = Math.random();
    const r2 = Math.random();
    const sr = Math.sqrt(r1);
    const a0 = 1 - sr;
    const a1 = (1 - r2) * sr;
    const a2 = r2 * sr;
    out.set(0, 0, 0);
    out.addScaledVector(_va, a0);
    out.addScaledVector(_vb, a1);
    out.addScaledVector(_vc, a2);
    _ab.subVectors(_vb, _va);
    _ac.subVectors(_vc, _va);
    normal.copy(_ab).cross(_ac);
    if (normal.lengthSq() < 1e-20) {
      normal.set(0, 1, 0);
    } else {
      normal.normalize();
    }
  };

  const nShell = Math.max(0, Math.floor(total * SHELL_FRACTION));
  const nVol = Math.max(0, total - nShell);
  const arr = new Float32Array((nShell + nVol) * 3);
  let w = 0;

  for (let i = 0; i < nShell; i++) {
    const tIdx = pickTriangle(Math.random() * 0.999999 + 1e-9);
    sampleBaryAndNormal(tIdx, _p, _n);
    // Same micro depth jitter as PointCloudObject shell
    const jitter = (Math.random() - 0.5) * maxDim * 0.006;
    _p.addScaledVector(_n, jitter);
    arr[w++] = _p.x;
    arr[w++] = _p.y;
    arr[w++] = _p.z;
  }

  for (let i = 0; i < nVol; i++) {
    const tIdx = pickTriangle(Math.random() * 0.999999 + 1e-9);
    sampleBaryAndNormal(tIdx, _p, _n);
    const inward = Math.random() * depthJitter;
    _p.addScaledVector(_n, -inward);
    if (inward > 1e-6) {
      _gram.set(Math.random() * 2 - 1, Math.random() * 2 - 1, Math.random() * 2 - 1);
      _tangent.copy(_gram).addScaledVector(_n, -_gram.dot(_n));
      if (_tangent.lengthSq() < 1e-12) {
        _tangent.set(1, 0, 0);
        _tangent.addScaledVector(_n, -_tangent.dot(_n));
      }
      if (_tangent.lengthSq() < 1e-12) {
        _tangent.set(0, 1, 0);
        _tangent.addScaledVector(_n, -_tangent.dot(_n));
      }
      _tangent.normalize();
      _bitangent.copy(_n).cross(_tangent).normalize();
      const r0 = (Math.random() - 0.5) * inward * 0.55;
      const r1 = (Math.random() - 0.5) * inward * 0.55;
      _p.addScaledVector(_tangent, r0);
      _p.addScaledVector(_bitangent, r1);
    }
    arr[w++] = _p.x;
    arr[w++] = _p.y;
    arr[w++] = _p.z;
  }

  const g = new THREE.BufferGeometry();
  g.setAttribute("position", new THREE.BufferAttribute(arr, 3));
  return g;
}

const easeInOutCubic = (t: number) =>
  t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;

/** One full loop: solid → to cloud → hold cloud → to solid → hold solid. */
const T_P1 = SOLID_SECONDS;
const T_P2 = T_P1 + SOLID_TO_CLOUD;
const T_P3 = T_P2 + HOLD_CLOUD;
const T_P4 = T_P3 + CLOUD_TO_SOLID;
const HERO_DEMO_CYCLE = T_P4 + HOLD_SOLID_AFTER;

/** Point cloud: max wave while crossfading from solid; idle is subtle vibration. */
const WAVE_DURING_FADE = 1.0;
const WAVE_IDLE = 0.2;
/** Start of the cloud-only hold (gentle), before swell toward solid. */
const WAVE_HOLD_EARLY = 0.3;
/** Peak at end of hold — “sound” builds through the dots just before solid returns. */
const WAVE_HOLD_PRE_SOLID = 1.12;

const PC_VS = /* glsl */ `
uniform float uTime;
uniform float uWaveStrength;
varying float vWave;
varying float vWaveFast;
void main() {
  vec3 pos = position;
  // Traveling + interference — reads like a sound wave through the object
  float a = dot(pos, vec3(1.2, 0.85, 0.5)) * 3.2 - uTime * 2.6;
  float b = dot(pos, vec3(-0.4, 1.1, 0.7)) * 2.7 - uTime * 2.1;
  float c = length(pos) * 2.1 - uTime * 1.5;
  float w = 0.45 * sin(a) + 0.32 * sin(b * 1.3) + 0.23 * sin(c);
  // Higher-frequency “ripples” (stronger with uWaveStrength in JS during cloud hold)
  float d = dot(pos, vec3(0.6, 1.0, 0.35)) * 4.1 - uTime * 3.3;
  float w2 = 0.22 * sin(d) + 0.12 * sin(d * 1.7 - uTime * 1.1);
  float wComb = w + w2 * uWaveStrength;
  vWave = wComb;
  vWaveFast = sin(d);
  // Per-point hash micro-jitter
  float h = fract(sin(dot(pos, vec3(12.7, 37.3, 19.1))) * 43758.5453);
  float j = 0.16 * (h - 0.5) * uWaveStrength;
  vec3 n = normalize(pos + vec3(0.0001));
  float disp = 0.012 * wComb * uWaveStrength + 0.0028 * j;
  pos += n * disp;
  vec4 mv = modelViewMatrix * vec4(pos, 1.0);
  gl_Position = projectionMatrix * mv;
  float dist = -mv.z;
  gl_PointSize = clamp(1.55 * 250.0 / max(dist, 0.45), 1.0, 5.0);
}
`;

const PC_FS = /* glsl */ `
uniform float uOpacity;
uniform float uWaveStrength;
varying float vWave;
varying float vWaveFast;
void main() {
  vec2 p = gl_PointCoord - 0.5;
  if (dot(p, p) > 0.21) discard;
  vec3 base = mix(vec3(0.18, 0.4, 0.92), vec3(0.38, 0.62, 1.0), 0.45);
  // Brighten dots on wave crests so motion reads through the point cloud
  float crest = 0.5 + 0.5 * vWave;
  float tw = 0.5 + 0.5 * vWaveFast;
  float pulse = 1.0 + 0.42 * uWaveStrength * (0.4 * crest * crest + 0.2 * abs(tw));
  vec3 col = min(base * pulse, vec3(1.0));
  gl_FragColor = vec4(col, uOpacity);
}
`;

function StlToPointCloud() {
  const stlLoaded = useLoader(STLLoader, STL_PATH);
  const meshGeo = useMemo(() => frameStl(stlLoaded), [stlLoaded]);
  const pointsGeo = useMemo(() => buildDemoStylePointCloud(meshGeo, MAX_CLOUD_POINTS), [meshGeo]);

  const solidMaterial = useMemo(
    () =>
      new THREE.MeshStandardMaterial({
        color: 0x4a6bc8,
        emissive: 0x2a3f80,
        emissiveIntensity: 0.08,
        metalness: 0.12,
        roughness: 0.45,
        transparent: false,
        opacity: 1,
        depthWrite: true,
        side: THREE.DoubleSide,
      }),
    [],
  );

  const pointMat = useMemo(
    () =>
      new THREE.ShaderMaterial({
        transparent: true,
        depthWrite: false,
        depthTest: true,
        blending: THREE.NormalBlending,
        uniforms: {
          uTime: { value: 0 },
          uOpacity: { value: 0 },
          uWaveStrength: { value: WAVE_IDLE },
        },
        vertexShader: PC_VS,
        fragmentShader: PC_FS,
      }),
    [],
  );

  const meshRef = useRef<THREE.Mesh | null>(null);
  const pointsRef = useRef<THREE.Points | null>(null);
  /** Scene clock can be >0 on first frame (HMR, route timing) — time from our mount. */
  const t0Ref = useRef<number | null>(null);

  useLayoutEffect(() => {
    return () => {
      meshGeo.dispose();
      pointsGeo.dispose();
      pointMat.dispose();
      solidMaterial.dispose();
    };
  }, [meshGeo, pointsGeo, pointMat, solidMaterial]);

  useFrame((state) => {
    const clockT = state.clock.elapsedTime;
    if (t0Ref.current === null) t0Ref.current = clockT;
    const t = clockT - t0Ref.current;
    const c = ((t % HERO_DEMO_CYCLE) + HERO_DEMO_CYCLE) % HERO_DEMO_CYCLE;
    const pmu = pointMat.uniforms;
    pmu.uTime.value = state.clock.elapsedTime;

    const setFullSolid = () => {
      solidMaterial.transparent = false;
      solidMaterial.opacity = 1;
      solidMaterial.depthWrite = true;
      pmu.uOpacity.value = 0;
      pmu.uWaveStrength.value = WAVE_IDLE;
      if (meshRef.current) meshRef.current.visible = true;
      if (pointsRef.current) pointsRef.current.visible = false;
    };

    if (c < T_P1) {
      setFullSolid();
      return;
    }
    if (c < T_P2) {
      const u = THREE.MathUtils.clamp((c - T_P1) / SOLID_TO_CLOUD, 0, 1);
      const s = easeInOutCubic(u);
      solidMaterial.transparent = true;
      solidMaterial.opacity = 1 - s;
      solidMaterial.depthWrite = 1 - s > 0.5;
      pmu.uOpacity.value = CLOUD_OPACITY * s;
      pmu.uWaveStrength.value = THREE.MathUtils.lerp(
        WAVE_DURING_FADE,
        WAVE_IDLE,
        s * s * (2 - s),
      );
      if (meshRef.current) meshRef.current.visible = true;
      if (pointsRef.current) pointsRef.current.visible = true;
      return;
    }
    if (c < T_P3) {
      solidMaterial.transparent = true;
      solidMaterial.opacity = 0;
      if (meshRef.current) meshRef.current.visible = false;
      if (pointsRef.current) pointsRef.current.visible = true;
      pmu.uOpacity.value = CLOUD_OPACITY;
      // Swell: gentle early in hold, stronger ripples and crest pulse toward T_P3 (before solid).
      const holdK = (c - T_P2) / (T_P3 - T_P2);
      const t = 1 - (1 - holdK) ** 2.15;
      pmu.uWaveStrength.value = THREE.MathUtils.lerp(
        WAVE_HOLD_EARLY,
        WAVE_HOLD_PRE_SOLID,
        t,
      );
      return;
    }
    if (c < T_P4) {
      const u = THREE.MathUtils.clamp((c - T_P3) / CLOUD_TO_SOLID, 0, 1);
      const s = easeInOutCubic(u);
      solidMaterial.transparent = true;
      solidMaterial.opacity = s;
      solidMaterial.depthWrite = s > 0.5;
      pmu.uOpacity.value = CLOUD_OPACITY * (1 - s);
      pmu.uWaveStrength.value = THREE.MathUtils.lerp(
        WAVE_DURING_FADE,
        WAVE_IDLE,
        s * s * (2 - s),
      );
      if (meshRef.current) meshRef.current.visible = true;
      if (pointsRef.current) pointsRef.current.visible = true;
      return;
    }
    setFullSolid();
  });

  return (
    <group rotation={[0, MODEL_YAW, 0]}>
      <points
        ref={pointsRef}
        frustumCulled={false}
        geometry={pointsGeo}
        material={pointMat}
        renderOrder={0}
      />
      <mesh
        ref={meshRef}
        geometry={meshGeo}
        material={solidMaterial}
        frustumCulled
        renderOrder={1}
      />
    </group>
  );
}

export function LandingHeroModel() {
  return (
    <div className="relative h-full min-h-[260px] w-full [&_canvas]:block">
      <Canvas
        className="!absolute inset-0 h-full w-full min-h-[260px] cursor-grab active:cursor-grabbing"
        gl={{ antialias: true, alpha: true, powerPreference: "high-performance" }}
        dpr={[1, 2]}
        camera={{ fov: 45, near: 0.01, far: 50, position: HERO_CAM_INITIAL }}
        onCreated={({ gl }) => {
          gl.setClearColor(0, 0);
        }}
      >
        <OrbitControls
          makeDefault
          enableDamping
          dampingFactor={0.08}
          enablePan={false}
          enableZoom={false}
          minPolarAngle={0.15}
          maxPolarAngle={Math.PI - 0.1}
          minDistance={1.2}
          maxDistance={7.5}
          target={[0, 0, 0]}
        />
        <ambientLight intensity={0.72} />
        <directionalLight position={[4.5, 6.2, 4]} intensity={1.0} castShadow={false} />
        <directionalLight position={[-2.2, 1.2, -2.5]} intensity={0.3} color="#c8d4ff" />
        <pointLight position={[1.2, 2, 1.2]} intensity={0.45} />
        <Suspense fallback={null}>
          <StlToPointCloud />
        </Suspense>
      </Canvas>
    </div>
  );
}
