import { useMemo, type CSSProperties } from "react";
import { GradText, glassHudClass } from "./deckStyles";

const hookRand = (i: number) => {
  const seed = Math.sin(i * 9301 + 49297) * 233280;
  return seed - Math.floor(seed);
};

function RulerIllustration() {
  return (
    <svg viewBox="0 0 80 180" className="mx-auto h-[180px] w-[80px] shrink-0" aria-hidden>
      <rect
        x="20"
        y="10"
        width="14"
        height="160"
        fill="hsl(var(--muted) / 0.35)"
        stroke="hsl(var(--border))"
        strokeWidth="1"
      />
      {Array.from({ length: 16 }).map((_, i) => {
        const y = 18 + i * 10;
        const long = i % 2 === 0;
        return (
          <line
            key={i}
            x1="20"
            y1={y}
            x2={long ? 30 : 26}
            y2={y}
            stroke="hsl(var(--muted-foreground) / 0.55)"
            strokeWidth="0.8"
          />
        );
      })}
      <line
        x1="40"
        y1="20"
        x2="40"
        y2="170"
        stroke="hsl(var(--muted-foreground) / 0.5)"
        strokeWidth="0.6"
        strokeDasharray="3 3"
      />
      <text x="46" y="40" fill="hsl(var(--muted-foreground))" fontSize="9" className="font-jetbrains">
        62 mm
      </text>
      <text x="46" y="100" fill="hsl(var(--muted-foreground))" fontSize="9" className="font-jetbrains">
        ?
      </text>
      <text x="46" y="160" fill="hsl(var(--muted-foreground))" fontSize="9" className="font-jetbrains">
        178 mm
      </text>
    </svg>
  );
}

function pointCloudR(variant: "slide2" | "slide3", i: number) {
  return variant === "slide2" ? 0.6 + hookRand(i + 11) * 0.6 : 0.7 + hookRand(i + 11) * 0.6;
}

function PointCloud({ count, variant, className = "h-auto w-[112px] shrink-0" }: { count: number; variant: "slide2" | "slide3"; className?: string }) {
  const circles = useMemo(() => {
    return Array.from({ length: count }, (_, i) => {
      const t = i / count;
      const baseY = 10 + t * 160;
      const neckRadius =
        baseY < 32 ? 10 : baseY < 50 ? 10 + (baseY - 32) * 1.4 : 32;
      const noise = (hookRand(i) - 0.5) * 8;
      const sideJitter = (hookRand(i + 5) - 0.5) * 5;
      const angle = hookRand(i * 3) * 2 * Math.PI;
      const radius = neckRadius + noise;
      const cx = 40 + Math.cos(angle) * radius + sideJitter;
      const cy = baseY + (hookRand(i + 7) - 0.5) * 4;
      const r = pointCloudR(variant, i);
      const o = 0.4 + hookRand(i + 13) * 0.5;
      return { cx, cy, r, o, k: i };
    });
  }, [count, variant]);
  return (
    <svg viewBox="0 0 80 180" className={className} aria-hidden>
      {circles.map(({ cx, cy, r, o, k }) => (
        <circle
          key={k}
          cx={cx}
          cy={cy}
          r={r}
          className="fill-highlight"
          opacity={o}
        />
      ))}
    </svg>
  );
}

const hookPanel =
  `flex h-full min-h-[320px] flex-col items-center justify-center ${glassHudClass} p-6`;

const HOOK01_COMPETITORS = ["ADAM CAD", "Spline AI", "Zoo.dev", "Leo AI"] as const;

const HOOK01_PROMPT_WALL: string[] = [
  "TEXT-TO-CAD: 180 mm bottle, 64 mm body Ø, 1.2 mm wall, STEP out",
  "TEXT TO CAD: M6 through-hole grid 40mm pitch, 3mm steel",
  "TEXT-TO-CAD: flanged pipe DN50, PN16, length 200mm",
  "TEXT TO CAD: keyboard plate ISO, 1.5mm, switch cutouts",
  "TEXT-TO-CAD: parametric wine glass, stem 80mm, stable base",
  "TEXT TO CAD: Lego-compatible brick 2×4, stud height 1.7mm",
  "TEXT-TO-CAD: heat-set insert boss, M3, 5.2mm minor Ø",
  "TEXT TO CAD: sheet bracket, 90° fold, 2mm AL5052",
  "TEXT-TO-CAD: drone arm 5\" props, 16×16 M3 mount",
  "TEXT TO CAD: carabiner gate, 8mm body, 6061",
  "TEXT-TO-CAD: lamp shade loft 220mm, revolve+shell",
  "TEXT TO CAD: gear module 1.0, 24T, 20° pressure angle",
  "TEXT-TO-CAD: enclosure snap-fit clips, 0.6mm deflection",
  "TEXT TO CAD: PCB standoff, M2.5, 8mm standoff h",
  "TEXT-TO-CAD: manifold 3/8 NPT, internal passages",
  "TEXT TO CAD: handwheel Ø120, 8mm shaft D-flat",
  "TEXT-TO-CAD: sprocket 08B-1, 11 teeth, hub pilot",
  "TEXT TO CAD: vacuum cup 40mm, G1/4 thread boss",
  "TEXT-TO-CAD: bicycle bottle cage, 64mm min spacing",
  "TEXT TO CAD: rosette 6-fold, 2mm fillet, mill from top",
  "TEXT-TO-CAD: cable strain relief, TPU 95A, wall 1.0mm",
  "TEXT-TO-CAD: injection mold parting line, draft 2° per side",
  "TEXT-TO-CAD: optical bench post 12.7mm, 1/4-20",
  "TEXT-TO-CAD: robot link, hollow 8×8mm tube, 3mm wall",
  "TEXT-TO-CAD: ship hull fairing, loft through waterlines",
  "TEXT-TO-CAD: impeller 7 blades, 120mm, bull nose hub",
  "TEXT-TO-CAD: concrete anchor wedge, 12mm rebar",
  "TEXT-TO-CAD: cam profile 45mm lift, dwell 60°",
  "TEXT-TO-CAD: spring clip 0.4mm, 5mm travel",
  "TEXT-TO-CAD: knurl 1.0mm pitch, 20mm long grip",
  "TEXT-TO-CAD: o-ring groove 2.0 CS, 10mm ID housing",
  "TEXT-TO-CAD: waffle grid infill, 20% visual density",
  "TEXT-TO-CAD: living hinge, PP 0.5mm web",
  "TEXT-TO-CAD: emboss logo 0.2mm, draft friendly",
  "TEXT-TO-CAD: lattice strut, TPMS gyroid, 2mm unit cell",
  "TEXT-TO-CAD: robot EE coupling ISO 9409-1-50-4-M6",
  "TEXT-TO-CAD: bearing housing 6002-2RS, H7 fit",
  "TEXT-TO-CAD: vise soft jaw, serrated 1.2mm pitch",
  "TEXT-TO-CAD: spray nozzle, internal helix, 0.4mm orifice",
  "TEXT-TO-CAD: drone canopy, 2mm FPV cam slot",
  "TEXT-TO-CAD: ship prop guard, 9\" disc, 4 mounting holes",
  "TEXT-TO-CAD: clock gear train, 1Hz escapement, brass",
  "TEXT-TO-CAD: kelly knob M12, 25mm across flats",
  "TEXT-TO-CAD: vacuum chamber O-ring 150mm, radial seal",
  "TEXT-TO-CAD: 80/20 t-slot, 20 series, 4-hole corner cube",
  "TEXT-TO-CAD: ESD tray pockets, 10×4 cells, 1mm draft",
  "TEXT-TO-CAD: bicycle crank spider 110BCD, 5-arm",
  "TEXT-TO-CAD: servo horn 25T, 3mm hub depth",
  "TEXT-TO-CAD: shower drain hair trap, 2mm slot pattern",
  "TEXT-TO-CAD: EV charging holster, Type 2, cable relief",
  "TEXT-TO-CAD: lab jack platform 80×80, 12mm leadscrew",
];

function Hook01PromptBackdrop() {
  const longText = useMemo(() => {
    const run = HOOK01_PROMPT_WALL.join(" · ");
    return Array(18).fill(run).join(" · ");
  }, []);

  const maskStyle = useMemo((): CSSProperties => {
    // Large, sharp-edged “hole” so prompt copy never peeks into the title zone; headline sits on a solid layer above.
    const m =
      "radial-gradient(ellipse min(96vw, 38rem) min(62dvh, 32rem) at 50% 50%, #000 0% 50%, #fff 50.2%)";
    return {
      WebkitMaskImage: m,
      maskImage: m,
    };
  }, []);

  return (
    <div
      className="pointer-events-none absolute inset-0 z-[2] select-none overflow-hidden"
      style={maskStyle}
      aria-hidden
    >
      <p
        className="h-full w-full origin-center p-3 text-justify font-mono text-[0.45rem] leading-[1.55] text-muted-foreground/18 sm:text-[0.52rem] sm:leading-[1.5] sm:text-muted-foreground/16"
        style={{ wordBreak: "break-word", hyphens: "auto" }}
      >
        {longText}
      </p>
    </div>
  );
}

export function Hook01Bam() {
  return (
    <div className="relative left-1/2 flex min-h-dvh w-screen max-w-[100vw] -translate-x-1/2 overflow-hidden stage-bg">
      <Hook01PromptBackdrop />
      <div
        className="pointer-events-none absolute inset-0 z-[1] hook-grid-48 opacity-[0.2]"
        aria-hidden
      />

      <div className="relative z-20 flex min-h-dvh w-full flex-col items-center justify-center px-3 py-10 sm:px-6 sm:py-12">
        <div className="max-w-[min(100%,52rem)] rounded-[1.75rem] border border-border/50 bg-background px-5 py-6 shadow-sm sm:px-8 sm:py-8 md:px-10 md:py-9">
          <div className="mb-3 flex w-full max-w-2xl flex-wrap items-center justify-center gap-x-4 gap-y-1 sm:mb-4 sm:gap-x-6">
            {HOOK01_COMPETITORS.slice(0, 2).map((name) => (
              <span
                key={name}
                className="font-mono text-[0.6rem] font-medium uppercase tracking-[0.2em] text-foreground/38 sm:text-[0.65rem] sm:tracking-[0.22em]"
              >
                {name}
              </span>
            ))}
          </div>

          <h1 className="max-w-[min(100%,52rem)] text-center">
            <span className="block font-outfit text-2xl font-bold uppercase leading-[1.1] tracking-[0.06em] text-foreground sm:text-3xl sm:leading-tight sm:tracking-[0.05em] md:text-4xl md:tracking-[0.045em] lg:text-5xl">
              Everyone and their mum is building
            </span>
            <span
              className="mt-2 block font-outfit text-3xl font-extrabold uppercase leading-[1.02] tracking-[0.04em] text-highlight sm:mt-2.5 sm:text-4xl sm:tracking-[0.05em] md:mt-3 md:text-5xl md:leading-none md:tracking-[0.05em] lg:mt-3.5 lg:text-[2.9rem] lg:leading-none"
              style={{ textShadow: "0 0 32px hsl(var(--highlight) / 0.12)" }}
            >
              text-to-cad
              <span className="text-foreground/90">.</span>
            </span>
          </h1>

          <div className="mt-3 flex w-full max-w-2xl flex-wrap items-center justify-center gap-x-4 gap-y-1 sm:mt-4 sm:gap-x-6">
            {HOOK01_COMPETITORS.slice(2, 4).map((name) => (
              <span
                key={name}
                className="font-mono text-[0.6rem] font-medium uppercase tracking-[0.2em] text-foreground/38 sm:text-[0.65rem] sm:tracking-[0.22em]"
              >
                {name}
              </span>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}

const HOOK2_BOTTLE_GRAD = "hook-s2-bottle-fill";
/** Isolated: single slide 2 in deck, stable ASCII id (useId() can yield chars that break SVG url(#id) resolution). */
function Hook02BottleSvg() {
  return (
    <div className="hook-bottle-in flex flex-col items-center">
      <svg
        viewBox="0 0 80 180"
        className="mx-auto block h-[252px] w-[min(100%,112px)] shrink-0"
        aria-hidden
      >
        <defs>
          <linearGradient id={HOOK2_BOTTLE_GRAD} x1="0" y1="0" x2="1" y2="0">
            <stop offset="0%" stopColor="hsl(220 32% 18%)" />
            <stop offset="50%" stopColor="hsl(var(--highlight))" stopOpacity="0.32" />
            <stop offset="100%" stopColor="hsl(220 40% 12%)" />
          </linearGradient>
        </defs>
        <path
          d="M30 6 L50 6 L50 28 Q50 34 54 38 Q66 50 66 68 L66 162 Q66 174 54 174 L26 174 Q14 174 14 162 L14 68 Q14 50 26 38 Q30 34 30 28 Z"
          fill={`url(#${HOOK2_BOTTLE_GRAD})`}
          className="stroke-highlight"
          strokeWidth="1.5"
        />
        <line x1="30" y1="6" x2="50" y2="6" className="stroke-highlight" strokeWidth="2" />
      </svg>
    </div>
  );
}

export function Hook02Bottle() {
  return (
    <div className="relative left-1/2 flex h-full min-h-dvh w-screen max-w-[100vw] -translate-x-1/2 flex-col overflow-hidden stage-bg">
      <div className="hook-grid-48 pointer-events-none absolute inset-0" />
      <div className="relative z-[1] mx-auto flex w-full min-h-dvh max-w-6xl flex-1 flex-col px-4 py-3 sm:px-6 sm:py-5 md:py-6">
        <header className="text-center">
          <p className="hook-stagger-0 font-jetbrains text-sm font-medium uppercase tracking-[0.4em] text-highlight">
            BAM. Next.
          </p>
          <h2 className="hook-stagger-1 font-outfit mt-4 text-4xl font-bold uppercase leading-[1.05] tracking-tight text-foreground sm:text-5xl md:text-6xl">
            But how do I model <GradText>this bottle</GradText>?
          </h2>
        </header>

        <div className="mt-8 grid min-h-0 flex-1 grid-cols-1 items-stretch gap-6 md:mt-10 md:grid-cols-3 md:gap-8">
          <div className={`hook-card-left ${hookPanel} flex flex-col`}>
            <p className="mb-4 font-jetbrains text-xs font-medium uppercase tracking-[0.3em] text-muted-foreground">
              Option A
            </p>
            <RulerIllustration />
            <p className="font-outfit mt-4 text-xl font-semibold text-foreground">Measure with a ruler.</p>
            <p className="mt-1 text-center text-base leading-relaxed text-muted-foreground">
              Then redo every dimension by hand in CAD.
            </p>
          </div>

          <div className="flex min-h-[280px] flex-col items-center justify-center">
            <Hook02BottleSvg />
            <p className="mt-4 font-jetbrains text-xs font-medium uppercase tracking-[0.3em] text-muted-foreground">
              One bottle.
            </p>
          </div>

          <div className={`hook-card-right ${hookPanel} flex flex-col`}>
            <p className="mb-4 font-jetbrains text-xs font-medium uppercase tracking-[0.3em] text-muted-foreground">
              Option B
            </p>
            <PointCloud count={220} variant="slide2" />
            <p className="font-outfit mt-4 text-xl font-semibold text-foreground">Scan it.</p>
            <p className="mt-1 text-center text-base leading-relaxed text-muted-foreground">
              Get a noisy point cloud. Now what?
            </p>
          </div>
        </div>

        <p className="hook-footer-stuck mt-6 text-center text-lg italic leading-relaxed text-muted-foreground">
          Either way, I&apos;m stuck.
        </p>
      </div>
    </div>
  );
}

const HOOK3_BOTTLE_CLEAN = "hook-s3-bottle-clean";
function AfterBottleParametric() {
  return (
    <svg
      viewBox="0 0 160 200"
      className="mx-auto h-[220px] w-full max-w-[176px] shrink-0"
      aria-hidden
    >
      <defs>
        <linearGradient id={HOOK3_BOTTLE_CLEAN} x1="0" y1="0" x2="1" y2="0">
          <stop offset="0%" stopColor="hsl(220 40% 12%)" />
          <stop offset="50%" stopColor="hsl(var(--highlight))" stopOpacity="0.42" />
          <stop offset="100%" stopColor="hsl(220 40% 12%)" />
        </linearGradient>
      </defs>
      <path
        d="M52 14 L92 14 L92 36 Q92 42 96 46 Q104 56 104 76 L104 174 Q104 186 92 186 L52 186 Q40 186 40 174 L40 76 Q40 56 48 46 Q52 42 52 36 Z"
        fill={`url(#${HOOK3_BOTTLE_CLEAN})`}
        className="stroke-highlight"
        strokeWidth="1.5"
      />
      <line x1="52" y1="14" x2="92" y2="14" className="stroke-highlight" strokeWidth="2" />
    </svg>
  );
}

export function Hook03Cadabra() {
  return (
    <div className="relative left-1/2 flex h-full min-h-dvh w-screen max-w-[100vw] -translate-x-1/2 flex-col overflow-hidden stage-bg">
      <div className="hook-grid-48 pointer-events-none absolute inset-0" />
      <div className="relative z-[1] mx-auto flex w-full min-h-dvh max-w-7xl flex-1 flex-col px-4 py-3 sm:px-6 sm:py-5 md:py-6">
        <header className="text-center">
          <p className="hook-s3-header-eyebrow font-jetbrains text-sm font-medium uppercase tracking-[0.4em] text-highlight">
            ABRA · CAD · ABRA
          </p>
          <h2 className="hook-s3-header-title font-outfit mt-4 text-4xl font-bold uppercase leading-[1.05] tracking-tight text-foreground sm:text-5xl md:text-6xl">
            Mess in. <GradText>Editable geometry</GradText> out.
          </h2>
        </header>

        <div className="mt-8 grid min-h-0 flex-1 grid-cols-1 items-center gap-4 lg:mt-10 lg:grid-cols-11">
          <div className="hook-s3-before flex justify-center lg:col-span-5">
            <div className={`w-full max-w-md ${hookPanel}`}>
              <p className="mb-4 font-jetbrains text-xs font-medium uppercase tracking-[0.3em] text-muted-foreground">
                Before
              </p>
              <div className="flex w-[128px] max-w-full justify-center">
                <div className="w-[128px]">
                  <PointCloud count={260} variant="slide3" className="h-auto w-full shrink-0" />
                </div>
              </div>
              <p className="mt-3 text-center text-base leading-relaxed text-muted-foreground">Noisy point cloud.</p>
            </div>
          </div>

          <div className="hook-s3-arrow flex flex-col items-center justify-center gap-2 py-2 lg:col-span-1">
            <p className="font-jetbrains text-xs font-medium uppercase tracking-[0.3em] text-highlight">CADabra</p>
            <svg width="42" height="20" viewBox="0 0 42 20" className="text-highlight" aria-hidden>
              <path
                d="M2 10 L40 10 M32 4 L40 10 L32 16"
                className="stroke-current"
                strokeWidth="2"
                strokeLinecap="round"
                strokeLinejoin="round"
                fill="none"
              />
            </svg>
          </div>

          <div className="hook-s3-after flex justify-center lg:col-span-5">
            <div className={`w-full max-w-md ${hookPanel}`}>
              <p className="mb-4 font-jetbrains text-xs font-medium uppercase tracking-[0.3em] text-highlight">After</p>
              <div className="flex justify-center">
                <AfterBottleParametric />
              </div>
              <p className="mt-3 text-center text-base leading-relaxed text-foreground/90">Editable parametric CAD.</p>
            </div>
          </div>
        </div>

        <p className="hook-s3-footer mt-5 text-center text-lg leading-relaxed text-muted-foreground">
          472 hours saved per year per engineer.
        </p>
      </div>
    </div>
  );
}
