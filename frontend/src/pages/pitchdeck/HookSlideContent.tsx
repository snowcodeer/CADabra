import { useMemo, type CSSProperties } from "react";
import { CadabraCadLockup } from "@/components/CadabraWordmark";
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

/** Shrink-wrapped plate behind a single headline line (use inline text inside—no inner `block` or the band goes full width). */
const hookHeadlinePlate =
  "inline-block max-w-full rounded-md stage-bg px-1.5 py-0.5 text-center sm:px-2 sm:py-1 md:px-2.5 md:py-1";

/** Logo chips: white card so brand marks (esp. dark/inverted) read clearly. */
const hookCompetitorPlate =
  "rounded-md border border-border/60 bg-white px-2 py-1 shadow-sm sm:px-2.5 sm:py-1";

/**
 * Real marks: Adam icon + name; Zoo favicon (inverted for white); Leo Framer wordmark.
 */
const HOOK01_COMPETITORS: {
  label: string;
  place: CSSProperties;
  src: string;
  primary?: boolean;
  wordmark?: boolean;
  /** Show next to the mark (e.g. “Adam” beside icon). */
  name?: string;
  /** e.g. Zoo favicon is designed for dark UI—invert for dark-on-white on #fff. */
  invertImg?: boolean;
}[] = [
  {
    label: "ADAM (adam.new)",
    src: "/pitch-deck/logos/adam-180.png",
    name: "Adam",
    place: { top: "3.5%", left: "50%", zIndex: 16, transform: "translateX(-50%) rotate(-1deg)" },
    primary: true,
  },
  {
    label: "Zoo (zoo.dev)",
    src: "/pitch-deck/logos/zoo-wordmark.svg",
    wordmark: true,
    place: { bottom: "4.5%", left: "8%", transform: "rotate(2deg)" },
  },
  {
    label: "Leo AI (getleo.ai)",
    src: "/pitch-deck/logos/leo.svg",
    place: { top: "35%", right: "3.5%", transform: "translateY(-50%) rotate(-2deg)" },
    wordmark: true,
  },
];

/** What people actually type—no “magic prefix”, reads like a chat to the model. */
const HOOK01_PROMPT_WALL: string[] = [
  "make me a 180mm water bottle, body about 64mm across, 1.2mm wall, export step",
  "need an M6 grid of through holes, 40mm on center, 3mm steel",
  "short DN50 flanged pipe, PN16, 200mm long please",
  "ISO enter keyboard plate, 1.5mm, cherry cutouts",
  "simple wine glass, stem 80mm, base should be stable",
  "Lego 2x4 style brick, stud height like real lego",
  "boss for M3 heat set insert, 5.2mm minor diameter",
  "90 degree bracket, two flanges, 2mm 5052",
  "drone frame arm for 5in props, 16x16 M3 stack mount",
  "small carabiner style gate, 8mm body, 6061 is fine",
  "loft a lamp shade profile I can revolve, ~220mm wide",
  "spur gear module 1, 24 teeth, 20 degree pressure",
  "snap fit tabs for a lid, about 0.6mm deflection",
  "standoff M2.5, 8mm tall, for PCB",
  "manifold 3/8 npt, three out ports, internal passages pls",
  "hand wheel about 120mm, 8mm shaft with a flat",
  "08B-1 sprocket 11T, with hub to pilot on a 20mm shaft",
  "vacuum cup 40mm with a G1/4 boss on the side",
  "bottle cage for a road bike, 64mm minimum hole spacing",
  "decorative 6 petal rosette, 2mm fillet, mill from the top",
  "strain relief for a usb cable, tpu, wall around 1mm",
  "injection molding friendly housing, 2 deg draft on sides",
  "optical post 12.7mm, 1/4-20 thread, 150mm tall",
  "hollow 8x8mm tube link, 3mm wall, 90mm long",
  "loft a fairing through these waterline sketches",
  "impeller 7 blades, 120mm od, need a small bull nose",
  "wedge anchor pocket for 12mm rebar, concrete",
  "cam, 45mm total lift, long dwell in the high position",
  "sheet metal spring clip, 0.4mm, 5mm of travel",
  "knurled grip, 1mm pitch, 20mm long",
  "o ring groove for 2mm cord section, 10mm id housing",
  "infill as a lightweight waffle not solid",
  "living hinge, pp, 0.5mm thinnest part",
  "emboss our logo, 0.2mm, keep it moldable",
  "tpms gyroid strut, 2mm cell if you can",
  "tool flange per iso 9409-1, 50mm pattern, 4x M6",
  "housing for a 6002 bearing, h7 on the od",
  "soft jaws, serration about 1.2mm for a 4in vise",
  "small spray nozzle, helix inside, 0.4mm orifice at tip",
  "fpv drone canopy, slot for 19mm camera",
  "prop guard for a 9in prop, 4 mounting holes on a 45mm pcd",
  "clock gear train, escapement runs about 1Hz",
  "M12 kelly, 25mm across the flats, 20mm long",
  "o ring for a 150mm id vacuum port, groove on the lid",
  "corner cube for 20 series 8020, 4 bolt pattern",
  "esd tray, 10 by 4 pockets, 1mm draft in each pocket",
  "110 bcd 5 arm crank spider, standard mtb offset",
  "servo arm 25t spline, 3mm thick hub",
  "shower drain cover, hair catch, 2mm slot grid",
  "wall holster for a type2 ev plug, with cable loop",
  "lab jack, 80x80 top plate, 12mm trapezoidal screw",
  "phone stand, adjustable angle, print in pla",
  "clamp for 12mm round tube, M5 bolt, quick release if possible",
  "replacement hinge for a laptop, left side, 90mm knuckle spacing",
  "ring light mount for a 15mm rail, 1/4 camera thread on top",
  "simple drawer slide bracket for ikea 40cm cabinet",
  "skateboard truck riser, 1/8in thick, 6 hole pattern",
  "router template for 35mm cup hinge, euro style",
  "plant pot with built in saucer, 180mm high, 6mm wall",
  "car phone vent clip, tpu, fits vertical vanes 3-5mm",
  "rc car wishbone, for 4mm ball studs, 85mm long eye to eye",
];

function Hook01PromptBackdrop() {
  const longText = useMemo(() => {
    const run = HOOK01_PROMPT_WALL.join(" · ");
    return Array(18).fill(run).join(" · ");
  }, []);

  return (
    <div
      className="pointer-events-none absolute inset-0 z-[2] select-none overflow-hidden"
      aria-hidden
    >
      <p
        className="h-full w-full origin-center p-3 text-justify font-mono text-[0.5rem] leading-[1.5] text-foreground/20 sm:text-[0.58rem] sm:leading-[1.45] sm:text-foreground/18"
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

      <div
        className="pointer-events-none absolute inset-0 z-[15] select-none"
        aria-hidden
      >
        {HOOK01_COMPETITORS.map(({ label, place, src, primary, wordmark, name, invertImg }) => (
          <div
            key={label + src}
            aria-label={label}
            className={
              primary
                ? `${hookCompetitorPlate} absolute flex w-auto min-w-0 max-w-[min(92vw,15rem)] items-center gap-2 p-1.5 sm:max-w-[16rem] sm:gap-2.5 sm:p-2.5 ${
                    name ? "h-auto min-h-[3.5rem] flex-row" : "h-16 min-w-[3.5rem] max-w-[min(28vw,5.5rem)] justify-center sm:h-[4.5rem] sm:max-w-[6rem]"
                  }`
                : wordmark
                  ? `${hookCompetitorPlate} absolute flex h-11 w-auto min-w-0 max-w-[min(58vw,10.5rem)] items-center justify-center px-1.5 py-1 sm:h-12 sm:max-w-[11rem] sm:px-2`
                  : `${hookCompetitorPlate} absolute flex h-12 w-12 shrink-0 items-center justify-center p-1.5 sm:h-14 sm:w-14 sm:p-2`
            }
            style={place}
          >
            {primary && name ? (
              <>
                <div className="shrink-0 overflow-hidden rounded-md bg-white p-0.5 sm:p-1">
                  <img
                    src={src}
                    alt=""
                    className="h-9 w-9 object-contain sm:h-11 sm:w-11"
                    width={44}
                    height={44}
                    decoding="async"
                    draggable={false}
                    aria-hidden
                  />
                </div>
                <span className="pr-0.5 font-outfit text-lg font-semibold leading-none tracking-tight text-foreground sm:text-xl">
                  {name}
                </span>
              </>
            ) : (
              <img
                src={src}
                alt={label}
                className={
                  (invertImg ? "invert " : "") +
                  (primary
                    ? "h-12 w-12 object-contain sm:h-14 sm:w-14"
                    : wordmark
                      ? "h-8 w-auto max-w-full object-contain object-center sm:h-9"
                      : "h-9 w-9 object-contain sm:h-11 sm:w-11")
                }
                width={primary ? 56 : wordmark ? 120 : 44}
                height={primary ? 56 : wordmark ? 40 : 44}
                decoding="async"
                draggable={false}
              />
            )}
          </div>
        ))}
      </div>

      <div className="relative z-20 flex min-h-dvh w-full flex-col items-center justify-center px-2 py-8 sm:px-5 sm:py-10">
        <div className="max-w-[min(100%,52rem)] px-1 py-1.5 sm:px-2 sm:py-2">
          <div className="relative z-10 mx-auto max-w-full text-center">
            <h1 className="m-0 flex w-full max-w-full flex-col items-center gap-0">
              <span className={`${hookHeadlinePlate} !pb-0`}>
                <span className="font-outfit text-lg font-bold uppercase leading-none tracking-[0.05em] text-foreground [text-rendering:geometricPrecision] sm:text-xl">
                  <span className="whitespace-nowrap">Everyone and their mum is building</span>
                </span>
              </span>
              <span className={`${hookHeadlinePlate} !pt-0`}>
                <span
                  className="font-outfit text-4xl font-extrabold uppercase leading-none tracking-[0.04em] text-highlight sm:text-5xl sm:tracking-[0.05em] md:text-6xl md:tracking-[0.05em] lg:text-[3.2rem] lg:tracking-[0.05em]"
                  style={{ textShadow: "0 0 32px hsl(var(--highlight) / 0.12)" }}
                >
                  text-to-cad
                  <span className="text-foreground/90">.</span>
                </span>
              </span>
            </h1>
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
          <h2 className="hook-stagger-0 m-0 flex w-full max-w-full flex-col items-center gap-2 sm:gap-1.5 md:gap-2">
            <span className={hookHeadlinePlate}>
              <span className="font-outfit text-3xl font-bold uppercase leading-[1.05] tracking-tight text-foreground sm:text-4xl md:text-5xl">
                But how do I model
              </span>
            </span>
            <span className={hookHeadlinePlate}>
              <span className="font-outfit text-3xl font-bold uppercase leading-[1.05] tracking-tight sm:text-4xl md:text-5xl">
                <GradText>this bottle</GradText>
                <span className="text-foreground">?</span>
              </span>
            </span>
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
            <p className="mt-4 max-w-md text-balance text-center text-sm font-medium leading-snug text-muted-foreground sm:text-base">
              A scan is worth a thousand words
            </p>
          </div>

          <div className={`hook-card-right ${hookPanel} flex flex-col`}>
            <p className="mb-4 font-jetbrains text-xs font-medium uppercase tracking-[0.3em] text-muted-foreground">
              Option B
            </p>
            <PointCloud count={300} variant="slide2" />
            <p className="font-outfit mt-4 text-xl font-semibold text-foreground">Scan it.</p>
            <p className="mt-1 text-center text-base leading-relaxed text-muted-foreground">
              Get a noisy point cloud. Now what?
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}

const HOOK3_BOTTLE_CLEAN = "hook-s3-bottle-clean";
const hook3VizFrame = "mx-auto flex h-[220px] w-44 min-h-0 max-w-full shrink-0 items-center justify-center";

function AfterBottleParametric({ className = "" }: { className?: string }) {
  return (
    <svg
      viewBox="0 0 160 200"
      className={className}
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
          <h2 className="hook-s3-header-title m-0 flex w-full max-w-full flex-col items-center gap-2 sm:gap-1.5 md:gap-2">
            <span className={hookHeadlinePlate}>
              <span className="font-outfit text-3xl font-bold uppercase leading-[1.05] tracking-tight text-foreground sm:text-4xl md:text-5xl">
                Mess in.
              </span>
            </span>
            <span className={hookHeadlinePlate}>
              <span className="font-outfit text-3xl font-bold uppercase leading-[1.05] tracking-tight text-foreground sm:text-4xl md:text-5xl">
                <GradText>Editable geometry</GradText> out.
              </span>
            </span>
          </h2>
        </header>

        <div className="mt-8 grid min-h-0 flex-1 grid-cols-1 items-center gap-4 lg:mt-10 lg:grid-cols-11">
          <div className="hook-s3-before flex justify-center lg:col-span-4">
            <div className={`w-full max-w-md ${hookPanel}`}>
              <p className="mb-4 font-jetbrains text-xs font-medium uppercase tracking-[0.3em] text-muted-foreground">
                Before
              </p>
              <div className={hook3VizFrame}>
                <PointCloud count={260} variant="slide3" className="h-full w-full" />
              </div>
              <p className="mt-3 text-center text-base leading-relaxed text-muted-foreground">Noisy point cloud.</p>
            </div>
          </div>

          <div className="hook-s3-arrow flex flex-col items-center justify-center gap-1 py-2 lg:col-span-3">
            <p className="inline-flex flex-nowrap items-baseline justify-center gap-x-0 whitespace-nowrap text-center font-outfit text-2xl font-bold leading-none tracking-[-0.02em] text-foreground sm:text-3xl md:text-4xl">
              <CadabraCadLockup
                cadLetterClassName="font-outfit text-2xl font-bold leading-none tracking-[-0.02em] text-foreground sm:text-3xl md:text-4xl"
              />
              <GradText>abra</GradText>
            </p>
            <svg width="52" height="24" viewBox="0 0 42 20" className="text-highlight" aria-hidden>
              <path
                d="M2 10 L40 10 M32 4 L40 10 L32 16"
                className="stroke-current"
                strokeWidth="2.2"
                strokeLinecap="round"
                strokeLinejoin="round"
                fill="none"
                vectorEffect="non-scaling-stroke"
              />
            </svg>
          </div>

          <div className="hook-s3-after flex justify-center lg:col-span-4">
            <div className={`w-full max-w-md ${hookPanel}`}>
              <p className="mb-4 font-jetbrains text-xs font-medium uppercase tracking-[0.3em] text-highlight">After</p>
              <div className={hook3VizFrame}>
                <AfterBottleParametric className="h-full w-full" />
              </div>
              <p className="mt-3 text-center text-base leading-relaxed text-foreground/90">Editable parametric CAD.</p>
            </div>
          </div>
        </div>

        <p className="hook-s3-footer mt-6 text-balance text-center text-base leading-relaxed sm:text-lg">
          <span className="font-outfit text-2xl font-extrabold uppercase tabular-nums tracking-tight text-highlight sm:text-3xl md:text-4xl">
            472 hours
          </span>
          <span className="text-muted-foreground"> saved per year per engineer.</span>
        </p>
      </div>
    </div>
  );
}
