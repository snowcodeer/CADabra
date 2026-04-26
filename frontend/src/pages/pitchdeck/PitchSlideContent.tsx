import { useEffect, useRef, useState, type CSSProperties } from "react";
import { Play } from "lucide-react";
import { deckSlideTitleClass, Eyebrow, GradText } from "./deckStyles";

/* -- Demo -- */
export function Pitch03Demo() {
  return (
    <div className="mx-auto flex h-full w-full max-w-3xl flex-col justify-center px-2 py-4">
      <Eyebrow>Demo</Eyebrow>
      <h2 className={`mt-2 ${deckSlideTitleClass}`}>Product video</h2>
      <p className="mt-2 text-sm text-muted-foreground">Embed when the cut is ready.</p>
      <div
        className="relative mt-8 flex aspect-video w-full min-h-[180px] items-center justify-center rounded-xl border border-dashed border-border/70 bg-surface/50"
        role="region"
        aria-label="Video placeholder"
      >
        <div className="flex flex-col items-center gap-3 p-6 text-center">
          <div className="flex h-12 w-12 items-center justify-center rounded-full border border-border/60 text-muted-foreground">
            <Play className="h-5 w-5" strokeWidth={1.5} />
          </div>
          <p className="text-xs text-muted-foreground">Video, Vimeo, or Mux</p>
        </div>
      </div>
    </div>
  );
}

/* -- TAM: concentric circles (TAM / SAM / SOM) -- */
type TamTier = {
  key: "TAM" | "SAM" | "SOM";
  label: string;
  value: string;
  sub: string;
  size: number;
  ring: string;
  fill: string;
  /** Footnote index (1-based) into TAM_REFERENCES below. */
  cite: number;
};

const TAM_TIERS: TamTier[] = [
  {
    key: "TAM",
    label: "TAM",
    value: "$12B",
    sub: "Global CAD + reverse-engineering software",
    size: 100,
    ring: "border-foreground/20",
    fill: "bg-foreground/[0.03]",
    cite: 1,
  },
  {
    key: "SAM",
    label: "SAM",
    value: "$3.4B",
    sub: "Mechanical engineering teams using parametric CAD",
    size: 66,
    ring: "border-foreground/35",
    fill: "bg-foreground/[0.06]",
    cite: 2,
  },
  {
    key: "SOM",
    label: "SOM",
    value: "$280M",
    sub: "Hardware startups + makers needing fast scan→CAD",
    size: 36,
    ring: "border-highlight/70",
    fill: "bg-highlight/[0.12]",
    cite: 3,
  },
];

const TAM_REFERENCES: { n: number; href: string; label: string }[] = [
  {
    n: 1,
    href: "https://www.mordorintelligence.com/industry-reports/computer-aided-design-cad-market",
    label: "Mordor Intelligence — CAD market size & forecast (2024)",
  },
  {
    n: 2,
    href: "https://www.grandviewresearch.com/industry-analysis/mechanical-cad-market-report",
    label: "Grand View Research — Mechanical CAD segment (2024)",
  },
  {
    n: 3,
    href: "https://www.gartner.com/en/documents/cad-spend-emerging-hardware",
    label: "CADabra estimate · Gartner CAD-spend dataset, hardware-startup cohort",
  },
];

/** Inline numeric citation marker — superscript link to the footer ref. */
function Cite({ n }: { n: number }) {
  return (
    <a
      href={`#tam-ref-${n}`}
      className="ml-0.5 inline-block align-super text-[0.55em] font-semibold text-highlight transition-colors hover:text-foreground"
      aria-label={`Citation ${n}`}
    >
      [{n}]
    </a>
  );
}

export function Pitch04Tam() {
  // Pre-computed coordinates for desktop leader lines, in the SVG's
  // 100×60 viewBox (which is the figure container at md:aspect-[5/3]).
  // Each circle is bottom-anchored, centered at x=30 within a 60×60
  // square in the left half. The "right edge" point is the rightmost
  // point on the circle at the y where the in-circle text sits.
  const LEADER_GEOMETRY = [
    { key: "TAM" as const, x1: 48, y1: 6, x2: 64, y2: 6 },
    { key: "SAM" as const, x1: 42.7, y1: 25, x2: 64, y2: 25 },
    { key: "SOM" as const, x1: 40.8, y1: 49.2, x2: 64, y2: 49.2 },
  ];
  const LABEL_TOP_PCT: Record<TamTier["key"], string> = {
    TAM: "10%",
    SAM: "41.7%",
    SOM: "82%",
  };

  return (
    <div className="mx-auto flex h-full w-full max-w-5xl flex-col px-2 py-4">
      <Eyebrow>Market</Eyebrow>
      <h2 className={`mt-2 ${deckSlideTitleClass}`}>
        A <GradText>$12B</GradText> CAD market.
      </h2>
      <p className="mt-2 max-w-2xl text-sm text-muted-foreground">
        Concentric: total → addressable → obtainable.
      </p>

      <div className="mt-6 flex min-h-0 flex-1 items-center justify-center md:mt-8">
        <div className="relative w-full max-w-[820px] md:aspect-[5/3]">
          {/* ---------------- Mobile layout ---------------- */}
          <div className="md:hidden">
            <div className="relative mx-auto aspect-square w-[min(75vw,340px)]">
              {TAM_TIERS.map((t) => (
                <div
                  key={t.key}
                  className={`absolute bottom-0 left-1/2 flex flex-col items-center justify-start rounded-full border ${t.ring} ${t.fill} backdrop-blur-[1px]`}
                  style={{
                    width: `${t.size}%`,
                    height: `${t.size}%`,
                    transform: "translateX(-50%)",
                    paddingTop: t.key === "TAM" ? "4%" : t.key === "SAM" ? "8%" : "0",
                  }}
                >
                  {t.key !== "SOM" ? (
                    <div className="flex flex-col items-center">
                      <span className="font-mono text-[10px] font-medium uppercase tracking-[0.3em] text-muted-foreground">
                        {t.label}
                      </span>
                      <span className="mt-0.5 font-outfit text-xl font-extrabold tracking-tight text-foreground sm:text-2xl">
                        {t.value}
                        <Cite n={t.cite} />
                      </span>
                    </div>
                  ) : (
                    <div className="flex h-full flex-col items-center justify-center">
                      <span className="font-mono text-[10px] font-medium uppercase tracking-[0.3em] text-highlight">
                        {t.label}
                      </span>
                      <span className="mt-1 font-outfit text-2xl font-extrabold tracking-tight text-foreground sm:text-3xl">
                        {t.value}
                        <Cite n={t.cite} />
                      </span>
                    </div>
                  )}
                </div>
              ))}
            </div>
            <ul className="mt-6 flex flex-col gap-3">
              {TAM_TIERS.map((t) => (
                <li
                  key={t.key}
                  className="border-l-2 border-foreground/15 pl-4 last:border-l-highlight/70"
                >
                  <div className="flex items-baseline gap-2">
                    <span className="font-mono text-[10px] font-medium uppercase tracking-[0.3em] text-muted-foreground">
                      {t.label}
                    </span>
                    <span className="font-outfit text-lg font-bold tracking-tight text-foreground">
                      {t.value}
                      <Cite n={t.cite} />
                    </span>
                  </div>
                  <p className="mt-1 text-xs leading-relaxed text-muted-foreground">{t.sub}</p>
                </li>
              ))}
            </ul>
          </div>

          {/* ---------------- Desktop layout (md+) ---------------- */}
          <div className="hidden md:block">
            {/* Circles container: bottom-anchored 60×60 square in left
                60% of the 5:3 figure (so it's visually a square). */}
            <div className="absolute bottom-0 left-0 h-full" style={{ width: "60%" }}>
              <div className="relative h-full w-full">
                {TAM_TIERS.map((t) => (
                  <div
                    key={t.key}
                    className={`absolute bottom-0 left-1/2 flex flex-col items-center justify-start rounded-full border ${t.ring} ${t.fill} backdrop-blur-[1px]`}
                    style={{
                      width: `${t.size}%`,
                      height: `${t.size}%`,
                      transform: "translateX(-50%)",
                      paddingTop: t.key === "TAM" ? "4%" : t.key === "SAM" ? "8%" : "0",
                    }}
                  >
                    {t.key !== "SOM" ? (
                      <div className="flex flex-col items-center">
                        <span className="font-mono text-[10px] font-medium uppercase tracking-[0.3em] text-muted-foreground">
                          {t.label}
                        </span>
                        <span className="mt-0.5 font-outfit text-xl font-extrabold tracking-tight text-foreground lg:text-2xl">
                          {t.value}
                          <Cite n={t.cite} />
                        </span>
                      </div>
                    ) : (
                      <div className="flex h-full flex-col items-center justify-center">
                        <span className="font-mono text-[10px] font-medium uppercase tracking-[0.3em] text-highlight">
                          {t.label}
                        </span>
                        <span className="mt-1 font-outfit text-2xl font-extrabold tracking-tight text-foreground lg:text-3xl">
                          {t.value}
                          <Cite n={t.cite} />
                        </span>
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </div>

            {/* Leader lines: viewBox 100×60 stretches across the figure
                via preserveAspectRatio="none". Strokes are non-scaling. */}
            <svg
              className="pointer-events-none absolute inset-0 h-full w-full"
              viewBox="0 0 100 60"
              preserveAspectRatio="none"
              aria-hidden
            >
              {LEADER_GEOMETRY.map((g) => (
                <line
                  key={g.key}
                  x1={g.x1}
                  y1={g.y1}
                  x2={g.x2}
                  y2={g.y2}
                  stroke="currentColor"
                  className={g.key === "SOM" ? "text-highlight/70" : "text-foreground/30"}
                  strokeWidth="1"
                  strokeDasharray="2 2"
                  vectorEffect="non-scaling-stroke"
                />
              ))}
            </svg>

            {/* External labels positioned at line endpoints. */}
            {TAM_TIERS.map((t) => (
              <div
                key={t.key}
                className={`absolute -translate-y-1/2 border-l-2 pl-3 ${
                  t.key === "SOM" ? "border-highlight/70" : "border-foreground/20"
                }`}
                style={{ left: "65%", top: LABEL_TOP_PCT[t.key], width: "32%" }}
              >
                <div className="flex items-baseline gap-2">
                  <span className="font-mono text-[10px] font-medium uppercase tracking-[0.3em] text-muted-foreground">
                    {t.label}
                  </span>
                  <span className="font-outfit text-xl font-bold tracking-tight text-foreground lg:text-2xl">
                    {t.value}
                    <Cite n={t.cite} />
                  </span>
                </div>
                <p className="mt-1 text-xs leading-relaxed text-muted-foreground">{t.sub}</p>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* References footer */}
      <ol className="mt-6 flex flex-col gap-1 border-t border-border/40 pt-3 text-[11px] leading-snug text-muted-foreground sm:flex-row sm:flex-wrap sm:gap-x-4">
        {TAM_REFERENCES.map((r) => (
          <li key={r.n} id={`tam-ref-${r.n}`} className="font-mono">
            <span className="text-highlight">[{r.n}]</span>{" "}
            <a
              href={r.href}
              target="_blank"
              rel="noopener noreferrer"
              className="underline decoration-border/60 underline-offset-2 transition hover:text-foreground hover:decoration-foreground/40"
            >
              {r.label}
            </a>
          </li>
        ))}
      </ol>
    </div>
  );
}

/* -- Research: first-page thumbnails in a row, each links out -- */
const RESEARCH_PAPERS: { href: string; image: string; label: string }[] = [
  { href: "https://arxiv.org/abs/2412.14042", image: "/research1.png", label: "CAD-Recode" },
  { href: "https://arxiv.org/abs/2402.17678", image: "/research2.png", label: "CAD-SIGNet" },
  {
    href: "https://openaccess.thecvf.com/content/CVPR2024/papers/Liu_Point2CAD_Reverse_Engineering_CAD_Models_from_3D_Point_Clouds_CVPR_2024_paper.pdf",
    image: "/research3.png",
    label: "Point2CAD (CVPR 2024)",
  },
  { href: "https://arxiv.org/abs/2401.15563", image: "/research4.png", label: "BrepGen" },
  { href: "https://arxiv.org/abs/2409.17106", image: "/research5.png", label: "Text2CAD" },
  { href: "https://arxiv.org/abs/2409.16294", image: "/research6.png", label: "GenCAD" },
  { href: "https://neurips.cc/virtual/2025/loc/san-diego/poster/118942", image: "/research7.png", label: "MiCADangelo (NeurIPS 2025)" },
];

/** Spread-on-a-table: heavy overlap (>50% via negative left margin), big
 *  vertical wobble, and large rotations so they look like papers tossed
 *  on a desk rather than laid out neatly. The first card has no left
 *  margin; every subsequent card eats into the previous one. */
const RESEARCH_SCATTER: { y: number; r: number; z: number }[] = [
  { y: 14, r: -11, z: 4 },
  { y: -6, r: 7, z: 3 },
  { y: 18, r: -5, z: 6 },
  { y: -12, r: 10, z: 2 },
  { y: 8, r: -8, z: 5 },
  { y: -4, r: 5, z: 1 },
  { y: 16, r: -9, z: 7 },
];

const prefersReducedMotion = () =>
  typeof globalThis !== "undefined" &&
  "matchMedia" in globalThis &&
  globalThis.matchMedia("(prefers-reduced-motion: reduce)").matches;

export function Pitch06Research() {
  const papersRef = useRef<HTMLDivElement>(null);
  const [slap, setSlap] = useState(prefersReducedMotion);

  useEffect(() => {
    if (slap) return;
    const el = papersRef.current;
    if (!el) return;
    const io = new IntersectionObserver(
      (entries) => {
        for (const e of entries) {
          if (e.isIntersecting) {
            setSlap(true);
            io.disconnect();
          }
        }
      },
      { threshold: 0.2, rootMargin: "0px 0px -6% 0px" },
    );
    io.observe(el);
    return () => io.disconnect();
  }, [slap]);

  return (
    <div className="mx-auto w-full max-w-6xl px-1 py-2">
      <Eyebrow>Research</Eyebrow>
      <h2 className={`mt-2 ${deckSlideTitleClass}`}>
        Still an <GradText>unsolved</GradText> problem.
      </h2>
      <p className="mt-3 max-w-2xl text-sm leading-relaxed text-muted-foreground">
        Academic progress on CAD from clouds and text, not yet production reverse engineering for real scans.
      </p>
      <div className="mt-4 w-full min-w-0">
        <div className="flex min-h-[min(60vh,540px)] w-full items-center justify-center overflow-x-auto overflow-y-visible px-1 py-8 sm:min-h-[min(68vh,660px)] sm:px-2">
          <div
            ref={papersRef}
            className="flex w-max max-w-full flex-nowrap items-center justify-center sm:mx-auto"
          >
            {RESEARCH_PAPERS.map((p, i) => {
              const s = RESEARCH_SCATTER[i] ?? { y: 0, r: 0, z: 1 };
              // First card: no overlap. Every following card pulls left
              // by ~55% of its own width so each is more than half hidden
              // under its predecessor — papers tossed on a table look.
              const overlapMl =
                i === 0
                  ? ""
                  : "-ml-[5rem] sm:-ml-[6.5rem] md:-ml-[7.5rem] lg:-ml-[8.5rem]";
              const paperStyle: CSSProperties = {
                zIndex: s.z,
                ["--paper-y" as string]: `${s.y}px`,
                ["--paper-r" as string]: `${s.r}deg`,
                ["--slap-delay" as string]: `${0.08 + i * 0.11}s`,
              };
              return (
                <a
                  key={p.href}
                  href={p.href}
                  target="_blank"
                  rel="noopener noreferrer"
                  title={p.label}
                  className={`group relative flex w-[min(9rem,38vw)] shrink-0 flex-col overflow-visible rounded-lg border border-foreground/10 bg-white p-1 shadow-[0_14px_36px_rgba(15,23,42,0.18),0_3px_8px_rgba(15,23,42,0.1)] hover:!z-30 hover:shadow-[0_24px_48px_rgba(15,23,42,0.24)] sm:w-48 md:w-56 lg:w-64 ${overlapMl} ${
                    slap ? "research-slap-in" : "research-slap-pending"
                  }`}
                  style={paperStyle}
                >
                  <div className="relative aspect-[3/4] w-full overflow-hidden rounded-[6px] bg-white">
                    <img
                      src={p.image}
                      alt={`First page: ${p.label}`}
                      className="h-full w-full object-contain object-top transition-transform duration-200 ease-out group-hover:scale-[1.04]"
                      loading={i < 3 ? "eager" : "lazy"}
                      draggable={false}
                    />
                  </div>
                  <span className="mt-2 line-clamp-2 px-1 text-center font-mono text-[11px] leading-tight text-muted-foreground/90 sm:text-xs">
                    {p.label}
                  </span>
                </a>
              );
            })}
          </div>
        </div>
      </div>
      <p className="mt-2 text-center font-mono text-[10px] text-muted-foreground/80 sm:hidden">Tap a first page to open</p>
      <details className="mt-4 border-t border-border/20 pt-2.5 text-center">
        <summary className="cursor-pointer list-none font-mono text-[10px] font-medium tracking-[0.22em] text-muted-foreground/45 transition-colors hover:text-muted-foreground/70 [&::-webkit-details-marker]:hidden">
          All 7 papers
        </summary>
        <ul
          className="mx-auto mt-2.5 max-w-lg columns-1 gap-x-4 gap-y-0.5 text-left text-[10px] text-muted-foreground/75 sm:columns-2"
          role="list"
        >
          {RESEARCH_PAPERS.map((p) => (
            <li key={p.href} className="mb-1 break-inside-avoid">
              <a
                href={p.href}
                target="_blank"
                rel="noopener noreferrer"
                className="font-mono text-muted-foreground/80 underline decoration-border/50 underline-offset-2 transition hover:text-foreground hover:decoration-foreground/40"
              >
                {p.label}
              </a>
            </li>
          ))}
        </ul>
      </details>
    </div>
  );
}

/**
 * Avatars: local files under public/team/, then unavatar fallbacks.
 */
const PITCH_TEAM: {
  name: string;
  linkedin: string;
  photoSources: string[];
  initials: string;
  bio: string[];
}[] = [
  {
    name: "Leyanster Fernandes",
    linkedin: "https://www.linkedin.com/in/leyanster-fernandes5/",
    photoSources: ["/team/leyanster-fernandes.jpg", "https://unavatar.io/linkedin/user:leyanster-fernandes5"],
    initials: "LF",
    bio: ["CS @ RHUL", "10x Hackathon Wins"],
  },
  {
    name: "Natalie Chan",
    linkedin: "https://www.linkedin.com/in/nataliefwc/",
    photoSources: ["/team/natalie-chan.jpg", "https://unavatar.io/linkedin/user:nataliefwc"],
    initials: "NC",
    bio: ["Bioengineering @ Imperial", "9x Hackathon Wins", "Researching CAD space for 6 months."],
  },
  {
    name: "Cosmin C.",
    linkedin: "https://www.linkedin.com/in/cosmincal/",
    photoSources: ["/team/cosmin.jpg", "https://unavatar.io/linkedin/user:cosmincal"],
    initials: "CC",
    bio: ["CS @ RHUL", "10x Hackathon Wins"],
  },
];

function TeamAvatar({ photoSources, initials, name }: { photoSources: string[]; initials: string; name: string }) {
  const [i, setI] = useState(0);
  if (i >= photoSources.length) {
    return (
      <div
        className="flex h-40 w-40 shrink-0 items-center justify-center rounded-full border border-border/50 bg-muted/30 font-outfit text-lg font-medium text-foreground/80"
        aria-hidden
      >
        {initials}
      </div>
    );
  }
  return (
    <img
      src={photoSources[i]}
      alt={name}
      width={160}
      height={160}
      className="h-40 w-40 shrink-0 rounded-full border border-border/50 object-cover"
      loading="lazy"
      onError={() => setI((k) => k + 1)}
    />
  );
}

export function Pitch12Cta() {
  return (
    <div className="mx-auto w-full max-w-4xl px-1 py-2">
      <h2 className={`text-center ${deckSlideTitleClass}`}>Team</h2>
      <div className="mt-10 grid grid-cols-1 gap-10 md:grid-cols-3 md:gap-6">
        {PITCH_TEAM.map((p) => (
          <div
            key={p.linkedin}
            className="flex flex-col items-center rounded-xl border border-border/40 p-5 text-center"
          >
            <TeamAvatar photoSources={p.photoSources} initials={p.initials} name={p.name} />
            <a
              href={p.linkedin}
              target="_blank"
              rel="noopener noreferrer"
              className="mt-4 text-sm font-medium text-foreground transition hover:text-highlight"
            >
              {p.name}
            </a>
            <div className="mt-3 flex flex-col gap-1.5 text-xs leading-relaxed text-muted-foreground">
              {p.bio.map((line) => (
                <p key={`${p.name}-${line}`}>{line}</p>
              ))}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
