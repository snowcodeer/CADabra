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

/* -- Research: papers stacked + “slap” in (--tx, --ty, --tr on each card) -- */
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

const RESEARCH_STACK: { x: number; y: number; r: number }[] = [
  { x: -28, y: 40, r: -5.5 },
  { x: 20, y: 28, r: 4.2 },
  { x: -14, y: 24, r: -3.8 },
  { x: 26, y: 36, r: 3.5 },
  { x: -20, y: 22, r: -4.5 },
  { x: 16, y: 30, r: 2.8 },
  { x: -8, y: 20, r: -2.2 },
];

export function Pitch06Research() {
  const [visibleCount, setVisibleCount] = useState(0);
  const [hasStarted, setHasStarted] = useState(false);
  const rootRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const el = rootRef.current;
    if (!el || hasStarted) return;

    const observer = new IntersectionObserver(
      (entries) => {
        const entry = entries[0];
        if (entry?.isIntersecting) {
          setHasStarted(true);
          setVisibleCount(1);
          observer.disconnect();
        }
      },
      { threshold: 0.55 },
    );

    observer.observe(el);
    return () => observer.disconnect();
  }, [hasStarted]);

  useEffect(() => {
    if (!hasStarted) return;
    const id = window.setInterval(() => {
      setVisibleCount((n) => {
        if (n >= 7) {
          window.clearInterval(id);
          return 7;
        }
        return n + 1;
      });
    }, 1000);
    return () => window.clearInterval(id);
  }, [hasStarted]);

  return (
    <div ref={rootRef} className="mx-auto w-full max-w-5xl px-1 py-2">
      <Eyebrow>Research</Eyebrow>
      <h2 className={`mt-2 ${deckSlideTitleClass}`}>
        Still an <GradText>unsolved</GradText> problem.
      </h2>
      <p className="mt-3 max-w-2xl text-sm leading-relaxed text-muted-foreground">
        Academic progress on CAD from clouds and text, not yet production reverse engineering for real scans.
      </p>
      <div className="relative mx-auto mt-6 min-h-[min(56vh,560px)] w-full max-w-4xl sm:min-h-[min(58vh,600px)]">
        {RESEARCH_PAPERS.map((p, i) => {
          if (i >= visibleCount) return null;
          const s = RESEARCH_STACK[i] ?? { x: 0, y: 0, r: 0 };
          /** Final pose from first paint; avoids % translate reflow when image height resolves. */
          const placeStyle: CSSProperties = {
            zIndex: i + 1,
            transform: `translate(calc(-50% + ${s.x}px), calc(-50% + ${s.y}px)) rotate(${s.r}deg)`,
          };
          return (
            <a
              key={p.href}
              href={p.href}
              target="_blank"
              rel="noopener noreferrer"
              title={p.label}
              className="research-slap-in group absolute left-1/2 top-1/2 w-[min(400px,92vw)] max-w-[24rem] origin-center overflow-hidden rounded-lg border border-foreground/10 bg-white p-0.5 shadow-[0_14px_40px_rgba(15,23,42,0.12),0_2px_8px_rgba(15,23,42,0.06)] transition-shadow duration-200 ease-out [will-change:opacity] hover:z-[60] hover:shadow-[0_20px_44px_rgba(15,23,42,0.16)] sm:w-[min(420px,90vw)] sm:max-w-[26rem]"
              style={placeStyle}
            >
              <div className="max-h-[min(52vh,520px)] w-full max-w-full origin-center overflow-hidden rounded-[6px] transition-transform duration-200 ease-out group-hover:scale-[1.04]">
                <img
                  src={p.image}
                  alt={`First page: ${p.label}`}
                  className="h-auto w-full object-contain object-top"
                  loading={i < 3 ? "eager" : "lazy"}
                  draggable={false}
                />
              </div>
            </a>
          );
        })}
      </div>
      <p className="mt-2 text-center font-mono text-[10px] text-muted-foreground/80 sm:hidden">Tap a page to open</p>
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
