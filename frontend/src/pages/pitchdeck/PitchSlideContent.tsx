import { useState } from "react";
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

/** Skewed row: light vertical wobble, small rotations, enough gap to avoid overlap. */
const RESEARCH_SCATTER: { y: number; r: number }[] = [
  { y: 8, r: -5.5 },
  { y: 0, r: 3.2 },
  { y: 10, r: -2.8 },
  { y: -4, r: 4.5 },
  { y: 6, r: -3.4 },
  { y: -2, r: 2.6 },
  { y: 7, r: -4.1 },
];

export function Pitch06Research() {
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
        <div className="flex min-h-[min(40vh,420px)] w-full items-end justify-center overflow-x-auto overflow-y-visible px-1 py-3 pb-6 sm:min-h-[min(44vh,480px)] sm:px-2">
          <div className="flex w-max max-w-full flex-nowrap items-end justify-center gap-2.5 sm:mx-auto sm:gap-3.5 md:gap-4">
            {RESEARCH_PAPERS.map((p, i) => {
              const s = RESEARCH_SCATTER[i] ?? { y: 0, r: 0 };
              return (
                <a
                  key={p.href}
                  href={p.href}
                  target="_blank"
                  rel="noopener noreferrer"
                  title={p.label}
                  className="research-slap-in group relative z-[1] flex w-[min(4.5rem,19vw)] shrink-0 flex-col overflow-visible rounded-lg border border-foreground/10 bg-white p-0.5 shadow-[0_8px_24px_rgba(15,23,42,0.08),0_1px_4px_rgba(15,23,42,0.04)] transition duration-200 ease-out hover:z-20 hover:shadow-[0_14px_32px_rgba(15,23,42,0.14)] sm:w-24 md:w-28"
                  style={{
                    transform: `translateY(${s.y}px) rotate(${s.r}deg)`,
                    animationDelay: `${0.04 + i * 0.06}s`,
                  }}
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
                  <span className="mt-1.5 line-clamp-2 px-0.5 text-center font-mono text-[0.5rem] leading-tight text-muted-foreground/90 sm:text-[0.55rem]">
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
