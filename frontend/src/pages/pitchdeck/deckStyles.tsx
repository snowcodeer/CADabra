import type { ReactNode } from "react";

/** Near-flat backdrop: very light grid */
export function DeckBackdrop() {
  return (
    <div
      aria-hidden
      className="pointer-events-none fixed inset-0 -z-10 opacity-[0.18]"
      style={{
        backgroundImage:
          "linear-gradient(hsl(220 14% 88% / 0.4) 1px, transparent 1px), linear-gradient(90deg, hsl(220 14% 88% / 0.4) 1px, transparent 1px)",
        backgroundSize: "56px 56px",
        maskImage: "radial-gradient(ellipse 80% 60% at 50% 40%, black 0%, transparent 70%)",
        WebkitMaskImage: "radial-gradient(ellipse 80% 60% at 50% 40%, black 0%, transparent 70%)",
      }}
    />
  );
}

export function Eyebrow({ children, className = "" }: { children: ReactNode; className?: string }) {
  return (
    <p className={`font-mono text-[10px] font-medium uppercase tracking-[0.35em] text-muted-foreground ${className}`}>
      {children}
    </p>
  );
}

export function GradText({ children, className = "" }: { children: ReactNode; className?: string }) {
  return (
    <span
      className={`bg-gradient-to-br from-highlight to-blue-800 bg-clip-text text-transparent ${className}`}
    >
      {children}
    </span>
  );
}

/** Section titles on product slides: matches /workflow + /demo (Outfit, tight, semibold) */
export const deckSlideTitleClass =
  "font-outfit text-2xl font-semibold leading-[1.15] tracking-tight text-foreground sm:text-3xl md:text-4xl";

export const glassHudClass =
  "rounded-2xl border border-border/80 bg-surface/90 p-5 shadow-sm backdrop-blur-sm";
export const technicalBorderClass =
  "rounded-2xl border border-foreground/15 bg-surface/95 p-6 shadow-sm";

export function GlassPanel({ children, className = "" }: { children: ReactNode; className?: string }) {
  return <div className={`${glassHudClass} ${className}`}>{children}</div>;
}

export function TechPanel({ children, className = "" }: { children: ReactNode; className?: string }) {
  return (
    <div
      className={`relative rounded-2xl border border-foreground/12 bg-surface p-6 shadow-sm before:pointer-events-none before:absolute before:inset-0 before:rounded-2xl before:border before:border-highlight/10 ${className}`}
    >
      {children}
    </div>
  );
}

export function CardAccentLeft({
  accent = "border-l-highlight",
  children,
  className = "",
}: {
  accent?: string;
  children: ReactNode;
  className?: string;
}) {
  return <div className={`border-l-2 ${accent} pl-4 ${className}`}>{children}</div>;
}
