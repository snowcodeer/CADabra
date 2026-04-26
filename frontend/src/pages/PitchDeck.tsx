import { useCallback, useEffect, useRef, useState } from "react";
import { Link } from "react-router-dom";
import { ArrowLeft, ChevronLeft, ChevronRight } from "lucide-react";
import { DeckBackdrop } from "./pitchdeck/deckStyles";
import { Pitch03Demo, Pitch06Research, Pitch12Cta } from "./pitchdeck/PitchSlideContent";
import { Hook01Bam, Hook02Bottle, Hook03Cadabra } from "./pitchdeck/HookSlideContent";

/** Hooks (3) + demo, research, team. */
const SLIDES = [Hook01Bam, Hook02Bottle, Hook03Cadabra, Pitch03Demo, Pitch06Research, Pitch12Cta] as const;

const TOTAL = SLIDES.length;

const PitchDeck = () => {
  const containerRef = useRef<HTMLDivElement>(null);
  const slideRefs = useRef<Array<HTMLElement | null>>([]);
  const [activeSlide, setActiveSlide] = useState(0);

  const scrollToSlide = useCallback(
    (idx: number) => {
      const bounded = Math.max(0, Math.min(TOTAL - 1, idx));
      const node = slideRefs.current[bounded];
      if (!node) return;
      node.scrollIntoView({ behavior: "smooth", block: "start" });
    },
    [],
  );

  useEffect(() => {
    const host = containerRef.current;
    if (!host) return;
    const onScroll = () => {
      const center = host.scrollTop + host.clientHeight / 2;
      let bestIdx = 0;
      let bestDist = Number.POSITIVE_INFINITY;
      slideRefs.current.forEach((el, idx) => {
        if (!el) return;
        const elCenter = el.offsetTop + el.clientHeight / 2;
        const dist = Math.abs(center - elCenter);
        if (dist < bestDist) {
          bestDist = dist;
          bestIdx = idx;
        }
      });
      setActiveSlide(bestIdx);
    };
    onScroll();
    host.addEventListener("scroll", onScroll, { passive: true });
    return () => host.removeEventListener("scroll", onScroll);
  }, []);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) return;
      if (e.key === "Home") {
        e.preventDefault();
        scrollToSlide(0);
        return;
      }
      if (e.key === "End") {
        e.preventDefault();
        scrollToSlide(TOTAL - 1);
        return;
      }
      if (e.key === " " && e.shiftKey) {
        e.preventDefault();
        scrollToSlide(activeSlide - 1);
        return;
      }
      if (e.key === " " || e.key === "PageDown" || e.key === "ArrowRight") {
        e.preventDefault();
        scrollToSlide(activeSlide + 1);
        return;
      }
      if (e.key === "PageUp" || e.key === "ArrowLeft") {
        e.preventDefault();
        scrollToSlide(activeSlide - 1);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [activeSlide, scrollToSlide]);

  return (
    <main className="stage-bg relative h-dvh w-full overflow-hidden text-foreground">
      <DeckBackdrop />

      <header className="pointer-events-none absolute inset-x-0 top-0 z-20">
        <div className="mx-auto flex max-w-6xl items-center justify-start px-5 py-4 sm:px-6">
          <Link
            to="/"
            className="pointer-events-auto text-xs text-muted-foreground transition-colors hover:text-foreground"
            aria-label="Back to home"
          >
            <span className="inline-flex items-center gap-1">
              <ArrowLeft className="h-3.5 w-3.5" strokeWidth={1.5} />
              Home
            </span>
          </Link>
        </div>
      </header>

      <div
        ref={containerRef}
        className="h-dvh snap-y snap-mandatory overflow-y-auto scroll-smooth"
      >
        {SLIDES.map((Slide, idx) => {
          const isHook = idx < 3;
          return (
            <section
              key={idx}
              ref={(el) => {
                slideRefs.current[idx] = el;
              }}
              className={
                isHook
                  ? "flex min-h-dvh snap-start flex-col p-0"
                  : "flex min-h-dvh snap-start flex-col px-4 py-10 sm:px-6 sm:py-12"
              }
            >
              <div
                className={
                  isHook
                    ? "flex h-full min-h-dvh w-full min-h-0 flex-1 flex-col"
                    : "flex min-h-0 w-full max-w-7xl flex-1 flex-col justify-center self-center"
                }
              >
                <Slide />
              </div>
            </section>
          );
        })}
      </div>

      <div className="pointer-events-none fixed inset-x-0 bottom-4 z-30 flex items-end justify-center px-4 sm:px-6">
        <div className="pointer-events-auto inline-flex items-center gap-0.5 rounded-full border border-border/60 bg-background/80 px-1 py-0.5 backdrop-blur-sm sm:px-1.5">
          <button
            type="button"
            onClick={() => scrollToSlide(activeSlide - 1)}
            disabled={activeSlide === 0}
            className="inline-flex h-8 w-8 items-center justify-center rounded-full text-muted-foreground transition hover:text-foreground disabled:cursor-not-allowed disabled:opacity-30"
            aria-label="Previous slide"
          >
            <ChevronLeft className="h-4 w-4" />
          </button>
          <span className="min-w-[72px] text-center font-mono text-[10px] font-medium tabular-nums tracking-widest text-muted-foreground">
            {String(activeSlide + 1).padStart(2, "0")} / {TOTAL}
          </span>
          <button
            type="button"
            onClick={() => scrollToSlide(activeSlide + 1)}
            disabled={activeSlide === TOTAL - 1}
            className="inline-flex h-8 w-8 items-center justify-center rounded-full text-muted-foreground transition hover:text-foreground disabled:cursor-not-allowed disabled:opacity-30"
            aria-label="Next slide"
          >
            <ChevronRight className="h-4 w-4" />
          </button>
        </div>
      </div>
    </main>
  );
};

export default PitchDeck;
