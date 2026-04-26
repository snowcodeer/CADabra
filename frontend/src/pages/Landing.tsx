import { Link } from "react-router-dom";
import { ArrowRight } from "lucide-react";
import { LandingHeroModel } from "@/components/landing/LandingHeroModel";

const Landing = () => {
  return (
    <main className="relative flex h-screen-dvh min-h-0 w-full flex-col overflow-hidden stage-bg text-foreground site-pad-bottom">
      <div
        aria-hidden
        className="pointer-events-none absolute inset-0 opacity-[0.2]"
        style={{
          backgroundImage:
            "linear-gradient(hsl(220 14% 88% / 0.5) 1px, transparent 1px), linear-gradient(90deg, hsl(220 14% 88% / 0.5) 1px, transparent 1px)",
          backgroundSize: "min(10vw, 72px) min(10vw, 72px)",
          maskImage: "radial-gradient(ellipse at 50% 50%, black 0%, transparent 78%)",
          WebkitMaskImage: "radial-gradient(ellipse at 50% 50%, black 0%, transparent 78%)",
        }}
      />
      <section className="site-gutter-x relative z-10 mx-auto flex w-full min-h-0 max-w-6xl flex-1 flex-col items-stretch justify-center gap-6 py-6 sm:gap-8 sm:py-8 md:flex-row md:items-center md:gap-3 md:py-4 lg:max-w-7xl lg:gap-4 2xl:max-w-8xl">
        <div className="flex w-full min-w-0 shrink-0 justify-end md:flex-[0_0_min(46%,30rem)] md:pl-0 lg:pl-1">
          <div className="w-full max-w-md text-center md:max-w-lg md:translate-x-4 md:pl-0 md:pr-0 md:text-right 2xl:translate-x-10 3xl:translate-x-12">
            <p className="font-mono text-[0.625rem] uppercase leading-relaxed tracking-[0.28em] text-muted-foreground sm:text-[0.65rem] sm:tracking-[0.32em]">
              Scan to editable CAD
            </p>
            <h1 className="font-wordmark mt-4 text-[length:clamp(2.4rem,calc(1.1rem+3.2vw),3.75rem)] leading-[0.98] tracking-[-0.02em] text-foreground sm:mt-5">
              CAD<span className="font-light italic text-foreground/70">abra</span>
            </h1>
            <p className="mt-5 text-balance text-[length:clamp(0.94rem,calc(0.82rem+0.2vw),1.2rem)] leading-[1.55] text-muted-foreground sm:mt-6 sm:leading-[1.6] md:leading-[1.65]">
              Turn noisy point clouds into clean, parametric geometry with engineering intent intact.
            </p>
            <div className="mt-10 flex flex-wrap items-center justify-center gap-3 md:justify-end">
              <Link
                to="/workflow"
                className="inline-flex min-h-11 min-w-[2.75rem] items-center justify-center gap-2 rounded-full border border-foreground/20 bg-foreground px-6 py-2.5 text-xs font-medium uppercase tracking-[0.18em] text-background transition hover:opacity-90"
              >
                BEGIN
                <ArrowRight className="h-3.5 w-3.5" strokeWidth={1.8} />
              </Link>
              <Link
                to="/pitchdeck"
                className="inline-flex min-h-11 min-w-[2.75rem] items-center justify-center rounded-full border border-border bg-surface/90 px-6 py-2.5 text-xs font-medium uppercase tracking-[0.18em] text-foreground transition hover:bg-surface"
              >
                PITCH DECK
              </Link>
            </div>
          </div>
        </div>
        <div className="relative w-full min-h-[260px] min-w-0 max-w-md flex-1 md:max-w-none md:-translate-x-1 lg:-translate-x-2.5">
          <div className="h-[min(44vh,340px)] w-full md:h-[min(58vh,500px)]">
            <LandingHeroModel />
          </div>
        </div>
      </section>
    </main>
  );
};

export default Landing;
