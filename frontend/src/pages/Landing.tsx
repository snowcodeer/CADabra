import { Link } from "react-router-dom";
import { ArrowRight } from "lucide-react";
import { LandingHeroModel } from "@/components/landing/LandingHeroModel";

const Landing = () => {
  return (
    <main className="relative h-dvh w-full overflow-hidden stage-bg text-foreground">
      <div
        aria-hidden
        className="pointer-events-none absolute inset-0 opacity-[0.2]"
        style={{
          backgroundImage:
            "linear-gradient(hsl(220 14% 88% / 0.5) 1px, transparent 1px), linear-gradient(90deg, hsl(220 14% 88% / 0.5) 1px, transparent 1px)",
          backgroundSize: "72px 72px",
          maskImage: "radial-gradient(ellipse at 50% 50%, black 0%, transparent 78%)",
          WebkitMaskImage: "radial-gradient(ellipse at 50% 50%, black 0%, transparent 78%)",
        }}
      />
      <section className="relative z-10 mx-auto flex h-dvh w-full max-w-6xl flex-col items-stretch justify-center gap-8 px-4 py-8 sm:px-5 md:flex-row md:items-center md:gap-3 md:pl-10 md:pr-6 lg:max-w-7xl lg:gap-4 lg:pl-14">
        <div className="flex w-full min-w-0 shrink-0 justify-end md:flex-[0_0_min(46%,30rem)] md:pl-2 lg:pl-4">
          <div className="w-full max-w-md text-center md:max-w-lg md:translate-x-6 md:pl-0 md:pr-0 md:text-right lg:translate-x-10 xl:translate-x-14">
            <p className="font-mono text-[10px] uppercase leading-relaxed tracking-[0.28em] text-muted-foreground sm:tracking-[0.32em]">
              Scan to editable CAD
            </p>
            <h1 className="font-wordmark mt-5 text-5xl leading-[0.98] tracking-[-0.02em] text-foreground sm:text-6xl">
              CAD<span className="font-light italic text-foreground/70">abra</span>
            </h1>
            <p className="mt-6 text-balance text-base leading-relaxed text-muted-foreground sm:text-lg">
              Turn noisy point clouds into clean, parametric geometry with engineering intent intact.
            </p>
            <div className="mt-10 flex flex-wrap items-center justify-center gap-3 md:justify-end">
              <Link
                to="/workflow"
                className="inline-flex items-center gap-2 rounded-full border border-foreground/20 bg-foreground px-6 py-2.5 text-xs font-medium uppercase tracking-[0.18em] text-background transition hover:opacity-90"
              >
                BEGIN
                <ArrowRight className="h-3.5 w-3.5" strokeWidth={1.8} />
              </Link>
              <Link
                to="/pitchdeck"
                className="inline-flex items-center rounded-full border border-border bg-surface/90 px-6 py-2.5 text-xs font-medium uppercase tracking-[0.18em] text-foreground transition hover:bg-surface"
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
