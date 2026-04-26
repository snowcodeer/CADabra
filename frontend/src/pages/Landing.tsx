import { Link } from "react-router-dom";
import { Cog } from "lucide-react";
import { CadabraCadLockup } from "@/components/CadabraWordmark";
import { LandingHeroModel } from "@/components/landing/LandingHeroModel";

const Landing = () => {
  return (
    <main className="relative flex h-screen-dvh min-h-0 w-full flex-col overflow-hidden stage-bg text-foreground site-pad-bottom">
      <div
        aria-hidden
        className="pointer-events-none absolute inset-0 opacity-[0.36]"
        style={{
          backgroundImage:
            "linear-gradient(hsl(220 14% 82% / 0.65) 1px, transparent 1px), linear-gradient(90deg, hsl(220 14% 82% / 0.65) 1px, transparent 1px)",
          backgroundSize: "min(10vw, 72px) min(10vw, 72px)",
          maskImage: "radial-gradient(ellipse at 50% 50%, black 0%, transparent 78%)",
          WebkitMaskImage: "radial-gradient(ellipse at 50% 50%, black 0%, transparent 78%)",
        }}
      />
      <div
        aria-hidden
        className="pointer-events-none absolute inset-0 opacity-[0.2]"
        style={{
          backgroundImage:
            "linear-gradient(hsl(220 14% 80% / 0.45) 1px, transparent 1px), linear-gradient(90deg, hsl(220 14% 80% / 0.45) 1px, transparent 1px)",
          backgroundSize: "min(5vw, 36px) min(5vw, 36px)",
          maskImage: "radial-gradient(ellipse at 50% 50%, black 0%, transparent 78%)",
          WebkitMaskImage: "radial-gradient(ellipse at 50% 50%, black 0%, transparent 78%)",
        }}
      />
      <section className="site-gutter-x relative z-10 flex w-full min-h-0 flex-1 flex-col items-center justify-center py-6 sm:py-8">
        {/* w-fit: only as wide as copy + small gap + viewer — no huge empty space between */}
        <div className="flex w-full max-w-full min-w-0 flex-col items-center gap-5 sm:gap-6 md:mx-auto md:w-fit md:max-w-full md:flex-row md:items-center md:gap-0">
          <div className="w-full max-w-[20rem] shrink-0 text-left sm:max-w-[21rem] md:max-w-[20.5rem] md:pr-0.5">
            <p className="font-mono text-[0.625rem] uppercase leading-relaxed tracking-[0.28em] text-muted-foreground/90 sm:text-[0.65rem] sm:tracking-[0.32em]">
              Scan to editable CAD
            </p>
            <h1 className="font-wordmark mt-3 inline-flex flex-wrap items-baseline gap-x-0 text-[2.5rem] leading-[0.97] tracking-[-0.025em] text-foreground sm:mt-4 sm:text-[2.75rem] lg:text-[3.25rem]">
              <CadabraCadLockup cadLetterClassName="text-inherit" />
              <span className="font-light italic text-foreground/70">abra</span>
            </h1>
            <p className="mt-4 max-w-[24ch] text-balance text-[1.03rem] leading-[1.66] tracking-[-0.01em] text-muted-foreground/95 sm:mt-5 sm:text-[1.08rem]">
              Turn noisy point clouds into clean, parametric geometry with engineering intent intact.
            </p>
            <div className="mt-8 flex flex-wrap items-center justify-start gap-3 sm:mt-10">
              <Link
                to="/workflow"
                className="group inline-flex min-h-11 min-w-[2.75rem] items-center justify-center gap-2 rounded-full border border-foreground/20 bg-foreground px-6 py-2.5 text-xs font-medium uppercase tracking-[0.18em] text-background transition hover:opacity-90"
              >
                BEGIN
                <Cog
                  className="h-3.5 w-3.5 transition-transform duration-300 ease-out group-hover:rotate-90 group-hover:scale-110"
                  strokeWidth={1.8}
                />
              </Link>
              <Link
                to="/pitchdeck"
                className="inline-flex min-h-11 min-w-[2.75rem] items-center justify-center rounded-full border border-border bg-surface/90 px-6 py-2.5 text-xs font-medium uppercase tracking-[0.18em] text-foreground transition hover:bg-surface"
              >
                PITCH DECK
              </Link>
            </div>
          </div>
          <div className="relative h-[19rem] w-[19rem] shrink-0 sm:h-[20.5rem] sm:w-[20.5rem] md:h-[22rem] md:w-[22rem] md:pl-0.5 lg:h-96 lg:w-96">
            <LandingHeroModel />
          </div>
        </div>
      </section>
    </main>
  );
};

export default Landing;
