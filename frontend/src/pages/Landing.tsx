import { Link } from "react-router-dom";
import { Cog } from "lucide-react";
import { CadabraCadLockup } from "@/components/CadabraWordmark";
import { LandingHeroModel } from "@/components/landing/LandingHeroModel";

const Landing = () => {
  return (
    <main className="relative flex min-h-dvh min-h-0 w-full flex-col overflow-hidden stage-bg text-foreground site-pad-bottom">
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
      <section className="site-gutter-x relative z-10 flex w-full min-h-0 flex-1 flex-col items-stretch justify-center py-6 sm:py-8">
        {/* md+: 50/50; copy end-aligned in left half, 3D start-aligned in right half (meet at center) */}
        <div className="mx-auto grid w-full min-w-0 max-w-full grid-cols-1 content-center items-center justify-items-center gap-5 sm:gap-6 md:grid-cols-2 md:items-center md:justify-items-stretch md:gap-0">
          <div className="flex w-full min-w-0 max-w-full justify-center justify-self-stretch md:justify-start">
            <div className="w-full max-w-[28rem] min-w-0 shrink-0 text-left sm:max-w-[32rem] md:ms-auto md:translate-x-2">
              <div className="flex flex-col gap-2 sm:gap-2.5">
                <p className="font-mono text-base uppercase leading-tight tracking-[0.26em] text-muted-foreground/90 sm:text-lg sm:tracking-[0.28em]">
                  Scan to editable CAD
                </p>
                <h1 className="font-wordmark m-0 inline-flex flex-wrap items-baseline gap-x-0 text-[3.75rem] leading-[0.97] tracking-[-0.025em] text-foreground sm:text-[4.25rem] md:text-[4.75rem] lg:text-[5.25rem]">
                  <CadabraCadLockup
                    cadLetterClassName="text-inherit"
                    logoWrapperClassName="inline-block shrink-0 [aspect-ratio:352/402] h-[1.45cap] w-auto translate-y-[2px] align-baseline [font:inherit]"
                  />
                  <span className="font-light italic text-foreground/70">abra</span>
                </h1>
                <p className="m-0 max-w-[32ch] text-balance text-2xl leading-[1.45] tracking-[-0.01em] text-muted-foreground/95 sm:max-w-[36ch] sm:text-[1.75rem] sm:leading-[1.4] md:text-3xl md:leading-[1.4]">
                  Turn noisy point clouds into clean, parametric geometry with engineering intent intact.
                </p>
              </div>
              <div className="mt-8 flex flex-wrap items-center justify-start gap-3.5 sm:mt-9">
                <Link
                  to="/workflow"
                  className="group inline-flex min-h-12 min-w-[3rem] items-center justify-center gap-2 rounded-full border border-foreground/20 bg-foreground px-7 py-3 text-sm font-medium uppercase tracking-[0.16em] text-background transition hover:opacity-90"
                >
                  BEGIN
                  <Cog
                    className="h-4 w-4 transition-transform duration-300 ease-out group-hover:rotate-90 group-hover:scale-110"
                    strokeWidth={1.8}
                  />
                </Link>
                <Link
                  to="/pitchdeck"
                  className="inline-flex min-h-12 min-w-[3rem] items-center justify-center rounded-full border border-border bg-surface/90 px-7 py-3 text-sm font-medium uppercase tracking-[0.16em] text-foreground transition hover:bg-surface"
                >
                  PITCH DECK
                </Link>
              </div>
            </div>
          </div>
          <div className="flex w-full min-w-0 max-w-full justify-center justify-self-stretch md:justify-start">
            <div className="relative h-[20.5rem] w-[20.5rem] shrink-0 sm:h-[22rem] sm:w-[22rem] md:h-[24rem] md:w-[24rem] lg:h-[28rem] lg:w-[28rem]">
              <LandingHeroModel />
            </div>
          </div>
        </div>
      </section>
      <footer className="site-gutter-x relative z-10 mt-auto flex shrink-0 justify-center border-t border-border/40 py-3 sm:py-4">
        <p className="text-center text-xs text-muted-foreground/80 sm:text-sm">
          CADabra © 2026
        </p>
      </footer>
    </main>
  );
};

export default Landing;
