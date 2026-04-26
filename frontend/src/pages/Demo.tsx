import { useCallback, useState } from "react";
import {
  MousePointer2,
  ArrowUp,
  ArrowDown,
  Grid2X2,
  Upload,
  GitCompareArrows,
  X,
  CheckCircle2,
  AlertTriangle,
} from "lucide-react";
import { Scene, type CompareAnalysis } from "@/components/cad/Scene";

/**
 * CADabra — single-screen showcase: Point Cloud → Parametric CAD.
 * Three podiums; the centerpiece is interactive (hover face, drag to extrude).
 * Compare mode hides the point cloud, glides the camera into an inspection
 * angle, tints the Ground Truth (green = matched, red = missed) and opens
 * a side analysis panel summarising coverage.
 */
const Index = () => {
  const [compareMode, setCompareMode] = useState(false);
  const [analysis, setAnalysis] = useState<CompareAnalysis | null>(null);

  // Stable callback so Scene's useEffect dependency doesn't churn each render.
  const handleAnalysis = useCallback((a: CompareAnalysis) => setAnalysis(a), []);

  return (
    <main className="relative h-screen w-screen overflow-hidden stage-bg">
      {/* H1 for SEO — visually hidden */}
      <h1 className="sr-only">
        CADabra · Point Cloud to Parametric CAD reconstruction
      </h1>

      {/* Top bar */}
      <header className="pointer-events-none absolute inset-x-0 top-0 z-20 flex items-center justify-between px-8 py-6">
        {/* Wordmark — slides in from the left */}
        <div
          className="pointer-events-auto opacity-0 animate-[demo-slide-in-left_700ms_var(--ease-out-soft)_120ms_forwards]"
        >
          <span
            className="font-wordmark text-[38px] font-bold tracking-[-0.02em]
                       bg-gradient-to-br from-foreground via-foreground to-foreground/60
                       bg-clip-text text-transparent
                       [text-shadow:0_1px_0_rgba(0,0,0,0.04)]"
          >
            CAD
          </span>
          <span
            className="font-wordmark text-[38px] font-light italic tracking-[-0.02em] text-foreground/80"
          >
            abra
          </span>
        </div>

        {/* Actions — slide in from the right, staggered */}
        <div className="pointer-events-auto flex items-center gap-2">
          <button
            type="button"
            className="opacity-0 animate-[demo-slide-in-right_650ms_var(--ease-out-soft)_220ms_forwards]
                       inline-flex items-center gap-2 rounded-full border border-border
                       bg-background/70 px-4 py-2 text-sm font-medium text-foreground
                       backdrop-blur-sm transition-colors hover:bg-background"
          >
            <Grid2X2 className="h-4 w-4" strokeWidth={1.8} />
            <span>Gallery</span>
          </button>
          <button
            type="button"
            className="opacity-0 animate-[demo-slide-in-right_650ms_var(--ease-out-soft)_320ms_forwards]
                       inline-flex items-center gap-2 rounded-full bg-foreground
                       px-4 py-2 text-sm font-medium text-background
                       transition-opacity hover:opacity-90"
          >
            <Upload className="h-4 w-4" strokeWidth={1.8} />
            <span>Export STEP</span>
          </button>
        </div>
      </header>

      {/* 3D stage — fades up underneath the chrome */}
      <div className="absolute inset-0 opacity-0 animate-[demo-stage-in_900ms_var(--ease-out-soft)_80ms_forwards]">
        <Scene compareMode={compareMode} onAnalysis={handleAnalysis} />
      </div>

      {/* Bottom hint (hidden in compare mode) — pops up last */}
      {!compareMode && (
        <div
          className="pointer-events-none absolute inset-x-0 bottom-10 z-10
                     flex justify-center opacity-0
                     animate-[demo-rise-in_700ms_var(--ease-out-soft)_520ms_forwards]"
        >
          <div className="flex items-center gap-3 rounded-2xl bg-surface/70 px-4 py-2.5 backdrop-blur-sm">
            <div className="flex flex-col items-center text-muted-foreground">
              <ArrowUp className="h-2.5 w-2.5" strokeWidth={2.2} />
              <MousePointer2 className="my-0.5 h-4 w-4 text-foreground" strokeWidth={1.6} />
              <ArrowDown className="h-2.5 w-2.5" strokeWidth={2.2} />
            </div>
            <div className="text-left">
              <div className="text-xs font-semibold text-foreground">Pull to extrude</div>
              <p className="max-w-[18rem] text-[11px] leading-relaxed text-muted-foreground">
                Click and pull the model from any face to extrude and explore the geometry.
              </p>
            </div>
          </div>
        </div>
      )}

      {/* Bottom analysis panel — minimal: coverage bar + matched / missed.
          Slides up from the bottom when compare mode activates. */}
      <aside
        className={`pointer-events-none absolute inset-x-0 bottom-0 z-20
                    flex justify-center px-6 pb-6
                    transition-all duration-500 ease-out
                    ${compareMode
                      ? "translate-y-0 opacity-100"
                      : "translate-y-8 opacity-0"}
                   `}
        aria-hidden={!compareMode}
      >
        <div className="pointer-events-auto flex w-full max-w-[640px] items-center gap-5 rounded-2xl border border-border bg-background/85 px-5 py-3.5 shadow-xl backdrop-blur-md">
          {/* Coverage — primary readout */}
          <div className="flex flex-1 items-center gap-3">
            <div className="text-[10px] font-medium uppercase tracking-[0.22em] text-muted-foreground">
              Coverage
            </div>
            <div className="flex flex-1 items-center gap-3">
              <div className="h-1.5 flex-1 overflow-hidden rounded-full bg-foreground/10">
                <div
                  className="h-full rounded-full bg-[#22c55e] transition-all duration-500"
                  style={{ width: `${analysis ? analysis.coverage : 100}%` }}
                />
              </div>
              <span className="w-10 text-right text-base font-semibold tabular-nums text-foreground">
                {analysis ? analysis.coverage.toFixed(0) : 100}%
              </span>
            </div>
          </div>

          {/* Matched / Missed — compact pair */}
          <div className="flex items-center gap-4 border-l border-border pl-5">
            <div className="flex items-center gap-1.5">
              <CheckCircle2 className="h-3.5 w-3.5 text-[#22c55e]" strokeWidth={2.2} />
              <span className="text-sm font-semibold tabular-nums text-foreground">
                {analysis ? analysis.matchedMm : 0}
                <span className="ml-0.5 text-[10px] font-normal text-muted-foreground">mm</span>
              </span>
            </div>
            <div className="flex items-center gap-1.5">
              <AlertTriangle className="h-3.5 w-3.5 text-[#ef4444]" strokeWidth={2.2} />
              <span className="text-sm font-semibold tabular-nums text-foreground">
                {analysis ? analysis.missedMm : 0}
                <span className="ml-0.5 text-[10px] font-normal text-muted-foreground">mm</span>
              </span>
            </div>
          </div>
        </div>
      </aside>

      {/* Compare button — bottom right when not in compare mode, top when in
          compare mode (the bottom area is used by the analysis panel). */}
      <div
        className={`pointer-events-none absolute right-8 z-30 transition-all duration-300 ${
          compareMode ? "top-24" : "bottom-10"
        }`}
      >
        <button
          type="button"
          onClick={() => setCompareMode((v) => !v)}
          aria-pressed={compareMode}
          className={`pointer-events-auto inline-flex items-center gap-2 rounded-full px-5 py-2.5
                     text-sm font-medium shadow-lg backdrop-blur-sm transition-all
                     opacity-0 animate-[demo-rise-in_650ms_var(--ease-out-soft)_420ms_forwards]
                     ${compareMode
                       ? "bg-foreground text-background hover:opacity-90"
                       : "border border-border bg-background/80 text-foreground hover:bg-background"
                     }`}
        >
          {compareMode ? (
            <>
              <X className="h-4 w-4" strokeWidth={1.8} />
              <span>Exit Compare</span>
            </>
          ) : (
            <>
              <GitCompareArrows className="h-4 w-4" strokeWidth={1.8} />
              <span>Compare</span>
            </>
          )}
        </button>
      </div>
    </main>
  );
};

export default Index;
