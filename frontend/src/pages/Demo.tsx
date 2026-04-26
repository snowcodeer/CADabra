import { useCallback, useMemo, useState } from "react";
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
  Download,
  PencilRuler,
  RotateCcw,
} from "lucide-react";
import { Scene, type CompareAnalysis } from "@/components/cad/Scene";
import { CadabraCadLockup } from "@/components/CadabraWordmark";
import {
  SAMPLE_000035_INITIAL_PARAMS,
  Sample000035EditorScene,
  type Sample000035Params,
} from "@/components/cad/Sample000035EditorScene";
import { API_BASE } from "@/lib/api";

const STEP_URL = `${API_BASE}/outputs/ortho_deepcadimg_000035.step`;

const Demo = () => {
  const [compareMode, setCompareMode] = useState(false);
  const [analysis, setAnalysis] = useState<CompareAnalysis | null>(null);
  const [editorOpen, setEditorOpen] = useState(false);
  const [params, setParams] = useState<Sample000035Params>(SAMPLE_000035_INITIAL_PARAMS);

  const handleAnalysis = useCallback((a: CompareAnalysis) => setAnalysis(a), []);

  const delta = useMemo(
    () => ({
      base: params.baseHeightMm - SAMPLE_000035_INITIAL_PARAMS.baseHeightMm,
      boss: params.bossHeightMm - SAMPLE_000035_INITIAL_PARAMS.bossHeightMm,
      counterbore:
        params.counterboreDepthMm - SAMPLE_000035_INITIAL_PARAMS.counterboreDepthMm,
    }),
    [params],
  );

  if (editorOpen) {
    return (
      <main className="relative h-screen-dvh w-full overflow-hidden stage-bg text-foreground site-pad-bottom">
        <header className="site-gutter-x pointer-events-none absolute inset-x-0 top-0 z-20 flex items-center justify-between py-4 sm:py-5 md:py-6">
          <div className="pointer-events-auto inline-flex items-baseline gap-0">
            <CadabraCadLockup
              cadLetterClassName="font-wordmark text-[length:clamp(1.75rem,calc(1.1rem+1.2vw),2.4rem)] font-bold leading-none tracking-[-0.02em] bg-gradient-to-br from-foreground via-foreground to-foreground/60 bg-clip-text text-transparent"
            />
            <span className="font-wordmark text-[length:clamp(1.75rem,calc(1.1rem+1.2vw),2.4rem)] font-light italic leading-none tracking-[-0.02em] text-foreground/80">
              abra
            </span>
          </div>

          <div className="pointer-events-auto flex items-center gap-2">
            <button
              type="button"
              onClick={() => setEditorOpen(false)}
              className="inline-flex items-center gap-2 rounded-full border border-border bg-background/75 px-4 py-2 text-sm font-medium text-foreground backdrop-blur-sm transition-colors hover:bg-background"
            >
              <X className="h-4 w-4" strokeWidth={1.8} />
              <span>Back to compare</span>
            </button>
            <button
              type="button"
              onClick={() => setParams(SAMPLE_000035_INITIAL_PARAMS)}
              className="inline-flex items-center gap-2 rounded-full border border-border bg-background/75 px-4 py-2 text-sm font-medium text-foreground backdrop-blur-sm transition-colors hover:bg-background"
            >
              <RotateCcw className="h-4 w-4" strokeWidth={1.8} />
              <span>Reset</span>
            </button>
            <a
              href={STEP_URL}
              target="_blank"
              rel="noreferrer"
              className="inline-flex items-center gap-2 rounded-full bg-foreground px-4 py-2 text-sm font-medium text-background transition-opacity hover:opacity-90"
            >
              <Download className="h-4 w-4" strokeWidth={1.8} />
              <span>Open STEP</span>
            </a>
          </div>
        </header>

        <div className="absolute inset-0">
          <Sample000035EditorScene
            params={params}
            onChange={setParams}
            interactive
          />
        </div>

        <aside className="pointer-events-none absolute left-6 top-24 z-20 w-[min(24rem,calc(100vw-3rem))] rounded-3xl border border-border bg-background/82 p-5 shadow-xl backdrop-blur-md sm:left-8 sm:top-28">
          <div className="pointer-events-auto">
            <div className="text-[10px] font-medium uppercase tracking-[0.24em] text-muted-foreground">
              Parametric editor
            </div>
            <h1 className="mt-2 text-xl font-semibold tracking-tight text-foreground">
              `ortho_deepcadimg_000035.step`
            </h1>
            <p className="mt-2 text-sm leading-6 text-muted-foreground">
              Drag the highlighted horizontal surfaces up and down to change the base height,
              boss height, and counterbore depth.
            </p>

            <div className="mt-5 space-y-3">
              <MetricRow label="Base extrusion" value={params.baseHeightMm} delta={delta.base} />
              <MetricRow label="Boss extrusion" value={params.bossHeightMm} delta={delta.boss} />
              <MetricRow
                label="Counterbore depth"
                value={params.counterboreDepthMm}
                delta={delta.counterbore}
              />
            </div>

            <div className="mt-5 rounded-2xl border border-border bg-surface/70 p-4">
              <div className="flex items-center gap-3">
                <div className="flex flex-col items-center text-muted-foreground">
                  <ArrowUp className="h-2.5 w-2.5" strokeWidth={2.2} />
                  <MousePointer2 className="my-0.5 h-4 w-4 text-foreground" strokeWidth={1.6} />
                  <ArrowDown className="h-2.5 w-2.5" strokeWidth={2.2} />
                </div>
                <div>
                  <div className="text-xs font-semibold text-foreground">Drag the lit surfaces</div>
                  <p className="mt-1 text-[11px] leading-relaxed text-muted-foreground">
                    Outer flange top changes the base thickness. The boss top changes its extrusion.
                    The recessed annulus inside the boss changes the counterbore depth.
                  </p>
                </div>
              </div>
            </div>
          </div>
        </aside>
      </main>
    );
  }

  return (
    <main className="relative h-screen-dvh w-full min-w-0 overflow-hidden stage-bg site-pad-bottom">
      <h1 className="sr-only">
        CADabra · Point Cloud to Parametric CAD reconstruction
      </h1>

      <header className="site-gutter-x pointer-events-none absolute inset-x-0 top-0 z-20 flex items-center justify-between py-4 sm:py-5 md:py-6">
        <div className="pointer-events-auto inline-flex items-baseline gap-0 opacity-0 animate-[demo-slide-in-left_700ms_var(--ease-out-soft)_120ms_forwards]">
          <CadabraCadLockup
            cadLetterClassName="font-wordmark text-[length:clamp(1.75rem,calc(1.1rem+1.2vw),2.4rem)] font-bold leading-none tracking-[-0.02em] bg-gradient-to-br from-foreground via-foreground to-foreground/60 bg-clip-text text-transparent [text-shadow:0_1px_0_rgba(0,0,0,0.04)]"
          />
          <span className="font-wordmark text-[length:clamp(1.75rem,calc(1.1rem+1.2vw),2.4rem)] font-light italic leading-none tracking-[-0.02em] text-foreground/80">
            abra
          </span>
        </div>

        <div className="pointer-events-auto flex items-center gap-2">
          <button
            type="button"
            className="opacity-0 animate-[demo-slide-in-right_650ms_var(--ease-out-soft)_220ms_forwards] inline-flex items-center gap-2 rounded-full border border-border bg-background/70 px-4 py-2 text-sm font-medium text-foreground backdrop-blur-sm transition-colors hover:bg-background"
          >
            <Grid2X2 className="h-4 w-4" strokeWidth={1.8} />
            <span>Gallery</span>
          </button>
          <button
            type="button"
            className="opacity-0 animate-[demo-slide-in-right_650ms_var(--ease-out-soft)_320ms_forwards] inline-flex items-center gap-2 rounded-full bg-foreground px-4 py-2 text-sm font-medium text-background transition-opacity hover:opacity-90"
          >
            <Upload className="h-4 w-4" strokeWidth={1.8} />
            <span>Export STEP</span>
          </button>
          <button
            type="button"
            onClick={() => setEditorOpen(true)}
            className="opacity-0 animate-[demo-slide-in-right_650ms_var(--ease-out-soft)_420ms_forwards] inline-flex items-center gap-2 rounded-full border border-border bg-background/78 px-4 py-2 text-sm font-medium text-foreground backdrop-blur-sm transition-colors hover:bg-background"
          >
            <PencilRuler className="h-4 w-4" strokeWidth={1.8} />
            <span>Edit Shape</span>
          </button>
        </div>
      </header>

      <div className="absolute inset-0 opacity-0 animate-[demo-stage-in_900ms_var(--ease-out-soft)_80ms_forwards]">
        <Scene compareMode={compareMode} onAnalysis={handleAnalysis} />
      </div>

      {!compareMode && (
        <div className="pointer-events-none absolute inset-x-0 bottom-10 z-10 flex justify-center opacity-0 animate-[demo-rise-in_700ms_var(--ease-out-soft)_520ms_forwards]">
          <div className="flex items-center gap-3 rounded-2xl bg-surface/70 px-4 py-2.5 backdrop-blur-sm">
            <div className="flex flex-col items-center text-muted-foreground">
              <ArrowUp className="h-2.5 w-2.5" strokeWidth={2.2} />
              <MousePointer2 className="my-0.5 h-4 w-4 text-foreground" strokeWidth={1.6} />
              <ArrowDown className="h-2.5 w-2.5" strokeWidth={2.2} />
            </div>
            <div className="text-left">
              <div className="text-xs font-semibold text-foreground">Inspect the reconstruction</div>
              <p className="max-w-[18rem] text-[11px] leading-relaxed text-muted-foreground">
                The three-podium view now uses the sample geometry instead of the placeholder brick. Use Edit Shape to enter the parametric editor.
              </p>
            </div>
          </div>
        </div>
      )}

      <aside
        className={`pointer-events-none absolute inset-x-0 bottom-0 z-20 flex justify-center px-6 pb-6 transition-all duration-500 ease-out ${
          compareMode ? "translate-y-0 opacity-100" : "translate-y-8 opacity-0"
        }`}
        aria-hidden={!compareMode}
      >
        <div className="pointer-events-auto flex w-full max-w-[640px] items-center gap-5 rounded-2xl border border-border bg-background/85 px-5 py-3.5 shadow-xl backdrop-blur-md">
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

      <div
        className={`pointer-events-none absolute right-[max(0.75rem,env(safe-area-inset-right))] z-30 transition-all duration-300 sm:right-6 md:right-8 ${
          compareMode ? "top-20 sm:top-24" : "bottom-[max(1.25rem,env(safe-area-inset-bottom))] sm:bottom-10"
        }`}
      >
        <button
          type="button"
          onClick={() => setCompareMode((v) => !v)}
          aria-pressed={compareMode}
          className={`pointer-events-auto inline-flex items-center gap-2 rounded-full px-5 py-2.5 text-sm font-medium shadow-lg backdrop-blur-sm transition-all opacity-0 animate-[demo-rise-in_650ms_var(--ease-out-soft)_420ms_forwards] ${
            compareMode
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

function MetricRow({
  label,
  value,
  delta,
}: {
  label: string;
  value: number;
  delta: number;
}) {
  const deltaLabel = `${delta >= 0 ? "+" : ""}${delta.toFixed(1)} mm`;
  return (
    <div className="rounded-2xl border border-border bg-surface/65 px-4 py-3">
      <div className="flex items-center justify-between gap-3">
        <span className="text-sm text-foreground">{label}</span>
        <span className="font-mono text-sm text-foreground">{value.toFixed(1)} mm</span>
      </div>
      <div className="mt-1 text-[11px] uppercase tracking-[0.14em] text-muted-foreground">
        Delta {deltaLabel}
      </div>
    </div>
  );
}

export default Demo;
