import { useEffect, useRef } from "react";
import {
  prepareWithSegments,
  layoutNextLine,
  type PreparedTextWithSegments,
  type LayoutCursor,
} from "@chenglou/pretext";
import { cursorCharacter } from "@/components/cursor/cursorCharacterState";

/**
 * Canvas-rendered paragraph that flows around the live `CursorCharacter`
 * silhouette using a per-frame scanline profile of the sprite footprint.
 *
 * The host DOM element controls position + size in normal flow; we
 * overlay a transparent <canvas> over it and paint glyphs every frame.
 *
 * - Typewriter reveals over `duration` ms.
 * - Optional drop-cap rect is taken from `dropCapRef.current`. Lines
 *   that vertically overlap the cap shift their start X past it.
 * - Text obstacle test: ask CursorCharacter to redraw its silhouette
 *   into a 640×640 profile canvas, sample alpha per row, and split
 *   each line into left/right segments around the blocked X range.
 */
export function CanvasReflowText({
  text,
  font = "300 20px Inter, ui-sans-serif, system-ui, sans-serif",
  lineHeight = 34,
  color = "currentColor",
  duration = 4000,
  startDelay = 350,
  dropCapRef,
  className = "",
}: {
  text: string;
  font?: string;
  lineHeight?: number;
  color?: string;
  duration?: number;
  startDelay?: number;
  dropCapRef?: React.RefObject<HTMLElement>;
  className?: string;
}) {
  const hostRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const host = hostRef.current;
    const canvas = canvasRef.current;
    if (!host || !canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    let dpr = window.devicePixelRatio || 1;
    let W = host.clientWidth;
    let H = host.clientHeight;

    const resize = () => {
      dpr = window.devicePixelRatio || 1;
      W = host.clientWidth;
      H = host.clientHeight;
      canvas.width = Math.round(W * dpr);
      canvas.height = Math.round(H * dpr);
      canvas.style.width = `${W}px`;
      canvas.style.height = `${H}px`;
    };
    resize();
    const ro = new ResizeObserver(resize);
    ro.observe(host);

    const prepared: PreparedTextWithSegments = prepareWithSegments(text, font);

    // Resolve the actual rendered text color so canvas glyphs match.
    const cs = getComputedStyle(host).color;

    const PROFILE_SIZE = 640;
    let textOpacity = 0;
    const startedAt = performance.now();

    let raf = 0;
    const tick = (now: number) => {
      // Typewriter budget.
      const elapsed = Math.max(0, now - startedAt - startDelay);
      const reveal = Math.min(1, elapsed / duration);
      const charBudget = Math.floor(reveal * text.length);
      const targetOp = reveal > 0 ? 1 : 0;
      textOpacity += (targetOp - textOpacity) * 0.12;

      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, W, H);
      ctx.font = font;
      ctx.textBaseline = "alphabetic";
      ctx.fillStyle = cs || color;
      ctx.globalAlpha = textOpacity;

      // ---- Build sprite profile in screen space ----
      const hostRect = host.getBoundingClientRect();
      let spriteBox: {
        left: number; top: number; right: number; bottom: number;
        rows: { left: number; right: number }[];
      } | null = null;

      if (
        cursorCharacter.active &&
        cursorCharacter.profileCanvas &&
        cursorCharacter.redrawProfile
      ) {
        const drawSize = cursorCharacter.drawSize;
        cursorCharacter.redrawProfile(drawSize);
        const pc = cursorCharacter.profileCanvas;
        const pctx = pc.getContext("2d");
        if (pctx) {
          // Sample only the bounding band of the sprite to keep this cheap.
          // The silhouette was drawn centered at (PROFILE_SIZE/2, PROFILE_SIZE/2)
          // with the same pivot/tilt/flip; rough half-extent = drawSize.
          const half = Math.ceil(drawSize * 0.9);
          const cx = pc.width / 2;
          const cy = pc.height / 2;
          const sx0 = Math.max(0, cx - half);
          const sy0 = Math.max(0, cy - half);
          const sw = Math.min(pc.width - sx0, half * 2);
          const sh = Math.min(pc.height - sy0, half * 2);
          const img = pctx.getImageData(sx0, sy0, sw, sh);
          const rows: { left: number; right: number }[] = [];
          let minY = Infinity;
          let maxY = -Infinity;
          let minX = Infinity;
          let maxX = -Infinity;
          for (let y = 0; y < sh; y++) {
            let l = -1;
            let r = -1;
            for (let x = 0; x < sw; x++) {
              const a = img.data[(y * sw + x) * 4 + 3];
              if (a > 32) {
                if (l < 0) l = x;
                r = x;
              }
            }
            if (l >= 0) {
              const screenY = sy0 + y - cy + cursorCharacter.y - hostRect.top;
              const screenL = sx0 + l - cx + cursorCharacter.x - hostRect.left;
              const screenR = sx0 + r - cx + cursorCharacter.x - hostRect.left;
              rows.push({ left: screenL, right: screenR });
              if (screenY < minY) minY = screenY;
              if (screenY > maxY) maxY = screenY;
              if (screenL < minX) minX = screenL;
              if (screenR > maxX) maxX = screenR;
            } else {
              rows.push({ left: NaN, right: NaN });
            }
          }
          if (rows.length && isFinite(minY)) {
            spriteBox = {
              left: minX,
              right: maxX,
              top: minY,
              bottom: maxY,
              rows,
            };
            (spriteBox as any).originY = sy0 - cy + cursorCharacter.y - hostRect.top;
          }
        }
      }

      // ---- Drop cap rect in host-local coords ----
      let cap: { left: number; top: number; right: number; bottom: number } | null = null;
      if (dropCapRef?.current) {
        const r = dropCapRef.current.getBoundingClientRect();
        cap = {
          left: r.left - hostRect.left,
          top: r.top - hostRect.top,
          right: r.right - hostRect.left,
          bottom: r.bottom - hostRect.top,
        };
      }

      // ---- Walk lines ----
      let cursor: LayoutCursor = { segmentIndex: 0, graphemeIndex: 0 };
      let y = lineHeight;
      let charsDrawn = 0;

      while (y < H + lineHeight) {
        const lineTop = y - lineHeight + 6;
        const lineBottom = y + 4;

        // Available x range — start with the host width, shrink past the cap
        // for any line that vertically overlaps it.
        let lineLeft = 0;
        let lineRight = W;
        if (cap && lineBottom > cap.top && lineTop < cap.bottom) {
          lineLeft = Math.max(lineLeft, cap.right + 6);
        }

        // Compute occlusion gap from sprite rows that fall in this line.
        let gapL = NaN;
        let gapR = NaN;
        if (spriteBox && lineBottom > spriteBox.top && lineTop < spriteBox.bottom) {
          const originY = (spriteBox as any).originY as number;
          const r0 = Math.max(0, Math.floor(lineTop - originY));
          const r1 = Math.min(spriteBox.rows.length, Math.ceil(lineBottom - originY));
          let lMin = Infinity;
          let rMax = -Infinity;
          for (let r = r0; r < r1; r++) {
            const row = spriteBox.rows[r];
            if (!row || isNaN(row.left)) continue;
            if (row.left < lMin) lMin = row.left;
            if (row.right > rMax) rMax = row.right;
          }
          if (isFinite(lMin)) {
            gapL = lMin - 4;
            gapR = rMax + 4;
          }
        }

        const drawSegment = (segLeft: number, segRight: number) => {
          const w = segRight - segLeft;
          if (w <= 4) return;
          const line = layoutNextLine(prepared, cursor, w);
          if (!line) return;
          // Respect typewriter budget — clip the line text by remaining chars.
          const remaining = Math.max(0, charBudget - charsDrawn);
          const visible = line.text.slice(0, Math.min(line.text.length, remaining));
          if (visible.length > 0) {
            ctx.fillText(visible, segLeft, y);
          }
          charsDrawn += line.text.length;
          cursor = line.end;
        };

        if (!isFinite(gapL)) {
          drawSegment(lineLeft, lineRight);
        } else {
          // Left segment up to the gap.
          const leftRight = Math.min(lineRight, gapL);
          if (leftRight - lineLeft > 4) {
            drawSegment(lineLeft, leftRight);
          }
          // Right segment after the gap, only if it's wide enough.
          const rightLeft = Math.max(lineLeft, gapR);
          if (lineRight - rightLeft > 40) {
            drawSegment(rightLeft, lineRight);
          }
        }

        y += lineHeight;
        // Stop walking once pretext has consumed all segments.
        if (
          cursor.segmentIndex >= (prepared as any).segments.length &&
          charsDrawn >= text.length
        ) {
          break;
        }
        if (charsDrawn > text.length + 50) break;
      }

      ctx.globalAlpha = 1;
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);

    return () => {
      cancelAnimationFrame(raf);
      ro.disconnect();
    };
  }, [text, font, lineHeight, duration, startDelay, color, dropCapRef]);

  return (
    <div
      ref={hostRef}
      className={className}
      style={{ position: "relative", width: "100%", height: "100%", color }}
    >
      <canvas
        ref={canvasRef}
        aria-hidden
        className="pointer-events-none absolute inset-0"
      />
    </div>
  );
}
