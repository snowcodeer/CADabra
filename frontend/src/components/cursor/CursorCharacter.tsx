import { useEffect, useRef } from "react";
import { createPortal } from "react-dom";
import lottie, { type AnimationItem } from "lottie-web";
import wrenchLottie from "@/assets/wrench-turning.lottie.json";
import { cursorCharacter } from "./cursorCharacterState";

/**
 * Global, full-viewport canvas overlay that renders a Lottie-driven
 * "character" silhouette gliding toward the cursor.
 *
 * - Off-screen Lottie buffer (renderer:"canvas") is advanced by
 *   wall-clock time and painted as a uniform near-black silhouette.
 * - The sprite trails the cursor with smoothing, leans toward it via
 *   atan2, and mirrors horizontally based on travel direction.
 * - Anchored by a normalized pivot so the broom/jaw tip rides the
 *   actual cursor point, not the image top-left.
 * - On coarse pointers, an on-screen joystick drives the sprite.
 *
 * Pointer-events are disabled so all clicks fall through to the page.
 */

const PIVOT_X = 0.59; // wrench head sits roughly 60% across the art
const PIVOT_Y = 0.34;
const BUFFER_W = 240;
const BUFFER_H = (240 * 696) / 1237;
const DRAW_SIZE = 110; // CSS pixels — visible silhouette footprint
const SILHOUETTE_COLOR = "#111111";
const FOLLOW_LERP = 0.12;
const TILT_LERP = 0.18;
const FLIP_LERP = 0.18;
const FADE_HOVER_LERP = 0.18;

export function CursorCharacter() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const offscreenHostRef = useRef<HTMLDivElement>(null);
  const joystickRef = useRef<HTMLDivElement>(null);
  const joystickThumbRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    const offscreenHost = offscreenHostRef.current;
    if (!canvas || !offscreenHost) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    let dpr = window.devicePixelRatio || 1;
    let W = window.innerWidth;
    let H = window.innerHeight;

    const resize = () => {
      dpr = window.devicePixelRatio || 1;
      W = window.innerWidth;
      H = window.innerHeight;
      canvas.width = Math.round(W * dpr);
      canvas.height = Math.round(H * dpr);
      canvas.style.width = `${W}px`;
      canvas.style.height = `${H}px`;
    };
    resize();
    window.addEventListener("resize", resize);

    // ---- Lottie buffer ----
    const anim: AnimationItem = lottie.loadAnimation({
      container: offscreenHost,
      renderer: "canvas",
      loop: false,
      autoplay: false,
      animationData: wrenchLottie,
      rendererSettings: {
        clearCanvas: true,
        preserveAspectRatio: "xMidYMid meet",
      },
    });

    let lottieCanvas: HTMLCanvasElement | null = null;
    const onLoaded = () => {
      lottieCanvas = offscreenHost.querySelector("canvas");
    };
    anim.addEventListener("DOMLoaded", onLoaded);

    // ---- Silhouette scratch ----
    const scratch = document.createElement("canvas");
    scratch.width = BUFFER_W;
    scratch.height = BUFFER_H;
    const scratchCtx = scratch.getContext("2d")!;

    // Off-screen profile canvas re-used by text reflow consumers.
    const profile = document.createElement("canvas");
    profile.width = 640;
    profile.height = 640;
    const profileCtx = profile.getContext("2d")!;
    cursorCharacter.profileCanvas = profile;

    // ---- Pointer / joystick state ----
    let mouseX = window.innerWidth * 0.54;
    let mouseY = window.innerHeight * 0.5;
    let hasMouse = false;
    let currentX = mouseX;
    let currentY = mouseY;
    let smoothTilt = 0;
    let flipScale = 1;
    let opacity = 0;
    let hoverFade = 1;

    const isCoarse = window.matchMedia("(pointer: coarse)").matches;
    let joyX = 0;
    let joyY = 0;
    const JOY_SPEED = 360; // px/sec at full deflection

    const onMove = (e: MouseEvent) => {
      mouseX = e.clientX;
      mouseY = e.clientY;
      hasMouse = true;
      // Hover-fade when the pointer is over interactive elements.
      const t = e.target as Element | null;
      hoverFade = t && t.closest("a, button, [role=button], input, textarea, select")
        ? 0.18
        : 1;
    };
    const onLeave = () => {
      hasMouse = false;
    };
    window.addEventListener("mousemove", onMove);
    window.addEventListener("mouseleave", onLeave);

    // Joystick (touch) — basic radial pad.
    const joystick = joystickRef.current;
    const thumb = joystickThumbRef.current;
    const JOY_RADIUS = 48;
    let joyTouchId: number | null = null;
    let joyOriginX = 0;
    let joyOriginY = 0;
    const onTouchStart = (e: TouchEvent) => {
      if (joyTouchId !== null) return;
      const t = e.changedTouches[0];
      joyTouchId = t.identifier;
      const r = (joystick as HTMLDivElement).getBoundingClientRect();
      joyOriginX = r.left + r.width / 2;
      joyOriginY = r.top + r.height / 2;
    };
    const onTouchMove = (e: TouchEvent) => {
      if (joyTouchId === null) return;
      for (const t of Array.from(e.changedTouches)) {
        if (t.identifier !== joyTouchId) continue;
        const dx = t.clientX - joyOriginX;
        const dy = t.clientY - joyOriginY;
        const d = Math.hypot(dx, dy);
        const k = d > JOY_RADIUS ? JOY_RADIUS / d : 1;
        const tx = dx * k;
        const ty = dy * k;
        joyX = tx / JOY_RADIUS;
        joyY = ty / JOY_RADIUS;
        if (thumb) thumb.style.transform = `translate(${tx}px, ${ty}px)`;
      }
    };
    const onTouchEnd = (e: TouchEvent) => {
      for (const t of Array.from(e.changedTouches)) {
        if (t.identifier === joyTouchId) {
          joyTouchId = null;
          joyX = 0;
          joyY = 0;
          if (thumb) thumb.style.transform = `translate(0, 0)`;
        }
      }
    };
    if (isCoarse && joystick) {
      joystick.addEventListener("touchstart", onTouchStart, { passive: true });
      joystick.addEventListener("touchmove", onTouchMove, { passive: true });
      joystick.addEventListener("touchend", onTouchEnd);
      joystick.addEventListener("touchcancel", onTouchEnd);
    }

    // ---- Render loop ----
    const totalFrames = anim.totalFrames;
    const frameRate = anim.frameRate;
    const startedAt = performance.now();
    let last = startedAt;
    let raf = 0;

    const drawSilhouetteInto = (
      targetCtx: CanvasRenderingContext2D,
      drawSize: number,
      tilt: number,
      flip: number,
    ) => {
      if (!lottieCanvas) return;
      // Refresh the silhouette scratch from the current Lottie buffer.
      scratchCtx.globalCompositeOperation = "source-over";
      scratchCtx.clearRect(0, 0, BUFFER_W, BUFFER_H);
      scratchCtx.drawImage(lottieCanvas, 0, 0, BUFFER_W, BUFFER_H);
      scratchCtx.globalCompositeOperation = "source-in";
      scratchCtx.fillStyle = SILHOUETTE_COLOR;
      scratchCtx.fillRect(0, 0, BUFFER_W, BUFFER_H);
      scratchCtx.globalCompositeOperation = "source-over";

      const w = drawSize;
      const h = drawSize * (BUFFER_H / BUFFER_W);
      targetCtx.save();
      targetCtx.scale(flip, 1);
      targetCtx.rotate(tilt);
      targetCtx.drawImage(scratch, -w * PIVOT_X, -h * PIVOT_Y, w, h);
      targetCtx.restore();
    };

    // Public hook for text-reflow consumers: render the silhouette
    // into the profile buffer at a requested draw size, centered.
    cursorCharacter.redrawProfile = (size: number) => {
      profileCtx.setTransform(1, 0, 0, 1, 0, 0);
      profileCtx.clearRect(0, 0, profile.width, profile.height);
      profileCtx.translate(profile.width / 2, profile.height / 2);
      drawSilhouetteInto(profileCtx, size, smoothTilt, flipScale);
      cursorCharacter.profileDrawSize = size;
    };

    const tick = (now: number) => {
      const dt = Math.min((now - last) / 1000, 0.05);
      last = now;

      // Advance lottie playback by wall-clock time.
      const elapsedSec = (now - startedAt) / 1000;
      const frame = Math.floor((elapsedSec * frameRate) % totalFrames);
      anim.goToAndStop(frame, true);

      // Joystick mode adjusts mouseX/Y directly.
      if (isCoarse) {
        mouseX = Math.max(0, Math.min(W, mouseX + joyX * JOY_SPEED * dt));
        mouseY = Math.max(0, Math.min(H, mouseY + joyY * JOY_SPEED * dt));
        hasMouse = true;
      }

      // Smooth follow.
      const targetX = hasMouse ? mouseX : W * 0.54;
      const targetY = hasMouse ? mouseY : H * 0.5;
      currentX += (targetX - currentX) * FOLLOW_LERP;
      currentY += (targetY - currentY) * FOLLOW_LERP;

      // Lean / facing.
      const dx = targetX - currentX;
      const dy = targetY - currentY;
      const dist = Math.hypot(dx, dy);
      if (dist > 4) {
        const goingRight = dx >= 0;
        const targetFlip = goingRight ? 1 : -1;
        flipScale += (targetFlip - flipScale) * FLIP_LERP;
        // Flip dy sign with facing so the lean reads as "tilting toward
        // the cursor" regardless of which way the figure faces.
        const targetTilt = Math.atan2(dy, Math.abs(dx)) * 0.55;
        smoothTilt += (targetTilt - smoothTilt) * TILT_LERP;
      } else {
        smoothTilt += (0 - smoothTilt) * TILT_LERP;
      }

      // Fade in once visible; respect hover-over-controls dim.
      const targetOpacity = (hasMouse ? 1 : 0) * hoverFade;
      opacity += (targetOpacity - opacity) * FADE_HOVER_LERP;

      // Publish runtime state for consumers (text reflow, etc.).
      cursorCharacter.active = hasMouse;
      cursorCharacter.x = currentX;
      cursorCharacter.y = currentY;
      cursorCharacter.drawSize = DRAW_SIZE;
      cursorCharacter.flipScale = flipScale;
      cursorCharacter.tilt = smoothTilt;
      cursorCharacter.pivotX = PIVOT_X;
      cursorCharacter.pivotY = PIVOT_Y;

      // ---- Frame paint ----
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, W, H);
      ctx.globalAlpha = opacity;
      ctx.translate(currentX, currentY);
      drawSilhouetteInto(ctx, DRAW_SIZE, smoothTilt, flipScale);
      ctx.globalAlpha = 1;

      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);

    return () => {
      cancelAnimationFrame(raf);
      window.removeEventListener("resize", resize);
      window.removeEventListener("mousemove", onMove);
      window.removeEventListener("mouseleave", onLeave);
      if (isCoarse && joystick) {
        joystick.removeEventListener("touchstart", onTouchStart);
        joystick.removeEventListener("touchmove", onTouchMove);
        joystick.removeEventListener("touchend", onTouchEnd);
        joystick.removeEventListener("touchcancel", onTouchEnd);
      }
      anim.removeEventListener("DOMLoaded", onLoaded);
      anim.destroy();
      cursorCharacter.profileCanvas = null;
      cursorCharacter.redrawProfile = null;
      cursorCharacter.active = false;
    };
  }, []);

  const isCoarse =
    typeof window !== "undefined" &&
    window.matchMedia("(pointer: coarse)").matches;

  return createPortal(
    <>
      <canvas
        ref={canvasRef}
        aria-hidden
        className="pointer-events-none fixed inset-0 z-[9999]"
        style={{ background: "transparent" }}
      />
      {/* Off-screen lottie host — rendered far off-page so it doesn't
          paint, but still produces a canvas we can sample from. */}
      <div
        ref={offscreenHostRef}
        aria-hidden
        style={{
          position: "fixed",
          left: -9999,
          top: -9999,
          width: BUFFER_W,
          height: BUFFER_H,
          pointerEvents: "none",
        }}
      />
      {isCoarse && (
        <div
          ref={joystickRef}
          aria-hidden
          className="fixed bottom-6 right-6 z-[9998] touch-none rounded-full border border-foreground/20 bg-foreground/10 backdrop-blur-md"
          style={{ width: 120, height: 120 }}
        >
          <div
            ref={joystickThumbRef}
            className="absolute left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2 rounded-full bg-foreground/70"
            style={{ width: 48, height: 48, transition: "none" }}
          />
        </div>
      )}
    </>,
    document.body,
  );
}
