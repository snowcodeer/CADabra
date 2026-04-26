/**
 * Shared, mutable runtime state for the global cursor character overlay.
 *
 * `CursorCharacter` writes the current sprite footprint here every
 * animation frame. Other components (e.g. the Workflow canvas text
 * reflow) read it to wrap text around the live silhouette.
 */
export type CursorCharacterState = {
  /** True after the user has moved the pointer at least once. */
  active: boolean;
  /** Smoothed sprite anchor in viewport (CSS) pixels. */
  x: number;
  y: number;
  /** On-screen draw size of the sprite (CSS pixels). */
  drawSize: number;
  /** Horizontal flip currently being applied (-1..1). */
  flipScale: number;
  /** Smoothed lean angle in radians. */
  tilt: number;
  /** Normalized broom/anchor pivot inside the sprite (0..1). */
  pivotX: number;
  pivotY: number;
  /**
   * Off-screen scanline-profile buffer. Components that need pixel-perfect
   * wrap can call `redrawProfile(size)` to render the silhouette into this
   * buffer at a known size and then sample the alpha mask.
   */
  redrawProfile: ((size: number) => void) | null;
  /** Profile canvas — same alpha shape as the visible silhouette. */
  profileCanvas: HTMLCanvasElement | null;
  /** Size at which the silhouette was drawn into `profileCanvas`. */
  profileDrawSize: number;
};

export const cursorCharacter: CursorCharacterState = {
  active: false,
  x: -9999,
  y: -9999,
  drawSize: 0,
  flipScale: 1,
  tilt: 0,
  // Match `CursorCharacter`: Nut layer center in wrench Lottie comp (1237×696).
  pivotX: 838.5 / 1237,
  pivotY: 239 / 696,
  redrawProfile: null,
  profileCanvas: null,
  profileDrawSize: 0,
};
