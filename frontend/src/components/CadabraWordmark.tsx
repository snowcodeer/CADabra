/**
 * Renders “CAD” with the capital A replaced by the brand mark (public/logo.png).
 * Outer + logo wrapper use `cadLetterClassName` for font metrics so the mark can use
 * `1cap` (true cap height) to match the uppercase C and D.
 */
const LOGO_WRAPPER =
  "inline-block shrink-0 [aspect-ratio:352/402] h-[1cap] w-auto translate-y-[1.5px] align-baseline [font:inherit]";
const LOGO_IMG = "h-full w-full object-contain [image-rendering:-webkit-optimize-contrast]";

export function CadabraCadLockup({
  cadLetterClassName,
  logoWrapperClassName = LOGO_WRAPPER,
  logoImgClassName = LOGO_IMG,
}: {
  cadLetterClassName: string;
  /** Wraps the PNG; default height is `1cap` for each usage’s font size. */
  logoWrapperClassName?: string;
  /** Applied to the `img` (fills the cap-height box). */
  logoImgClassName?: string;
}) {
  return (
    <span
      className={`inline-flex max-w-full items-baseline ${cadLetterClassName}`.trim()}
    >
      <span className={cadLetterClassName}>C</span>
      <span className={logoWrapperClassName} aria-hidden>
        <img
          src="/logo.png"
          alt=""
          width={352}
          height={402}
          decoding="async"
          draggable={false}
          className={logoImgClassName}
        />
      </span>
      <span className={cadLetterClassName}>D</span>
    </span>
  );
}
