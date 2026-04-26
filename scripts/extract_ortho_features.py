#!/usr/bin/env python3
"""Run the deterministic OpenCV segmenter on a cleaned 6-view PNG.

Use this to verify per-view polygon, flat-edge and through-hole detection
BEFORE the cross-view inferencer runs. It writes:

    backend/outputs/face_<sample>_recon_noisy_ortho_features.json
    backend/outputs/face_<sample>_recon_noisy_ortho_overlay.png

The overlay PNG draws:
  - red polygon = detected outer outline (per silhouette panel)
  - green dots  = polygon vertices
  - cyan box    = each detected interior through-hole / pocket
  - per-cell header line: vertex count, straight-flat count, hole count,
    depth-tier count

Usage:
    # by sample id (looks up the cleaned PNG via the recon manifest)
    python scripts/extract_ortho_features.py --sample-id deepcadimg_000035

    # or by direct PNG path
    python scripts/extract_ortho_features.py --png path/to/clean.png \
        --out-prefix backend/outputs/myrun
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from backend.ai_infra.ortho_view_segmenter import (  # noqa: E402
    render_debug_overlay,
    segment_ortho_png,
)

RECON_OUT_DIR = REPO_ROOT / "backend" / "outputs" / "deepcad_pc_recon_stl"
RECON_MANIFEST = RECON_OUT_DIR / "manifest.json"
DEFAULT_FACE_DIR = REPO_ROOT / "backend" / "outputs"


def _resolve_clean_png(sample_id: str) -> Path:
    if not RECON_MANIFEST.exists():
        raise SystemExit(
            f"manifest missing: {RECON_MANIFEST.relative_to(REPO_ROOT)}"
        )
    manifest = json.loads(RECON_MANIFEST.read_text())
    entry = next((e for e in manifest if e.get("sample_id") == sample_id), None)
    if entry is None:
        raise SystemExit(f"sample id not in manifest: {sample_id!r}")
    cohesion = entry.get("cohesion") or {}
    rel = cohesion.get("geometry_view_clean")
    if not rel:
        raise SystemExit(
            f"sample {sample_id!r} has no geometry_view_clean — run "
            "scripts/synthesize_clean_views.py first."
        )
    p = RECON_OUT_DIR / rel
    if not p.exists():
        raise SystemExit(f"clean PNG missing on disk: {p}")
    return p


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--sample-id", help="resolve clean PNG via the recon manifest")
    g.add_argument("--png", help="direct path to a cleaned 6-view PNG")
    p.add_argument(
        "--out-prefix",
        help="filename prefix for the JSON + overlay outputs. "
             "Defaults to backend/outputs/face_<stem>_ortho.",
    )
    p.add_argument(
        "--open", action="store_true",
        help="open the overlay PNG in Preview after writing (macOS only).",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()

    if args.sample_id:
        clean_png = _resolve_clean_png(args.sample_id)
        prefix = args.out_prefix or str(
            DEFAULT_FACE_DIR / f"face_{args.sample_id}_recon_noisy_ortho"
        )
    else:
        clean_png = Path(args.png).resolve()
        if not clean_png.exists():
            raise SystemExit(f"PNG not found: {clean_png}")
        prefix = args.out_prefix or str(
            DEFAULT_FACE_DIR / f"face_{clean_png.stem}_ortho"
        )

    print(f"segmenting {clean_png.relative_to(REPO_ROOT)}")
    feats = segment_ortho_png(clean_png)

    json_path = Path(f"{prefix}_features.json")
    overlay_path = Path(f"{prefix}_overlay.png")
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(feats.to_json())
    render_debug_overlay(feats, overlay_path)

    print()
    print(f"  features json: {json_path.relative_to(REPO_ROOT)}")
    print(f"  overlay png:   {overlay_path.relative_to(REPO_ROOT)}")
    print()
    print("per-view summary (counts from the canonical line/arc outline):")
    for name, v in feats.views.items():
        n_arcs = sum(1 for s in v.outline if s.kind == "arc")
        n_lines = sum(1 for s in v.outline if s.kind == "line")
        print(
            f"  {name:6}  arcs={n_arcs:2}  lines={n_lines:2}  "
            f"holes={len(v.interior_holes)}  "
            f"tiers={v.depth_tier_count}"
        )

    if args.open and sys.platform == "darwin":
        import subprocess
        subprocess.Popen(
            ["open", str(overlay_path)],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
