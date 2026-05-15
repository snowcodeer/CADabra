import cloud000035Url from "../../deepcad_pcrecon_stl/deepcadimg_000035_recon_noisy.stl?url";
import cloud002354Url from "../../deepcad_pcrecon_stl/deepcadimg_002354_recon_noisy.stl?url";
import cloud117514Url from "../../deepcad_pcrecon_stl/deepcadimg_117514_recon_noisy.stl?url";
import cloud128105Url from "../../deepcad_pcrecon_stl/deepcadimg_128105_recon_noisy.stl?url";

import groundTruth000035Url from "../../deepcad_stl/deepcadimg_000035.stl?url";
import groundTruth002354Url from "../../deepcad_stl/deepcadimg_002354.stl?url";
import groundTruth117514Url from "../../deepcad_stl/deepcadimg_117514.stl?url";
import groundTruth128105Url from "../../deepcad_stl/deepcadimg_128105.stl?url";

import orthoGrid000035Url from "../demos/ortho_deepcadimg_000035_recon_grid.png?url";
import orthoGrid002354Url from "../demos/ortho_deepcadimg_002354_recon_grid.png?url";
import orthoGrid117514Url from "../demos/ortho_deepcadimg_117514_recon_grid.png?url";
import orthoGrid128105Url from "../demos/ortho_deepcadimg_128105_recon_grid.png?url";

import stepOffs000035 from "../demos/ortho_deepcadimg_000035_step_offs.json";
import stepOffs002354 from "../demos/ortho_deepcadimg_002354_step_offs.json";
import stepOffs117514 from "../demos/ortho_deepcadimg_117514_step_offs.json";
import stepOffs128105 from "../demos/ortho_deepcadimg_128105_step_offs.json";

export type StepOffAxis = "X" | "Y" | "Z";
export type StepOffKind = "external" | "internal" | "through";
export type StepOffDirection = "up" | "down";

export interface StepOff {
  id: string;
  view: string;
  axis: StepOffAxis;
  kind: StepOffKind;
  step_direction: StepOffDirection;
  depth_mm: number;
  confidence: number;
  outer_bbox_px: [number, number, number, number];
  inner_bbox_px: [number, number, number, number];
  outer_polygon_px: [number, number][];
  inner_polygon_px: [number, number][];
  notes: string;
}

export interface StepOffAudit {
  sample_id: string;
  axis_picked: StepOffAxis;
  external_tallies: Record<StepOffAxis, number>;
  step_offs: StepOff[];
}

export const demoAssets = {
  deepcadimg_000035: {
    cloudStl: cloud000035Url,
    groundTruthStl: groundTruth000035Url,
    orthoGrid: orthoGrid000035Url,
    stepOffs: stepOffs000035 as StepOffAudit,
  },
  deepcadimg_002354: {
    cloudStl: cloud002354Url,
    groundTruthStl: groundTruth002354Url,
    orthoGrid: orthoGrid002354Url,
    stepOffs: stepOffs002354 as StepOffAudit,
  },
  deepcadimg_117514: {
    cloudStl: cloud117514Url,
    groundTruthStl: groundTruth117514Url,
    orthoGrid: orthoGrid117514Url,
    stepOffs: stepOffs117514 as StepOffAudit,
  },
  deepcadimg_128105: {
    cloudStl: cloud128105Url,
    groundTruthStl: groundTruth128105Url,
    orthoGrid: orthoGrid128105Url,
    stepOffs: stepOffs128105 as StepOffAudit,
  },
} as const;

export const sample000035Assets = demoAssets.deepcadimg_000035;
