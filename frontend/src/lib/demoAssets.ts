import cloud000035Url from "../../deepcad_pcrecon_stl/deepcadimg_000035_recon_noisy.stl?url";
import cloud002354Url from "../../deepcad_pcrecon_stl/deepcadimg_002354_recon_noisy.stl?url";
import cloud117514Url from "../../deepcad_pcrecon_stl/deepcadimg_117514_recon_noisy.stl?url";
import cloud128105Url from "../../deepcad_pcrecon_stl/deepcadimg_128105_recon_noisy.stl?url";

import groundTruth000035Url from "../../deepcad_stl/deepcadimg_000035.stl?url";
import groundTruth002354Url from "../../deepcad_stl/deepcadimg_002354.stl?url";
import groundTruth117514Url from "../../deepcad_stl/deepcadimg_117514.stl?url";
import groundTruth128105Url from "../../deepcad_stl/deepcadimg_128105.stl?url";

import orthoGrid000035Url from "../../public/demos/ortho_deepcadimg_000035_recon_grid.png?url";
import orthoGrid002354Url from "../../public/demos/ortho_deepcadimg_002354_recon_grid.png?url";
import orthoGrid117514Url from "../../public/demos/ortho_deepcadimg_117514_recon_grid.png?url";
import orthoGrid128105Url from "../../public/demos/ortho_deepcadimg_128105_recon_grid.png?url";

export const demoAssets = {
  deepcadimg_000035: {
    cloudStl: cloud000035Url,
    groundTruthStl: groundTruth000035Url,
    orthoGrid: orthoGrid000035Url,
  },
  deepcadimg_002354: {
    cloudStl: cloud002354Url,
    groundTruthStl: groundTruth002354Url,
    orthoGrid: orthoGrid002354Url,
  },
  deepcadimg_117514: {
    cloudStl: cloud117514Url,
    groundTruthStl: groundTruth117514Url,
    orthoGrid: orthoGrid117514Url,
  },
  deepcadimg_128105: {
    cloudStl: cloud128105Url,
    groundTruthStl: groundTruth128105Url,
    orthoGrid: orthoGrid128105Url,
  },
} as const;

export const sample000035Assets = demoAssets.deepcadimg_000035;
