import json
from pathlib import Path
from datasets import load_dataset

selected_ids = [
    9,11,17,28,35,39,41,42,43,55,60,78,
    33172,41850,69872,37407,61490,73806,67013,117514,11860,128105,
    65120,14528,64437,2354,37782,108596,53042,78812,52079,58993,
    30107,89073,97805,108680,58137,81945,4397,110917,57780,28804,
    67626,8104,81740,26528,14849,
]

out_dir = Path('/workspace/backend/sample_data/deepcad_selected')
out_dir.mkdir(parents=True, exist_ok=True)

rows = load_dataset('CADCODER/DeepCAD-CQ-Vision-Paired', split='train')
manifest = []
for sid in selected_ids:
    idx = sid - 1
    row = rows[idx]
    code = row.get('code') or ''
    file = out_dir / f'deepcadimg_{sid:06d}.py'
    file.write_text(code)
    manifest.append({'id': sid, 'row_index': idx, 'code_file': file.name})

(Path('/workspace/backend/sample_data/deepcad_selected/manifest.json')).write_text(json.dumps(manifest, indent=2))
print('wrote', len(manifest), 'codes')
