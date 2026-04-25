import json
from pathlib import Path

import cadquery as cq
from cadquery import exporters

REPO_ROOT = Path(__file__).resolve().parent.parent
src_dir = REPO_ROOT / 'backend/sample_data/deepcad_selected'
out_dir = REPO_ROOT / 'backend/outputs/deepcad_selected_stl'
out_dir.mkdir(parents=True, exist_ok=True)

manifest = json.loads((src_dir / 'manifest.json').read_text())
results = []
for item in manifest:
    sid = item['id']
    code_file = src_dir / item['code_file']
    code = code_file.read_text()
    namespace = {'cq': cq}
    ok = False
    err = None
    try:
        exec(code, namespace, namespace)
        shape = namespace.get('result') or namespace.get('r') or namespace.get('assembly') or namespace.get('solid') or namespace.get('part')
        if shape is None:
            raise ValueError('No result/r/assembly object in code namespace')
        out_stl = out_dir / f'deepcadimg_{sid:06d}.stl'
        exporters.export(shape, str(out_stl))
        ok = out_stl.exists() and out_stl.stat().st_size > 0
        if not ok:
            err = 'export produced empty STL'
    except Exception as e:
        err = str(e)
    results.append({'id': sid, 'success': ok, 'error': err, 'stl_file': f'deepcadimg_{sid:06d}.stl'})

(out_dir / 'manifest.json').write_text(json.dumps(results, indent=2))
print('success', sum(1 for r in results if r['success']), 'of', len(results))
