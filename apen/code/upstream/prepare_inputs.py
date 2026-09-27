import hashlib
import argparse
import json
import shutil
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(description='Rebuild the original input inventory from an explicitly supplied source tree; legacy intermediate files are required.')
parser.add_argument('--source-root', required=True, type=Path)
parser.add_argument('--output', required=True, type=Path)
options = parser.parse_args()
REPO = options.source_root.resolve()
OUT = options.output.resolve()
INPUTS = OUT / 'inputs'
INPUTS.mkdir(parents=True, exist_ok=True)
sources = {
    'turbine_coordinates.csv': 'offshore-task0-HuTingxian/output/task0/turbine_coordinates.csv',
    'farms_master.csv': 'offshore-task0-HuTingxian/output/task0/farms_master.csv',
    'geometry.csv': '重算交付_20260829/output-new/wp1_geometry_frozen.csv',
    'legacy_eta.npz': '重算交付_20260829/output-new/wp7a_real_eta.npz',
    'legacy_curves.npz': '重算交付_20260829/output-new/wp7a_real_curves.npz',
    'legacy_weights.npz': '重算交付_20260829/output-new/wp3_climate_weights.npz',
    'legacy_scenarios.csv': '重算交付_20260829/output-new/wp7c_scenario_table.csv',
    'legacy_floris_config.py': 'offshore-task2/floris_config.py',
    'legacy_weather_code.py': '补算/补算/wp3_climate_joint.py',
}
manifest = []
for name, source in sources.items():
    path = REPO / source
    shutil.copy2(path, INPUTS / name)
    manifest.append({'name': name, 'source': source,
                     'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
z = np.load(INPUTS / 'legacy_curves.npz')
idx = np.argmax(z['E_y'][:, :6, :].mean(axis=1), axis=1)
angles = pd.DataFrame({'farm_id': z['farm_ids'], 'n_turb': z['n_turb'],
                       'selected_index': idx, 'selected_angle_deg': idx * 5})
angles.to_csv(INPUTS / 'frozen_angles.csv', index=False)
(INPUTS / 'source_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
with zipfile.ZipFile(OUT / 'inputs.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
    for path in sorted(INPUTS.iterdir()):
        archive.write(path, 'inputs/' + path.name)
print(json.dumps({'archive': str(OUT / 'inputs.zip'), 'bytes': (OUT / 'inputs.zip').stat().st_size,
                  'farms': len(angles), 'planning_projects': int((angles.n_turb >= 10).sum())}))
