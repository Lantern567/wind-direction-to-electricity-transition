import json
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser(description='Recover full-precision terminal coordinates from DeepOWT feature IDs.')
parser.add_argument('--deepowt', required=True, type=Path)
parser.add_argument('--coordinates', required=True, type=Path)
parser.add_argument('--output', required=True, type=Path)
options = parser.parse_args()
features = json.loads(options.deepowt.read_text(encoding='utf-8'))['features']
tc = pd.read_csv(options.coordinates)
tc = tc.merge(tc.groupby('farm_id')['year'].max().reset_index(), on=['farm_id', 'year'], validate='many_to_one')
xy = np.array([features[int(tid)-1]['geometry']['coordinates'] for tid in tc.turbine_id])
assert np.max(np.abs(xy-tc[['lon', 'lat']].to_numpy())) < 5.01e-7
tc['lon'], tc['lat'] = xy[:, 0], xy[:, 1]
tc.to_csv(options.output, index=False)
print('verified one-based feature IDs', len(tc))
