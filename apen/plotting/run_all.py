from pathlib import Path
import argparse
import os
import subprocess
import sys

HERE = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(description='Redraw all six manuscript figures at 600 dpi from the released data.')
parser.add_argument('--data-root', type=Path, help='Directory containing analysis_data/ and server/ from the extracted reproducibility ZIP.')
parser.add_argument('--output', type=Path, help='Output directory; defaults to ../figures or APEN_FIGURE_OUTPUT.')
parser.add_argument('--map-root', type=Path, help='Natural Earth directory with physical/ and cultural/ subdirectories.')
args = parser.parse_args()
env = os.environ.copy()
for name, value in [('APEN_DATA_ROOT', args.data_root), ('APEN_FIGURE_OUTPUT', args.output), ('APEN_MAP_ROOT', args.map_root)]:
    if value is not None:
        env[name] = str(value.resolve())
for script in ['build_figure1.py', 'build_figure2.py', 'build_figure3.py', 'build_figure4.py', 'build_supplementary_figures.py']:
    print(f'Rendering {script}', flush=True)
    subprocess.run([sys.executable, '-X', 'utf8', str(HERE / script)], env=env, check=True)
