from pathlib import Path
import os
from PIL import Image
from matplotlib import font_manager

HERE = Path(__file__).resolve().parent
DATA_ROOT = Path(os.environ.get('APEN_DATA_ROOT', str(HERE.parent / 'data'))).resolve()
OUTPUT_ROOT = Path(os.environ.get('APEN_FIGURE_OUTPUT', str(HERE.parent / 'figures'))).resolve()
PLOT_INPUTS = HERE / 'inputs'
MAP_ROOT = Path(os.environ.get('APEN_MAP_ROOT', str(PLOT_INPUTS / 'natural_earth'))).resolve()
assert DATA_ROOT.is_dir(), f'Extract the reproducibility ZIP and set APEN_DATA_ROOT: {DATA_ROOT}'
font_manager.findfont('Arial', fallback_to_default=False)
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

def save_600dpi(fig, stem):
    png_path = OUTPUT_ROOT / f'{stem}.png'
    fig.savefig(png_path, dpi=600, facecolor='white')
    fig.savefig(OUTPUT_ROOT / f'{stem}.pdf', dpi=600, facecolor='white')
    with Image.open(png_path) as image:
        image.convert('RGB').save(OUTPUT_ROOT / f'{stem}.tif', compression='tiff_lzw', dpi=(600, 600))
