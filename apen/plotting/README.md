# Reproduce the manuscript figures

These scripts reproduce the four main figures and Supplementary Figures S1 and S2 from the released numerical results. They preserve the manuscript geometry, colors, axes, labels, and statistical definitions. Each figure is exported as a 600-dpi PNG, RGB TIFF with LZW compression, and PDF with vector text and 600-dpi raster data layers.

## Installation and execution

Use Python 3.12 and install the dependencies listed in `requirements.txt`:

```sh
python -m pip install -r plotting/requirements.txt
```

The manuscript plots use Arial. Install Arial through a licensed operating-system/font distribution before running the scripts; the font itself is not redistributed. The scripts check that Arial is available and stop if it is missing.

Extract `APEN_Reproducibility_20260927.zip` into a directory that directly contains `analysis_data/` and `server/`, then run:

```sh
python plotting/run_all.py --data-root /path/to/extracted-data --output /path/to/figures
```

With the reproducibility archive extracted to `data/` next to `plotting/`, `python plotting/run_all.py` uses this directory automatically. The environment variables `APEN_DATA_ROOT`, `APEN_FIGURE_OUTPUT`, and `APEN_MAP_ROOT` provide the same settings. Each `build_figure*.py` can also run independently with these environment variables.

## Numerical inputs

| Figure | Inputs in the extracted reproducibility archive | Additional inputs included here |
|---|---|---|
| 1 | `server/final_consistency/outputs/historical_annual.csv`; `historical_sites.csv` in the same directory | `inputs/wp9c_farm_metrics.csv`; Natural Earth land and border geometry |
| 2 | `analysis_data/mechanism_sites.csv`; `analysis_data/concentration_correlations.csv`; `server/final_consistency/outputs/standard_baseline.npz`; `server/outputs/models/baseline/F066_5deg.npz`; `server/inputs/legacy_eta.npz` | None |
| 3 | `server/final_consistency/outputs/standard_baseline.npz`; `analysis_data/baseline_sites.csv` | `inputs/wp9c_farm_metrics.csv`; Natural Earth land and border geometry |
| 4 | `analysis_data/planning_projects.csv`; `analysis_data/model_summary.csv`; `analysis_data/planning_group_years.csv` | None |
| S1 | None | `inputs/wp4c_jensen.npz`, using its `xy` and `pid` arrays |
| S2 | `analysis_data/model_summary_screened.csv`; `analysis_data/mechanism_sites.csv` | None |

`wp9c_farm_metrics.csv` supplies site locations, country labels, and the coastal comparison-group membership. `wp4c_jensen.npz` supplies the six standard-layout coordinate arrays. `legacy_eta.npz` supplies the zero-wake reference-power vector `P0` used to convert F66 modeled power to farm efficiency. The Figure 2 script validates the wind-speed grid and excludes zero-reference-power bins from the efficiency calculation.

The numerical plotting checks retain all site records, the negative energy contributions, the original temporal selection/evaluation design, and the same regional groups. No wake simulation is rerun by these plotting scripts. The 13 exported source-data CSVs can be compared directly with the released figure data.

## Map inputs

The exact Natural Earth 1:50m land and administrative boundary files used in Figures 1 and 3 are included under `inputs/natural_earth/`. These map data are in the public domain under the [Natural Earth terms of use](https://www.naturalearthdata.com/about/terms-of-use/). They can be replaced by a user-supplied directory through `--map-root`, preserving the `physical/` and `cultural/` subdirectories. The original maps use longitude/latitude coordinates displayed with the same Plate Carrée geometry; changing the boundary version may change the appearance of coastlines.

## Output and checks

Output names are `Figure_1_600dpi`, `Figure_2_600dpi`, `Figure_3_600dpi`, `Figure_4_600dpi`, `Figure_S1_600dpi`, and `Figure_S2_600dpi`, each with `.png`, `.tif`, and `.pdf` extensions. The scripts also write figure-data CSVs and numerical validation records. `inputs_manifest.json` records the additional input checksums and provenance. The full scientific computations and data descriptions remain in the reproducibility archive and Supplementary Information.
