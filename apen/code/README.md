# APEN reproducibility archive — 27 September 2026

This public package contains the numerical data and analysis code supporting the APEN manuscript. All archived energy values are simulations using common reference turbines. The scientific arrays, tables, turbine definitions and model configurations retain the exact bytes of `APEN_Supplementary_Data_v5.33_20260920.zip` (SHA-256 `0b55a3cf8b8c16084d747fd63828c5010dd4fa59160c4abb3cdc3ecb1157eebc`). The public package adds portable analysis entry points and upstream model/weather code. The original scientific archive is a separate retained artifact.

## Verify without extracting

Use Python 3.11 or later. The complete numerical check additionally requires NumPy; it was tested with Python 3.12.14 and NumPy 2.3.5. Download this ZIP and `verify_apen_archive.py` from the public repository/release. Obtain the expected checksum of this public ZIP from the release's external `SHA256SUMS` or `release_manifest.json`, then run:

```text
python verify_apen_archive.py APEN_Reproducibility_20260927.zip --expected-sha256 PUBLISHED_ZIP_SHA256 --output verification_results.json
```

Replace `PUBLISHED_ZIP_SHA256` with the published 64-character value. The complete ZIP checksum is external to this archive to avoid a self-referential checksum. Passing `--integrity-only` requires only the Python standard library and checks all manifest entries plus screened planning summaries. The verifier's default checksum identifies the original v5.33 scientific ZIP, so the public ZIP requires the explicit argument above.

The complete check reads NPZ files with `allow_pickle=False` and does not execute archived scripts. It verifies every manifest size and SHA-256, recomputes 1,010 site–model responses against archived joint weather probabilities, reselects 171 planning angles using 2014–2019 weather, evaluates 4,650 model–project–year records for 2020–2024, and reproduces all 36 model/screen/region summaries including the common-year bootstrap. It also reselects 171 historical angles from 1981–2010 weather and recomputes 1,203 annual-geometry results from their archived response arrays.

Expected baseline planning results include 155 projects before screening and 139 after excluding 16 flagged projects, including 11 retained coastal projects. The screened total energy gain is 0.4967388663%, and coastal projects account for 27.23111377% of screened net additional energy. Across all 171 baseline sites, 17 are flagged; F155 is the additional flagged site outside the planning sample. Flags identify negative rotor-sampled velocities in the final-geometry response grid and do not imply a separate audit of every historical annual geometry.

## Contents

| Path | Contents |
|---|---|
| `analysis_data/` | Selected angles, planning site/year results, model screening and mechanism statistics |
| `figure_data/` | Numerical tables used by the manuscript figures |
| `server/inputs/` | Turbine coordinates, geometry, weather-grid mapping, regional groups and shared model configuration code |
| `server/outputs/models/` | Six model settings, response arrays, velocity flags and per-site configurations |
| `server/outputs/weather_joint/` | Final-layout weather probabilities for 2014–2024 |
| `server/outputs/turbine_curves/` | NREL 5 MW, IEA 10 MW and IEA 15 MW definitions |
| `server/final_consistency/outputs/` | Historical and standardized-layout power responses, probability arrays and summaries |
| `scripts/upstream/` | Fixed-geometry simulation, weather download/binning and explicit source-input preparation |
| `scripts/calculations/` | Historical/standard-layout simulation and aggregation, planning reselection and mechanism statistics |
| `provenance/previous_selection_metrics.csv` | Earlier-angle audit input; current angles are in `analysis_data/frozen_angles.csv` |
| `manifest.json` | Actual package members, sizes and SHA-256 values; excludes this README and itself |

The repository's separate `apen/plotting/` directory contains the publication plotting workflow, its additional plotting inputs and information about the separate 600 dpi figure archive. Publication document files and correspondence are outside this numerical package.

## Analysis entry points

The scripts locate the package root relative to their own paths. `APEN_DATA_ROOT` can explicitly select another extraction directory. Run calculation scripts on a working copy: they write recalculated products into the documented data/output directories. The direct verifier above reads the ZIP without modifying it.

| Command from the package root | Purpose and inputs |
|---|---|
| `python scripts/calculations/reselect_planning.py` | Rebuild planning angles, site/year tables, screening summaries and bootstrap intervals from bundled probability/response arrays |
| `python scripts/calculations/mechanism_statistics.py` | Rebuild direction-concentration, spacing, country-exclusion and spatial-block statistics from bundled analysis tables and weather summaries |
| `python scripts/calculations/aggregate_consistency.py historical` | Rebuild annual-geometry historical results from bundled power responses and full probabilities |
| `python scripts/calculations/aggregate_consistency.py standard` | Rebuild standard-layout and climate-counterfactual summaries from bundled standard power responses |
| `python scripts/upstream/run_models.py benchmark` | Check the installed FLORIS environment against the archived reference calculations |
| `python scripts/upstream/run_models.py run --scenarios baseline nrel5_fixed_coordinates iea15_fixed_coordinates gauss_ti_minus002 gauss_ti_plus002 jensen --workers 8` | Generate fixed-geometry response arrays; existing complete response files are reused |
| `python scripts/calculations/server_compute.py annual` | Generate missing annual-geometry responses |
| `python scripts/calculations/server_compute.py standard` | Generate missing standardized-layout responses |

The four statistical entry points require NumPy, pandas and SciPy. Their validation environment is listed in `requirements-analysis.txt`. Model regeneration additionally requires FLORIS, pyproj and PyYAML. The original model environment is recorded in `server/outputs/environment.json`; it is not a complete environment lockfile. Model regeneration was not rerun for this public packaging step. The bundled scientific arrays provide the direct numerical verification route.

## Rebuild from external observations

DeepOWT v2.25.1 (DOI `10.5281/zenodo.17121863`) supplies turbine positions and deployment information. ERA5 supplies 100 m wind components. The original raw DeepOWT source and ERA5 time series are not included here. Historical selection uses daily 12:00 UTC samples for 1981–2010; evaluation uses hourly data for 2014–2024. The archived probability arrays preserve the zero-power class, circular direction bins and matched grid cells.

`scripts/upstream/download_arco.py` retrieves the evaluation weather for the archived grid mapping; `scripts/calculations/server_compute.py history` retrieves the historical selection weather. Both require the user's own ECMWF access token in the `ECMWF_ARCO_TOKEN` environment variable. No account credentials are distributed. Download support requires xarray, zarr, fsspec, aiohttp and a suitable NetCDF backend in addition to NumPy/pandas. These retrieval dependencies are not fully pinned by the original environment record.

Evaluation files are written to `server/weather_arco/`. `python scripts/upstream/build_weather.py` reconstructs the wrapped all-hours probabilities from those files. `python scripts/calculations/server_compute.py weather` constructs explicit zero-power probabilities and wind statistics after both periods are present. `APEN_WEATHER_DIR` can point to another evaluation NetCDF directory. The optional `build_weather.py --check-legacy` comparison additionally requires the unbundled `legacy_curves.npz`; the probability calculation itself does not require it.

`prepare_precise_coordinates.py` accepts explicit `--deepowt`, `--coordinates` and `--output` paths. `prepare_inputs.py` accepts `--source-root` and `--output` and preserves the original input-source mapping. Rebuilding that entire earlier source tree additionally requires the original intermediate files named in the mapping. The retained `source_manifest.json` includes three historical inputs that are not distributed: `legacy_curves.npz`, `legacy_weights.npz` and `legacy_weather_code.py`. They are provenance entries, not additional bundled files. The archived derived coordinates, geometry, weather mapping and probabilities support the current analysis without those earlier intermediates.

Natural Earth 50m land and administrative-boundary files are external plotting inputs described with the separate plotting workflow. They do not enter the energy computations.
