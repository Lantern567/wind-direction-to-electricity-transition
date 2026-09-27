# Reproduction materials for the offshore wind array orientation study

**Manuscript:** Prioritizing array orientation assessment in densely packed offshore wind farms

**Scientific scope:** 171 reconstructed offshore wind sites, 1,203 historical-selection site-years, and 155 fixed-2024-layout planning cases. Planning orientations are selected using 2014–2019 winds and evaluated using 2020–2024 winds. The five distinct standard geometries are kept separate from the rotation-equivalent B45 numerical check.

## Download

Download `APEN_Reproducibility_20260927.zip` from [release apen-reproducibility-20260927](https://github.com/Lantern567/wind-direction-to-electricity-transition/releases/tag/apen-reproducibility-20260927). The release includes a SHA-256 checksum file and the current 600 dpi figures. The exact archive identity and membership are recorded in `release_manifest.json`.

The numerical arrays, result tables, and model configurations preserve the scientific data used for the article. The public package supplies the analysis scripts and upstream computational code, with configurable paths and without private execution-account details. Manuscript files and author-contact records are not included in this research-data release.

## Verify the archived calculations

Use Python 3.11 or later and NumPy. Download the numerical ZIP to this directory and run:

```bash
python -m pip install numpy
python verify_apen_archive.py APEN_Reproducibility_20260927.zip --expected-sha256 f30ff4da380f904c195d93ccf4e5be79ecbe1794a1dfd3830e29de959a5e1803 --output verification_results.json
```

The verification command reads the ZIP, checks all 2,809 manifest entries, and recalculates planning and historical energy summaries from archived power responses and meteorological probabilities. It covers 1,010 site–model combinations, 4,650 planning site-years, 1,203 historical site-years, the selected orientations, and 36 model/screen/region summaries. This provides a direct check of the numerical aggregation without rerunning the wake model. The verification report identifies its coverage and numerical tolerances. Detailed upstream commands and environment requirements are in [the numerical archive guide](code/README.md).

## Files

| Path | Content |
|---|---|
| `code/calculations/` | Historical, mechanism, regional, and planning calculations |
| `code/upstream/` | Input preparation, weather construction, and model-response generation |
| `results/analysis_data/` | Site-level, yearly, and scenario-summary results |
| `results/figure_data/` | Numerical tables supporting the figures |
| `plotting/` | Current plotting code and figure reproduction instructions |
| `figures/` | Current vector PDFs and 600 dpi PNG figures |
| `release_manifest.json` | Version and checksum of the downloadable numerical package |

The numerical ZIP additionally contains `server/inputs/`, `server/outputs/`, and `server/final_consistency/`, including archived joint probabilities, annual and standard-layout responses, turbine curves, and per-site model configurations. Its internal manifest lists the exact files and hashes.

## Rebuild from source observations

The archive permits verification of the reported summaries using stored scientific arrays. Rebuilding those arrays starts from the following inputs:

1. [DeepOWT v2.25.1](https://doi.org/10.5281/zenodo.17121863), including turbine locations and deployment-state fields, using the source filtering and clustering procedure documented in SI S1.1.
2. [ERA5 reanalysis](https://doi.org/10.1002/qj.3803), using 100 m wind components for the archived site-to-grid mapping. Historical selection uses daily 12:00 UTC values for 1981–2010; evaluation uses hourly data for 2014–2024. Raw meteorological time series must be retrieved from the provider.
3. The archived reference-turbine curves and per-site model configurations. The scientific model environment is recorded in `server/outputs/environment.json` and uses FLORIS 4.6.6.
4. Natural Earth 50 m land and national-boundary layers for map plotting. These cartographic layers do not enter the energy calculations.

Weather retrieval may require provider access credentials, which are not distributed. Third-party datasets and turbine definitions remain subject to their original terms and attribution requirements. The archived software version record covers the numerical model and statistics; plotting and data-download dependencies are listed with the relevant scripts.

## Interpretation

The calculations use reference turbines at reconstructed coordinates. They do not constitute measured production or engineering feasibility validation. Model-output screening, turbine/model sensitivity, the distinction between orientation sensitivity and gain relative to the original orientation, and the fixed selection/evaluation periods must be retained when using these results.
