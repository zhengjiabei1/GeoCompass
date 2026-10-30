# Benchmark details

## Task families

| Family | Target | Input | Spectral mechanism | Success threshold |
|---|---|---|---|---|
| `landcover` | six-class land cover | single date | multispectral | 0.78 accuracy |
| `crop` | crop / non-crop | single date | vegetation index | 0.55 F1 |
| `water` | water | single date | NDWI | 0.65 F1 |
| `flood` | flood | single date | NDWI | 0.60 F1 |
| `change` | change | bi-temporal | NDVI change | 0.40 F1 |
| `urban` | urban expansion | bi-temporal | NDBI change | 0.45 F1 |
| `burn` | burn scar | bi-temporal | NBR change | 0.50 F1 |

The thresholds were fixed before the final test runs. They reflect the difficulty of each target rather than a single global threshold.

## Scene generator

Each scene is a 32 x 32 grid with six surface-reflectance bands (`blue`, `green`, `red`, `nir`, `swir1`, `swir2`). The generator varies:

- **Region** — central, north, south, coast. Region changes class priors.
- **Season** — spring, summer, fall. Season changes spectral gain.
- **Cloud** — 0.0, 0.08, 0.16. Cloud adds missing observations.
- **Resolution** — 10 m or 20 m. Resolution changes the effective spatial support.

The generator is deterministic given the task seed. `geoskillevo.raster.load_geotiff` is the extension point for real scenes.

## Splits

For each seed:

- `source_train` — historical tasks used to build the skill library and fit the discriminator.
- `source_val` — held-out source tasks used for calibration.
- `target_dev` — target tasks used for transfer diagnosis and gating.
- `target_test` — held-out target tasks used for the main results.
- `stream` — a prequential stream for memory evaluation.

The default configuration uses 10 source training tasks and 6 target test tasks **per family**, i.e. 70 and 42 total tasks respectively.

## Metrics

| Metric | Definition |
|---|---|
| **Success** | Fraction of target tasks whose selected skill passes the family threshold. |
| **Score** | Continuous utility of the selected skill, normalized per family. |
| **Negative transfer** | Fraction of reuse attempts whose utility is below the task baseline. |
| **AUROC / AUPRC** | Ranking quality of the reuse discriminator over candidate skills. |
| **ECE** | Expected calibration error of the predicted success probability. |
| **Brier / NLL** | Proper scoring rules for probabilistic predictions. |
| **Retained records / characters** | Memory footprint of the experience store. |

## Baselines

- **random** — sample a candidate uniformly.
- **semantic** — rank by description embedding similarity.
- **frequency** — rank by historical success frequency.
- **logistic** — uncalibrated logistic discriminator.
- **temperature / isotonic** — calibrated variants.
- **fewshot** — isotonic calibration fitted on target development data.
- **oracle** — candidate upper bound, not deployable.
