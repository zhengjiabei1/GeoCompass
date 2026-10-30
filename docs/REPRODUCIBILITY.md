# Reproducibility

## Requirements

- Python 3.11 or newer.
- No third-party packages. The benchmark uses only the standard library.

## Run

```bash
python -m geoskillcompass experiment --seeds 0,1,2 --output experiments/geoskillcompass
python -m geoskillcompass report --results experiments/geoskillcompass/results.json --output experiments/geoskillcompass
```

The default configuration uses 10 source-training and 6 target-test tasks per family, i.e. 70 and 42 tasks per seed. The run is CPU-only and takes roughly four minutes per seed on a laptop.

## Outputs

```
experiments/geoskillcompass/
  results.json          task-level records and cross-seed aggregates
  REPORT.md             generated tables and figure links
  tables/*.csv          main tables in CSV form
  figures/*.svg         method, reliability, ablation, memory, and transfer figures
```

`results.json` is the source of truth. The report is generated from it and can be rebuilt without rerunning the experiment.

## Determinism

Every task, scene, split, and candidate ordering is derived from the seed. The experiment does not use global random state outside the seeded generators. Running the same command twice should produce identical tables up to floating-point formatting.

## Extending to real data

`geoskillcompass.raster.load_geotiff(path)` is the extension point for real scenes. A real scene must expose the same fields as `Scene`:

- `pre` and `post`: band name to a flat list of reflectance values.
- `classes_pre` and `classes_post`: integer class labels.
- `change_mask`: binary change labels.
- `valid`: valid-pixel mask.
- `truth` and `truth_kind`: evaluation target.

Once a real adapter is in place, the same runner, selectors, memory, and report can be reused.
