# Reproducibility

## Requirements

- Python 3.11 or newer.
- No third-party packages. The benchmark uses only the standard library.

## Run

```bash
python -m geocompass experiment --seeds 0,1,2 --output experiments/geocompass
python -m geocompass report --results experiments/geocompass/results.json --output experiments/geocompass
```

The default configuration uses 10 source-training and 6 target-test tasks per family, i.e. 70 and 42 tasks per seed. The run is CPU-only and takes roughly four minutes per seed on a laptop; the online-stream block adds one extra pass over the candidate pool and is cached with the rest of the runner.

## Outputs

```
experiments/geocompass/
  results.json          task-level records and cross-seed aggregates
  REPORT.md             generated tables and figure links
  tables/*.csv          main tables in CSV form
  figures/*.svg         method, reliability, ablation, memory, transfer, and regret figures
```

`results.json` is the source of truth. The report is generated from it and can be rebuilt without rerunning the experiment.

## Determinism

Every task, scene, split, and candidate ordering is derived from the seed. The experiment does not use global random state outside the seeded generators. Running the same command twice should produce identical tables up to floating-point formatting.

## Tests

```bash
python -m unittest discover -s tests -v
```

The suite covers the selector, calibration, memory, transfer diagnosis, the
statistics helpers, BM25 and fusion retrieval, the conformal gate, the bandit
policies, the GeoTIFF reader, and the real-data adapter. The GeoTIFF suite writes
its own fixtures and skips the optional real-file check unless
`GEOTIFF_SAMPLE` points at a tile.

## Extending to real data

`geocompass.io.read_geotiff` decodes the file and `geocompass.realdata` maps the
arrays onto `Scene`; `docs/REAL_DATA.md` documents the contract, the supported
encodings, and the public collections that fit the six-band requirement. The
runner, selectors, memory, and report are unchanged once the adapter is in
place.
