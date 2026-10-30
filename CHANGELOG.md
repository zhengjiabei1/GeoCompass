# Changelog

## Unreleased

### Added

- `geocompass/retrieval.py` — BM25 over prose plus schema tokens, and
  reciprocal-rank fusion with the description embedding. Three first stages now
  feed the same reranker, and the report separates retrieval quality from
  reranking quality.
- `geocompass/conformal.py` — split conformal risk control for the commit
  threshold, with an exact Clopper-Pearson upper bound, a union-bound correction
  over the threshold grid, and `required_calibration_size` for stating how much
  calibration data a risk target needs. An infeasible target abstains instead of
  returning an optimistic threshold.
- `geocompass/bandit.py` — the stream decision as a contextual bandit: LinUCB,
  Thompson sampling, and epsilon-greedy over a shared linear reward model, with
  closed-form ridge warm starts and cumulative regret against the best candidate
  in hindsight.
- `geocompass/io.py` — dependency-free GeoTIFF and BigTIFF reader (strips,
  tiles, Deflate, horizontal predictor, integer and float samples, chunky and
  planar storage) plus a minimal writer used by the tests.
- `geocompass/realdata.py` — adapter from decoded rasters to `Scene`, with an
  explicit band contract and a nodata mask.
- `geocompass/dataset.py` — manifest loader that turns a directory of chips into
  the same splits, scenes, and skill library the synthetic generator produces,
  so `--dataset DIR` runs the whole pipeline on real imagery.
- Holm-Bonferroni adjustment over the paired-test family, reported next to the
  raw p-values.
- `docs/RELATED_WORK.md` — what each component was borrowed from, what was
  adapted, and what was deliberately left out.
- `docs/REAL_DATA.md` — the scene contract, supported encodings, and the public
  collections that fit the six-band requirement.
- New experiment blocks: retrieval ablation, conformal gate at three risk
  targets, and five online policies, with matching tables, CSVs, and a regret
  figure.

### Changed

- `experiment.py` accepts `--retriever hashed|bm25|hybrid`.
- The gate thresholds are reported with the confidence correction that produced
  them, and the uncorrected empirical threshold is reported next to them.
- `report.py` gained three sections and a cumulative-regret chart.
- The runner docstring no longer claims that every number comes from a
  single source document; the generated report is the artifact.

### Unchanged

- The task families, scene generator, skill library, splits, selector, memory,
  and transfer diagnosis are untouched, so results from earlier runs remain
  comparable on the shared tables.
