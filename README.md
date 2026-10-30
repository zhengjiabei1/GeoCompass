# GeoCompass

**Calibrated spatial-skill reuse and tiered experience memory for geospatial agents.**

Large language models are increasingly used as geospatial agents that plan and call GIS and remote-sensing tools. A recurring failure is **negative transfer**: a spatial-analysis skill that worked on one task family is reused on an incompatible one, and the agent fails silently. GeoCompass asks a simple question:

> When should a geospatial agent reuse an existing spatial-analysis skill, and when should it explore a new one?

The repository contains a deterministic, CPU-only benchmark, a calibrated reuse
discriminator, a risk-controlled gate, an online selection study, a tiered
experience memory, and a reader for real GeoTIFF chips. Every number in the
generated report comes from the included runner.

## Method

| Component | What it does |
|---|---|
| **Spatial skill library** | Expert pipelines for land cover, crop, water, flood, change, urban, and burn mapping. Each skill declares its target, input kind, spectral mechanism, and cost. |
| **Candidate retrieval** | Three first stages over the same library: a hashed description embedding, BM25 over prose plus schema tokens, and reciprocal-rank fusion of the two. Retrieval quality is reported separately so that a reranking gain is not confused with a retrieval gain. |
| **Calibrated reuse discriminator** | Logistic regression over task-skill structural features, calibrated with temperature scaling or isotonic regression. It estimates the probability that a skill will succeed **before** the agent executes it. |
| **Risk-controlled gate** | Split conformal risk control on the commit/abstain threshold, with an exact binomial upper bound, so that the accepted negative-transfer rate comes with a finite-sample guarantee. The uncorrected empirical threshold is reported beside it. |
| **Online selection** | The same decision as a contextual bandit: LinUCB, Thompson sampling, and epsilon-greedy over a shared linear reward model, warm-started from the source split and scored by cumulative regret against the best candidate in hindsight. |
| **Transfer gate** | A Beta-binomial diagnosis of source-to-target transferability. Skills with low expected utility on the target family are blocked, which reduces negative transfer. |
| **Tiered experience memory** | Short-term and long-term experience stores with deduplication. The tiered store keeps the useful outcomes while using roughly one sixth of the storage of a flat replay buffer. |

The key design choice is to separate two questions that similarity-based retrievers conflate:

1. **Is this skill relevant to the task?** (semantic and structural match)
2. **Am I confident enough that it will succeed here?** (calibrated outcome probability and transfer evidence)

## Benchmark

The benchmark generates synthetic multi-band scenes with six spectral bands (`blue`, `green`, `red`, `nir`, `swir1`, `swir2`) and seven task families:

- `landcover` — six-class land-cover mapping
- `crop` — crop / non-crop mapping
- `water` — water extraction
- `flood` — flood mapping
- `change` — bi-temporal change detection
- `urban` — urban-expansion detection
- `burn` — burn-scar detection

Scenes vary by region, season, cloud cover, and resolution. The generator is
deterministic and runs on CPU. Real chips enter through `geocompass.io` and
`geocompass.realdata`; see [`docs/REAL_DATA.md`](docs/REAL_DATA.md).

## Results

Run: 3 seeds (0, 1, 2), 70 source-training and 42 target-test tasks per seed, 20-skill library, CPU-only, 680 s.

| Method | Success | AUROC | ECE | Negative transfer |
|---|---:|---:|---:|---:|
| Semantic retrieval | 0.500 | 0.666 | 0.152 | 0.183 |
| Logistic discriminator | 0.683 | 0.842 | 0.121 | 0.056 |
| + isotonic calibration | 0.675 | 0.842 | 0.063 | 0.056 |
| + transfer gate | **0.698** | **0.842** | **0.063** | **0.048** |

Tiered memory, prequential stream:

| Memory | Success | ECE | Records | Characters |
|---|---:|---:|---:|---:|
| None | 0.774 | 0.232 | 0 | 0 |
| Flat | 0.780 | 0.113 | 616 | 57,315 |
| Tiered | 0.780 | 0.131 | 106 | 9,535 |

Candidate retrieval, each stage with its own reranker:

| First stage | Top-1 success | Top-1 utility | Candidate ceiling |
|---|---:|---:|---:|
| hashed embedding | 0.675 | 0.481 | 1.000 |
| BM25 only | 0.619 | 0.461 | 0.992 |
| reciprocal-rank fusion | **0.698** | **0.516** | 1.000 |

Risk-controlled gate, target negative-transfer rate 10%:

| Gate | Commit threshold | Coverage | Test risk | 90% upper bound |
|---|---:|---:|---:|---:|
| conformal | 0.854 | 0.286 | 0.028 | 0.104 |
| empirical only | 0.164 | 1.000 | 0.056 | 0.092 |

Online selection, 56-step stream, cumulative regret:

| Policy | Regret | Success | Negative transfer |
|---|---:|---:|---:|
| static, calibrated | **2.81** | **0.679** | 0.071 |
| LinUCB, warm | 5.15 | 0.554 | 0.054 |
| Thompson, warm | 4.89 | 0.577 | **0.042** |
| LinUCB, cold | 4.78 | 0.583 | 0.113 |
| epsilon-greedy | 5.99 | 0.500 | 0.083 |

**Key findings**

- Calibration lifts AUROC from 0.666 to 0.842 and cuts ECE from 0.152 to 0.063.
- The transfer gate raises success from 0.675 to 0.698 and cuts negative transfer from 0.056 to 0.048.
- Tiered memory keeps success at 0.780 while using about one sixth of the flat store's characters (9,535 vs. 57,315).
- Fusing BM25 with the description embedding raises top-1 success from 0.675 to 0.698, so part of what earlier looked like a reranking gain was a retrieval gap.
- A certified 10% negative-transfer gate costs 71% of coverage; at a 20% target it commits to every task while the bound still holds. The empirical-only threshold happens to satisfy the bound on this test split, which is exactly why it cannot be trusted to.
- Every online policy pays more regret than the static ranker on a 56-step stream. Warm-started Thompson sampling halves negative transfer but loses 10 points of success; exploration is not free at this horizon.
- Budgeted retry reaches 0.746 success but raises negative transfer to 0.286; the gate is the safer operating point.

Full tables and figures: [`experiments/geocompass/REPORT.md`](experiments/geocompass/REPORT.md).

The experiment reports:

- **Success** — task completion rate on held-out target tasks.
- **Negative transfer** — reuse attempts that reduce utility relative to the task baseline.
- **AUROC / AUPRC** — ranking quality of the reuse discriminator.
- **ECE / Brier / NLL** — calibration quality.
- **Retained records / characters** — memory footprint of the tiered store versus a flat store.
- **Coverage / certified risk** — how often the gate commits, and the upper bound on its negative-transfer rate.
- **Cumulative regret** — how much utility an online policy gives up against the best candidate in hindsight.

## Reproduction

```bash
python -m geocompass experiment --seeds 0,1,2 --output experiments/geocompass
python -m geocompass report
```

The runner writes `results.json`, CSV tables, SVG figures, and a generated `REPORT.md` under `experiments/geocompass/`.

## Repository layout

```
geocompass/
  types.py        shared task, scene, and skill dataclasses
  raster.py       synthetic multi-band scene generator
  io.py           dependency-free GeoTIFF/BigTIFF reader and test writer
  realdata.py     GeoTIFF chip to Scene adapter
  tasks.py        task families and expert skill library
  features.py     structural and semantic features
  retrieval.py    BM25 and reciprocal-rank fusion first stages
  selector.py     logistic, temperature, and isotonic selectors
  conformal.py    split conformal risk control for the commit threshold
  bandit.py       LinUCB, Thompson sampling, epsilon-greedy policies
  memory.py       flat and tiered experience memory
  transfer.py     transfer diagnosis and gating
  experiment.py   end-to-end runner
  report.py       tables, figures, and REPORT.md
docs/
  RELATED_WORK.md where each component comes from
  REAL_DATA.md    running on real imagery
experiments/
  geocompass/     results, tables, figures, report
```

## Documentation

- [`docs/RELATED_WORK.md`](docs/RELATED_WORK.md) — adopted and deliberately
  omitted mechanisms, with references.
- [`docs/REAL_DATA.md`](docs/REAL_DATA.md) — the scene contract, supported file
  formats, and how to build chips from public collections.
- [`docs/BENCHMARK.md`](docs/BENCHMARK.md) — task families, splits, protocol.
- [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) — exact commands and
  runtime expectations.

## Design notes

- **No third-party dependencies.** The runner uses only the Python standard library, so the benchmark is easy to reproduce on a laptop.
- **Deterministic.** Every split is derived from a seed; there is no hidden randomness outside the seeded generators.
- **Synthetic by design.** The benchmark isolates the skill-reuse decision from data-download and preprocessing noise. The same interface accepts real GeoTIFF chips through `geocompass.realdata`.
- **Negative transfer is a first-class metric.** A method that always reuses the most similar skill can look good on average while failing badly on incompatible families.
- **Retrieval is measured, not assumed.** All three first stages feed the same reranker, so the report separates retrieval quality from reranking quality.
- **Guarantees cost coverage.** The conformal gate can only promise what the calibration set can certify; when the sample is too small it abstains instead of returning an optimistic threshold, and `conformal.required_calibration_size` states the sample size a given risk target needs.
- **Exploration is not free.** Warm-starting the bandit policies from the source split is what makes their regret comparable to a static policy; the cold-start variant shows the difference.

## Citation

If you use this code or benchmark, please cite:

```bibtex
@misc{zheng2026geocompass,
  title  = {GeoCompass: Calibrated Spatial-Skill Reuse for Geospatial Agents},
  author = {Zheng, Jiabei},
  year   = {2026},
  url    = {https://github.com/zhengjiabei1/GeoCompass}
}
```

## License

MIT. See [LICENSE](LICENSE).
