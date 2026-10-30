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

The generator is deterministic given the task seed. Real chips enter through `geocompass.io` and `geocompass.realdata`; see `REAL_DATA.md`.

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

Online policies (section 8 of the report) take the same candidate pool but act
prequentially: `static` (no exploration, no update), `linucb_cold`,
`linucb`, `thompson`, and `epsilon_greedy`. They are scored by cumulative regret
rather than by success rate alone, because a policy that always commits can look
good on success while never discovering that a different candidate was better.

## Additions since the first protocol

### Retrieval ablation

Each first stage is evaluated with its own reranker, trained on that stage's candidate pool:

| Stage | Definition |
|---|---|
| `hashed` | cosine similarity between hashed bag-of-words descriptions |
| `bm25` | Okapi BM25 (k1 = 1.2, b = 0.75) over prose plus schema tokens |
| `hybrid` | reciprocal-rank fusion of the two, weight 60 |

Reported per stage: top-1 success, top-1 utility, the fraction of tasks where at
least one candidate beats the baseline (candidate ceiling), and candidates per
task.

### Risk-controlled gate

Calibration uses the target development split. The loss of a task is the
negative-transfer indicator, and abstaining is defined to have zero loss because
the fallback *is* the baseline. Thresholds are drawn from a quantile grid of the
calibration scores; `delta` is split across grid points by a union bound, and
the bound is the exact Clopper-Pearson upper limit on a binomial rate.

| Target risk | Interpretation |
|---|---|
| 0.10 | at most one committed task in ten may be worse than the baseline |
| 0.20 | loose operating point, useful when coverage matters more |
| 0.30 | near the empirical optimum for this benchmark |

An infeasible target abstains on every task rather than returning an
uncertified threshold.

### Online stream

The stream is evaluated prequentially by five policies over the same candidate
pool and the same features (structural set, no memory):

| Policy | Action rule |
|---|---|
| `static` | isotonic score, no exploration, no update |
| `linucb_cold` | LinUCB from an uninformative prior |
| `linucb` | LinUCB warm-started by ridge on the source split |
| `thompson` | one posterior draw per decision, warm-started |
| `epsilon_greedy` | greedy with a 0.1 random-action rate, warm-started |

Reward is `0.5 * (utility + 1)` clipped to `[0, 1]`. Regret is measured against
the best candidate executed on that task, which is available because the runner
evaluates every candidate for the report anyway.

### Candidate-pool sensitivity

The same trained discriminator is evaluated at pool widths 4, 6, 8, and 12. The
pool size is a free parameter of every retrieval-plus-rerank pipeline and is
rarely reported, while the rank feature and the feature normalisation both
depend on it.

### Stream horizon

The default stream is 8 tasks per family (56 steps). `--stream N` changes it;
`experiments/horizon16/COMPARISON.md` reports the 56-versus-112 comparison and
the marginal regret per extra step, which is the quantity that shows whether a
policy is still learning.
