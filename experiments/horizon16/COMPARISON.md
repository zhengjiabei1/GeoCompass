# Stream horizon: 56 versus 112 steps

The default protocol gives each policy 56 prequential steps. The earlier run
concluded that every online policy paid more regret than a static calibrated
ranker, which raises an obvious objection: 56 steps may be too short for
exploration to pay back. This note repeats the stream with 16 tasks per family
(112 steps) and compares the two.

```bash
python -m geocompass experiment --seeds 0,1,2 --stream 16 --no-ablations \
    --output experiments/horizon16
python scripts/compare_horizons.py
```

## Result

| Policy | Regret at 56 | Regret at 112 | Marginal regret per extra step | Success at 112 | Negative transfer at 112 |
|---|---:|---:|---:|---:|---:|
| static, calibrated | 3.07 | 5.58 | 0.0448 | 0.643 | 0.068 |
| LinUCB, warm | 7.12 | 10.17 | 0.0546 | 0.530 | 0.071 |
| Thompson, warm | 7.05 | 10.07 | 0.0540 | 0.539 | 0.062 |
| LinUCB, cold | 4.54 | 9.92 | 0.0961 | 0.515 | 0.068 |
| epsilon-greedy | 6.81 | 10.24 | 0.0612 | 0.530 | 0.083 |

## Interpretation

- The static policy costs about 0.045 units of reward per step and that rate is
  flat: it does not learn, and it does not need to.
- The warm-started policies improve. Their marginal regret in the second half
  (0.054) is well below their average over the whole run (0.091), so the
  posterior is doing what it should.
- Improvement is not enough. Their marginal rate is still above the static
  policy's 0.045, so the cumulative gap keeps widening, more slowly but in the
  same direction.
- The cold-start policy gets *worse* with length (0.096 marginal against 0.081
  average): with one shared model and 20 arms, relearning from an uninformative
  prior does not converge within a hundred steps.

The honest conclusion is that this benchmark does not contain enough
task-level diversity for exploration to beat a well-calibrated offline ranker.
The interesting question is not "does Thompson sampling win" but "what is the
prior worth", and the answer here is: a lot. Warming the posterior from the
source split is what keeps the bandits within a factor of two of a policy that
never explores.

A longer stream alone will not change the ordering. What would change it is
non-stationarity: a task distribution that drifts, an environment whose skill
set changes, or a candidate pool that grows. Those are the regimes where paying
for information is rational, and they are not what this benchmark tests.
