# GeoSkillEvo experiment report

Seeds: 0, 1, 2. Tasks per seed: {'source_train': 70, 'source_val': 28, 'target_dev': 56, 'target_test': 42, 'stream': 56}. Mean skill-library size: 20.0. Total runtime: 664 s.

## 1. Skill selection quality

| Method | Success | 95% CI | Score | Negative transfer | AUROC | AUPRC | Brier | NLL | ECE | P@1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| random | 0.349 | [0.270, 0.437] | 0.389 | 0.294 | 0.495 | 0.369 | 0.334 | 1.007 | 0.284 | 0.333 |
| semantic | 0.500 | [0.413, 0.587] | 0.505 | 0.183 | 0.666 | 0.547 | 0.233 | 1.127 | 0.152 | 0.500 |
| frequency | 0.540 | [0.452, 0.619] | 0.524 | 0.198 | 0.509 | 0.362 | 0.246 | 0.686 | 0.138 | 0.540 |
| logistic | 0.683 | [0.595, 0.762] | 0.632 | 0.056 | 0.842 | 0.655 | 0.164 | 0.491 | 0.120 | 0.683 |
| temperature | 0.683 | [0.595, 0.762] | 0.632 | 0.056 | 0.842 | 0.655 | 0.165 | 0.494 | 0.105 | 0.683 |
| isotonic | 0.675 | [0.587, 0.754] | 0.629 | 0.056 | 0.842 | 0.661 | 0.156 | 0.629 | 0.063 | 0.675 |
| fewshot | 0.683 | [0.595, 0.762] | 0.632 | 0.056 | 0.841 | 0.654 | 0.157 | 0.612 | 0.087 | 0.683 |
| isotonic_budget | 0.746 | [0.667, 0.817] | 0.701 | 0.286 | 0.842 | 0.661 | 0.156 | 0.629 | 0.063 | 0.675 |
| oracle (candidate upper bound) | 0.802 | [0.730, 0.865] | 0.753 | 0.000 | - | - | - | - | - | - |

![Method comparison](figures/methods.svg)

## 2. Calibration

Reliability diagram over all target-task candidate skills, averaged across seeds. The diagonal is perfect calibration.

![Reliability](figures/reliability.svg)

## 3. Feature ablation

All variants use isotonic calibration and the same candidate pool; only the discriminator features change.

| Feature set | Success | Score | AUROC | ECE |
|---|---:|---:|---:|---:|

![Ablation](figures/ablation.svg)

## 4. Cross-task transfer diagnosis

| Gate strategy | Success | Score | Negative transfer |
|---|---:|---:|---:|
| none | 0.675 | 0.629 | 0.056 |
| semantic | 0.579 | 0.559 | 0.103 |
| outcome | 0.675 | 0.629 | 0.056 |
| proposed | 0.698 | 0.647 | 0.048 |

### Source-to-target transferability

Each cell reports the mean probe utility and structural compatibility estimated on the target development split.

![Transfer heatmap](figures/transfer_heatmap.svg)

### Representative source-to-target pairs

| Pair | Strategy | Success | Negative transfer | Blocked skills | Library |
|---|---|---:|---:|---:|---:|
| landcover->change | none | 0.056 | 0.222 | 0.0 | 2.0 |
| landcover->change | semantic | 0.056 | 0.222 | 0.0 | 2.0 |
| landcover->change | proposed | 0.000 | 0.000 | 2.0 | 2.0 |
| landcover->change | oracle | 0.056 | 0.000 | 0.0 | 2.0 |
| change->landcover | none | 0.000 | 1.000 | 0.0 | 3.0 |
| change->landcover | semantic | 0.000 | 1.000 | 0.0 | 3.0 |
| change->landcover | proposed | 0.000 | 0.000 | 3.0 | 3.0 |
| change->landcover | oracle | 0.000 | 0.000 | 0.0 | 3.0 |
| water->flood | none | 1.000 | 0.000 | 0.0 | 4.7 |
| water->flood | semantic | 1.000 | 0.000 | 0.0 | 4.7 |
| water->flood | proposed | 1.000 | 0.000 | 0.0 | 4.7 |
| water->flood | oracle | 1.000 | 0.000 | 0.0 | 4.7 |
| flood->water | none | 0.000 | 0.000 | 0.0 | 1.0 |
| flood->water | semantic | 0.000 | 0.000 | 0.0 | 1.0 |
| flood->water | proposed | 0.000 | 0.000 | 0.0 | 1.0 |
| flood->water | oracle | 0.000 | 0.000 | 0.0 | 1.0 |
| crop->landcover | none | 0.944 | 0.000 | 0.0 | 5.7 |
| crop->landcover | semantic | 0.944 | 0.000 | 0.0 | 5.7 |
| crop->landcover | proposed | 0.944 | 0.000 | 0.0 | 5.7 |
| crop->landcover | oracle | 0.944 | 0.000 | 0.0 | 5.7 |
| landcover->crop | none | 0.611 | 0.056 | 0.0 | 2.0 |
| landcover->crop | semantic | 0.611 | 0.056 | 0.0 | 2.0 |
| landcover->crop | proposed | 0.611 | 0.056 | 0.0 | 2.0 |
| landcover->crop | oracle | 0.611 | 0.000 | 0.0 | 2.0 |
| change->urban | none | 0.667 | 0.444 | 0.0 | 3.0 |
| change->urban | semantic | 0.667 | 0.444 | 0.0 | 3.0 |
| change->urban | proposed | 0.667 | 0.444 | 1.0 | 3.0 |
| change->urban | oracle | 0.667 | 0.000 | 0.0 | 3.0 |
| urban->change | none | 0.667 | 0.278 | 0.0 | 1.7 |
| urban->change | semantic | 0.667 | 0.278 | 0.0 | 1.7 |
| urban->change | proposed | 0.611 | 0.167 | 0.7 | 1.7 |
| urban->change | oracle | 0.667 | 0.000 | 0.0 | 1.7 |
| urban->burn | none | 0.333 | 0.222 | 0.0 | 1.7 |
| urban->burn | semantic | 0.333 | 0.222 | 0.0 | 1.7 |
| urban->burn | proposed | 0.333 | 0.056 | 0.7 | 1.7 |
| urban->burn | oracle | 0.333 | 0.000 | 0.0 | 1.7 |
| burn->urban | none | 0.056 | 0.944 | 0.0 | 2.0 |
| burn->urban | semantic | 0.056 | 0.944 | 0.0 | 2.0 |
| burn->urban | proposed | 0.056 | 0.444 | 1.0 | 2.0 |
| burn->urban | oracle | 0.056 | 0.000 | 0.0 | 2.0 |

## 5. Tiered experience memory

Prequential evaluation on a held-out stream. Each method selects a skill before observing the outcome and then ingests the outcome.

| Memory mode | Success | Score | ECE | Retained records | Characters | Duplicate ratio |
|---|---:|---:|---:|---:|---:|---:|
| none | 0.774 | 0.709 | 0.232 | 0 | 0 | 0.000 |
| flat | 0.780 | 0.699 | 0.113 | 616 | 57315 | 0.000 |
| tiered | 0.780 | 0.702 | 0.131 | 106 | 9535 | 0.000 |
| tiered_dedup | 0.786 | 0.707 | 0.141 | 104 | 9428 | 0.500 |
| full | 0.774 | 0.705 | 0.120 | 109 | 9987 | 0.492 |

![Memory](figures/memory.svg)

## 6. Paired tests

| Comparison | Success delta | McNemar p | Score delta | Permutation p |
|---|---:|---:|---:|---:|
| isotonic_vs_semantic | 0.175 | 0.0000 | 0.124 | 0.0002 |
| isotonic_vs_random | 0.325 | 0.0000 | 0.240 | 0.0002 |
| temperature_vs_semantic | 0.183 | 0.0000 | 0.127 | 0.0002 |
| temperature_vs_random | 0.333 | 0.0000 | 0.243 | 0.0002 |

## Reproduction

```bash
python -m geoskillevo.experiment --seeds 0,1,2 --output experiments/geoskillevo
python -m geoskillevo.report
```
