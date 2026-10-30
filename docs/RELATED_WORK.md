# Related work and design rationale

This note records where the components of GeoCompass come from, which parts of
the literature were adopted as-is, which were adapted, and which were
deliberately left out.  It exists so that a reader can tell an inherited
mechanism from a new one, and so that the next change has a place to start.

## 1. What the project actually claims

GeoCompass studies one decision: **given a library of analysis procedures and a
new task, should the agent reuse a procedure or fall back to the generic
pipeline?**  Everything else in the repository supports that question.

The decision is framed twice, because it has two faces:

| View | Question | Machinery |
|---|---|---|
| Offline | How well can a scorer rank candidates by probability of success? | logistic discriminator, calibration, AUROC/Brier/ECE |
| Online | Should the agent commit now or spend an execution learning? | contextual bandit, cumulative regret |
| Safety | What is the worst acceptable rate of negative transfer? | split conformal risk control |

The transfer diagnosis (Beta-binomial probe) sits between the offline and
online views: it answers "would this skill transfer at all?" with a small amount
of target-side evidence.

## 2. Skill libraries and experience memory

The idea that an agent should accumulate executable procedures rather than
re-plan from scratch is established:

- **Voyager** (Wang et al., 2023) stores executable code skills in a growing
  library, retrieves them by description, and adds new ones when a task fails.
  GeoCompass borrows the library abstraction and the *deduplicate-by-signature*
  discipline; it does not borrow the embodied environment or the LLM planner.
- **ExpeL** (Zhao et al., 2024) extracts natural-language insights from
  success/failure trajectories and reuses them at inference time.  This is the
  conceptual ancestor of the tiered memory: recent episodes stay verbatim,
  older ones are consolidated.
- **Agent Workflow Memory** (Wang et al., 2024) induces reusable workflows from
  past trajectories.  Its evaluation shows that reuse helps most when the
  workflow is *structured*, which is exactly why GeoCompass stores typed
  procedure macros with declared targets, input kinds, and bands instead of
  free text.
- **Reflexion**, **MemGPT**, **MemoryBank**, and **Generative Agents** supply the
  memory-management vocabulary: recency, importance, consolidation, paging.  The
  two-tier store here is a deliberately small instance of the same idea, sized
  so that the storage effect can be measured in characters.

What is *not* borrowed: none of these systems calibrate the reuse probability,
and none measure negative transfer across task families.  That gap is the
project's entry point.

## 3. Candidate retrieval

Reuse starts with retrieval, and retrieval quality bounds everything downstream.
Three standard ingredients are used:

- **BM25** (Robertson & Zaragoza, 2009) as the lexical ranker.  The skill
  document mixes prose with schema tokens (`targets`, `input_kind`,
  `required_bands`, primitive names), because two recipes can read differently
  yet declare the same mechanism.
- **Dense/description retrieval** (Karpukhin et al., 2020 for the passage
  analogue) as the original first stage.  Here it is a hashed bag-of-words
  embedding rather than a trained dual encoder; that keeps the package
  dependency-free and makes the baseline reproducible on a laptop.
- **Reciprocal rank fusion** (Cormack et al., 2009) to combine the two without
  score normalisation, since the scorers live on different scales.

**Re-Invoke** (Chen et al., 2024) is the closest published work on tool
retrieval: it rewrites the user request into plausible tool invocations and uses
those rewrites as queries, avoiding any training.  The rewrite step is a natural
extension here and is listed as an open item rather than implemented, because it
requires an LLM in the loop and the current benchmark is deliberately LLM-free.

Reporting all three first stages matters for honesty: if a reranker is compared
only against a weak embedding baseline, part of the reported gain is really a
retrieval gain.  `experiments/REPORT.md` therefore reports the same downstream
reranker behind each first stage.

## 4. Calibration and risk control

Two literatures meet at the commit/abstain threshold.

**Post-hoc calibration.**  Temperature scaling and isotonic regression follow
Guo et al. (2017) and Zadrozny & Elkan (2002); the reliability diagram and ECE
follow the same convention.  Calibrated Kuleshov et al. (2018) motivates
evaluating a *probability* rather than an accuracy, which is what a gate needs.

**Selective prediction and risk control.**  A gate that abstains is a selective
classifier (Geifman & El-Yaniv, 2017), and the natural evaluation is a
risk-coverage curve rather than a single operating point.  The threshold itself
is chosen with split conformal risk control (Angelopoulos et al., 2024; Bates et
al., 2021), using the monotone structure of the nested threshold family.

Two implementation choices are worth stating because they change the numbers:

1. **Exact binomial bound instead of Hoeffding.**  The loss is Bernoulli, so the
   Clopper-Pearson upper limit is admissible and much tighter at small samples.
   With `n` calibration tasks, zero observed failures, a `delta` budget spread
   over a `|G|`-point threshold grid, a Hoeffding bound certifies nothing below
   `sqrt(log(|G|/delta) / 2n)`, which is still about 0.18 at `n = 56`.  The
   binomial bound certifies roughly `1 - (delta/|G|)^(1/n)`, about 0.10 at the
   same size.  `conformal.required_calibration_size` makes the requirement
   explicit instead of returning an unusable gate.
2. **Union bound over the grid.**  The threshold is selected on the same data
   used for the bound, so `delta` is split across grid points.  This is
   conservative but valid; the empirical-only gate is reported next to it so the
   cost of the correction is visible.

A conformal gate is not a better predictor.  It is a predictor plus a promise
about the rate at which the promise can be broken, and the promise costs
coverage.  The report shows both.

## 5. Explore versus commit

Reuse is not only prediction: executing a skill produces information.  Treating
the stream as a contextual bandit makes that explicit.

- **LinUCB** (Li et al., 2010) gives an optimism-under-uncertainty score
  `theta^T x + alpha * sqrt(x^T A^{-1} x)`; the quadratic form is the posterior
  variance of the linear reward model.
- **Thompson sampling** (Chapelle & Li, 2011) draws a coefficient vector per
  decision and acts greedily on the draw.
- **Epsilon-greedy** is kept as the cheapest possible baseline for the value of
  exploration.

All three share one reward model over the structural features rather than
keeping a parameter block per arm.  The reason is sample size: with about 20
skills and 18 features, per-arm models would need more stream data than the
benchmark provides, and the features already encode arm identity
(`mechanism_match`, `input_match`, `bitemporal_skill`).

The policies are **warm-started** with a closed-form ridge fit on the source
split, and a cold-start LinUCB is reported alongside.  This isolates the part of
the regret that comes from exploration from the part that comes from relearning
what the source split already taught.  Lattimore & Szepesvári (2020) is the
reference for the regret framing.

## 6. Transfer and negative transfer

"Reuse can hurt" is old knowledge under a different name: negative transfer in
transfer learning (Taylor & Stone, 2009; Zhuang et al., 2021).  The remote
sensing analogue is a procedure whose spectral mechanism does not match the new
target: NDWI works for water and floods but not for burn scars, and a
single-date classifier cannot use bi-temporal evidence at all.

The diagnosis uses a Beta-binomial posterior over probe utility, with the
structural compatibility score as a prior.  The prior weights are hand-set and
documented as such; the ablation shows that structural features carry most of
the static selection signal, which is an uncomfortable but reportable result.

## 7. GeoAI context and benchmarks

The setting is not hypothetical.  Recent benchmarks show that tool-using
geospatial agents remain far from solved: GISAgentBench reports 32.7% strict
completion for the best of six models, and the GeoNatureAgent benchmark reports
60.8% for the strongest frontier model against a production-style API, with mean
accuracies 25-35 points below general GIS benchmarks.  GeoBenchX (Krechetova &
Kochedykov, 2025) evaluates multi-step tool calling for commercial GIS practice.

Those benchmarks measure *end-to-end task completion*.  GeoCompass measures one
sub-decision inside such a pipeline, on a controlled generator where the ground
truth, the change type, and the spectral mechanism are known.  The two are
complementary, and the limitation is stated in the README: a synthetic
diagnostic cannot establish operational mapping accuracy.  `docs/REAL_DATA.md`
describes the adapter that carries the same interfaces onto real chips.

## 8. Deliberate omissions

- **No LLM planner.**  Adding one would confound selection quality with prompt
  quality, and would make the benchmark non-deterministic.  The cost is that
  "agent" here means "procedure-selecting controller", not "conversational
  agent"; the README says so.
- **No foundation-model embeddings.**  RemoteCLIP (Liu et al., 2024), Prithvi
  (Jakubik et al., 2023), and SpectralGPT (Hong et al., 2024) would very likely
  improve the semantic feature, and are the obvious next baseline.  They are
  excluded because the package is standard-library only and CPU-only by design;
  a comparison against them belongs in a separate, GPU-enabled evaluation.
- **No deep segmentation backbone.**  Skills are deterministic programs on
  purpose: the question is which program to reuse, not how to train a better
  one.

## 9. References

1. G. Wang et al., "Voyager: An Open-Ended Embodied Agent with Large Language Models," arXiv:2305.16291, 2023.
2. A. Zhao et al., "ExpeL: LLM Agents Are Experiential Learners," AAAI, 2024.
3. Z. Z. Wang et al., "Agent Workflow Memory," arXiv:2409.07429, 2024.
4. N. Shinn et al., "Reflexion: Language Agents with Verbal Reinforcement Learning," NeurIPS, 2023.
5. C. Packer et al., "MemGPT: Towards LLMs as Operating Systems," arXiv:2310.08560, 2023.
6. W. Zhong et al., "MemoryBank: Enhancing Large Language Models with Long-Term Memory," AAAI, 2024.
7. J. S. Park et al., "Generative Agents: Interactive Simulacra of Human Behavior," UIST, 2023.
8. S. Robertson and H. Zaragoza, "The Probabilistic Relevance Framework: BM25 and Beyond," Foundations and Trends in Information Retrieval, 2009.
9. V. Karpukhin et al., "Dense Passage Retrieval for Open-Domain Question Answering," EMNLP, 2020.
10. G. V. Cormack, C. L. A. Clarke, and S. Buettcher, "Reciprocal Rank Fusion Outperforms Condorcet and Individual Rank Learning Methods," SIGIR, 2009.
11. Y. Chen et al., "Re-Invoke: Tool Invocation Rewriting for Zero-Shot Tool Retrieval," Findings of EMNLP, 2024.
12. Y. Qin et al., "ToolLLM: Facilitating Large Language Models to Master 16000+ Real-world APIs," ICLR, 2024.
13. C. Guo et al., "On Calibration of Modern Neural Networks," ICML, 2017.
14. B. Zadrozny and C. Elkan, "Transforming Classifier Scores into Accurate Multiclass Probability Estimates," KDD, 2002.
15. V. Kuleshov, N. Fenner, and S. Ermon, "Accurate Uncertainties for Deep Learning Using Calibrated Regression," ICML, 2018.
16. A. N. Angelopoulos, S. Bates, A. Fisch, L. Lei, and T. Schuster, "Conformal Risk Control," ICLR, 2024.
17. A. N. Angelopoulos and S. Bates, "A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification," arXiv:2107.07511, 2021.
18. S. Bates, A. N. Angelopoulos, A. J. Romano, M. Sesia, and Y. Zhou, "Distribution-Free, Risk-Controlling Prediction Sets," Journal of the ACM, 2021.
19. Y. Geifman and R. El-Yaniv, "Selective Classification for Deep Neural Networks," NeurIPS, 2017.
20. C. J. Clopper and E. S. Pearson, "The Use of Confidence or Fiducial Limits Illustrated in the Case of the Binomial," Biometrika, 1934.
21. L. Li, W. Chu, J. Langford, and R. E. Schapire, "A Contextual-Bandit Approach to Personalized News Article Recommendation," WWW, 2010.
22. O. Chapelle and L. Li, "An Empirical Evaluation of Thompson Sampling," NeurIPS, 2011.
23. P. Auer, N. Cesa-Bianchi, and P. Fischer, "Finite-time Analysis of the Multiarmed Bandit Problem," Machine Learning, 2002.
24. T. Lattimore and C. Szepesvári, "Bandit Algorithms," Cambridge University Press, 2020.
25. M. E. Taylor and P. Stone, "Transfer Learning for Reinforcement Learning Domains: A Survey," JMLR, 2009.
26. F. Zhuang et al., "A Comprehensive Survey on Transfer Learning," Proceedings of the IEEE, 2021.
27. F. Liu et al., "RemoteCLIP: A Vision Language Foundation Model for Remote Sensing," IEEE TGRS, 2024.
28. J. Jakubik et al., "Foundation Models for Generalist Geospatial Artificial Intelligence," arXiv:2310.18660, 2023.
29. D. Hong et al., "SpectralGPT: Spectral Remote Sensing Foundation Model," IEEE TPAMI, 2024.
30. G. Mai et al., "On the Opportunities and Challenges of Foundation Models for GeoAI," ACM Transactions on Spatial Algorithms and Systems, 2024.
31. Y. Krechetova and D. Kochedykov, "Benchmarking LLMs in Agent Solving Multistep Geospatial Tasks," ACM SIGSPATIAL, 2025.
32. G. Diaz-Ireland et al., "GeoNatureAgent Benchmark: Benchmarking LLM Agents for Environmental Geospatial Analysis," ACM SIGSPATIAL, 2026; arXiv:2606.12821.
33. "GISAgentBench: A Practitioner-Sourced Benchmark for Evaluating LLM Agents on GIS Tasks," arXiv:2608.01645, 2026.
34. P. Helber et al., "EuroSAT: A Novel Dataset and Deep Learning Benchmark for Land Use and Land Cover Classification," IEEE JSTARS, 2019.
35. D. Wang et al., "Sen1Floods11: A Georeferenced Dataset to Train and Test Deep Learning Flood Algorithms," CVPR Workshops, 2020.
36. J. Wang et al., "LoveDA: A Remote Sensing Land-Cover Dataset for Domain Adaptive Semantic Segmentation," NeurIPS Datasets and Benchmarks, 2021.
37. H. Chen and Z. Shi, "A Spatial-Temporal Attention-Based Method and a New Dataset for Remote Sensing Image Change Detection," Remote Sensing, 2020.
