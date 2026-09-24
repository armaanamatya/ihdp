# IHDP: causal effect estimation, from benchmark to production

How well do causal inference methods recover a treatment effect from observational data, when the true answer is known? And what does it take to serve per-person effect estimates in production?

This repo compares classical and deep causal estimators on the IHDP benchmark, then exports the served model to ONNX behind a FastAPI scoring service.

## Dataset

IHDP is a **semi-synthetic** benchmark (Hill, 2011). The covariates come from a real randomized trial of an early-childhood health and development intervention for low-birth-weight infants. Hill made it observational by removing a non-random subset of treated children (18.6% treated remain) and simulated the outcomes, so each child's true effect is known.

This repo uses the 10 replications distributed with CEVAE: 747 children, 6 continuous and 19 binary covariates. `data.py` downloads them into `data/` on first run.

## Protocol

Every model choice was fixed before looking at any error against the ground truth, and nothing was tuned afterwards.

- **Average effect (ATE):** fit on all 747 children, report the absolute error against the true ATE.
- **Per-child effects (CATE):** fit on a 70% split (seed = replication), report root PEHE on the held-out 30%.
- 10 replications, mean with standard error. The true ATE is about 4 in nine replications and 10.5 in replication 9, whose extreme effect surface dominates most averages.

## Classical estimators

| Estimator | What it does |
|---|---|
| Naive difference | Mean outcome of treated minus untreated. Ignores confounding. |
| IPW | Normalized inverse-propensity weighting, logistic propensity clipped to [0.01, 0.99]. |
| AIPW (doubly robust) | Outcome models plus propensity weighting, cross-fitted over 5 folds. |
| Double ML (linear) | EconML `LinearDML`: residualize outcome and treatment, regress one residual on the other. |
| Causal forest | EconML `CausalForestDML`: per-child effects from a generalized random forest. |

Outcome models are gradient-boosted trees (sklearn defaults). Per-child effects are also compared with a T-learner and a constant-effect baseline.

## Results

**Average effect: absolute error**

| Method | Error | SE |
|---|---|---|
| Naive difference | 0.262 | 0.155 |
| IPW | **0.123** | 0.042 |
| AIPW (doubly robust) | 0.195 | 0.039 |
| Double ML (linear) | 0.745 | 0.499 |
| Causal forest | 0.562 | 0.326 |

**Per-child effects: root PEHE on held-out children**

| Method | sqrt PEHE | SE |
|---|---|---|
| Constant effect | 4.756 | 2.829 |
| T-learner | **2.183** | 1.172 |
| Causal forest | 3.284 | 1.921 |

**Targeting.** Ranking held-out children by the causal forest's predicted effect and treating the top 20% gives a mean true effect of 9.07, against 4.69 for treating everyone and 9.48 for a perfect ranking. The curve is scored on the known true effects. It is not a Qini curve: treatment in IHDP is not randomized, so an observed-outcome Qini would be biased.

![Targeting curve](results/targeting_curve.png)

**Refutations of the AIPW estimate** (mean over replications, written by hand in the style of DoWhy's refuters)

| Test | Result | Expected |
|---|---|---|
| Estimate | 4.611 | |
| Placebo treatment (shuffled) | -0.120 | near 0 |
| Random common cause added | 4.629 | unchanged |
| 80% random subsets | 4.632 | unchanged |

**Takeaways**

- Simple weighting (IPW, AIPW) recovered the average effect best. The heavier models did worst on the average, mostly because of replication 9.
- For per-child effects the T-learner beat the causal forest, and both beat assuming one effect for everyone.
- Even with an imperfect per-child fit, the causal forest's ranking captured most of the gain from targeting (9.07 of an oracle 9.48).

Full numbers: `results/summary.md`, `results/results.json`.

## Deep models

Two neural CATE models in PyTorch, scored with the same protocol:

- **TARNet** (Shalit et al., 2017): a shared representation trunk with one outcome head per arm.
- **DragonNet** (Shi et al., 2019): TARNet plus a propensity head and targeted regularization.

Hyperparameters follow the papers and were fixed up front: trunk 3 x 200 ELU, heads 2 x 100 ELU, Adam lr 1e-3, batch 64, L2 1e-4, early stopping on factual validation loss (20% of training rows). The true effects are never used in training or model selection.

| Method | ATE error | SE | sqrt PEHE | SE |
|---|---|---|---|---|
| TARNet | 0.229 | 0.064 | **1.250** | 0.522 |
| DragonNet | 0.407 | 0.167 | 1.321 | 0.568 |

Both neural models cut the held-out per-child error well below the best classical model (T-learner, 2.183), while the simple weighting estimators remain the most accurate for the average effect.

Full numbers: `results/deep_summary.md`, `results/deep_results.json`.

## Setup

```
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
.venv/Scripts/python data.py   # downloads the 10 replications into data/
.venv/Scripts/python run.py    # classical estimators, writes results/
.venv/Scripts/python run_deep.py   # TARNet and DragonNet
```

## References

- J. L. Hill. Bayesian nonparametric modeling for causal inference. JCGS, 2011.
- C. Louizos et al. Causal effect inference with deep latent-variable models (CEVAE). NeurIPS, 2017.
- U. Shalit, F. Johansson, D. Sontag. Estimating individual treatment effect: generalization bounds and algorithms. ICML, 2017.
- C. Shi, D. Blei, V. Veitch. Adapting neural networks for the estimation of treatment effects. NeurIPS, 2019.
- V. Chernozhukov et al. Double/debiased machine learning for treatment and structural parameters. Econometrics Journal, 2018.
- S. Wager, S. Athey. Estimation and inference of heterogeneous treatment effects using random forests. JASA, 2018.
