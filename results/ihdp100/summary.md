# IHDP results (ihdp100: 100 replications, mean and standard error)

## Absolute error of the average treatment effect (all 747 children)

| Method | Error | SE | Relative error (mean) | Relative error (max) |
|---|---|---|---|---|
| Naive difference | 0.285 | 0.039 | 2.9% | 17.0% |
| IPW | 0.125 | 0.012 | 2.3% | 9.1% |
| AIPW (doubly robust) | 0.202 | 0.017 | 3.9% | 14.6% |
| Double ML (linear) | 0.774 | 0.150 | 5.9% | 85.1% |
| Causal forest | 0.552 | 0.106 | 4.9% | 68.3% |

## Root PEHE of per-child effects (published 75-child test split)

| Method | sqrt PEHE | SE |
|---|---|---|
| Constant effect (AIPW ATE) | 5.712 | 0.888 |
| T-learner | 2.077 | 0.328 |
| Causal forest | 3.843 | 0.621 |

## Targeting (held-out, scored on true effects)

Mean true effect of the top 20% ranked by the causal forest: 9.217
Oracle top 20%: 9.901. Everyone (random targeting): 4.270
Lift over treating everyone: 2.16x, 93.1% of the oracle

## Refutations of the AIPW estimate (mean over reps)

Estimate 4.455
Placebo treatment (should be near 0): -0.086
Random common cause added (should match estimate): 4.434
80% subsets (should match estimate): 4.457
