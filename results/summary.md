# IHDP results (10 replications, mean and standard error)

## Absolute error of the average treatment effect (all 747 children)

| Method | Error | SE |
|---|---|---|
| Naive difference | 0.262 | 0.155 |
| IPW | 0.123 | 0.042 |
| AIPW (doubly robust) | 0.195 | 0.039 |
| Double ML (linear) | 0.745 | 0.499 |
| Causal forest | 0.562 | 0.326 |

## Root PEHE of per-child effects (held-out 30%)

| Method | sqrt PEHE | SE |
|---|---|---|
| Constant effect (AIPW ATE) | 4.756 | 2.829 |
| T-learner | 2.183 | 1.172 |
| Causal forest | 3.284 | 1.921 |

## Targeting (held-out, scored on true effects)

Mean true effect of the top 20% ranked by the causal forest: 9.071
Oracle top 20%: 9.481. Everyone (random targeting): 4.692

## Refutations of the AIPW estimate (mean over reps)

Estimate 4.611
Placebo treatment (should be near 0): -0.120
Random common cause added (should match estimate): 4.629
80% subsets (should match estimate): 4.632
