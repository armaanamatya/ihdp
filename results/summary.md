# IHDP results (cevae10: 10 replications, mean and standard error)

## Absolute error of the average treatment effect (all 747 children)

| Method | Error | SE | Relative error (mean) | Relative error (max) |
|---|---|---|---|---|
| Naive difference | 0.262 | 0.155 | 3.9% | 15.6% |
| IPW | 0.123 | 0.042 | 2.3% | 4.4% |
| AIPW (doubly robust) | 0.195 | 0.039 | 4.1% | 8.0% |
| Double ML (linear) | 0.745 | 0.499 | 10.3% | 49.8% |
| Causal forest | 0.562 | 0.326 | 8.4% | 32.8% |

## Root PEHE of per-child effects (held-out 30%)

| Method | sqrt PEHE | SE |
|---|---|---|
| Constant effect (AIPW ATE) | 4.756 | 2.829 |
| T-learner | 2.183 | 1.172 |
| Causal forest | 3.284 | 1.921 |

## Targeting (held-out, scored on true effects)

Mean true effect of the top 20% ranked by the causal forest: 9.071
Oracle top 20%: 9.481. Everyone (random targeting): 4.692
Lift over treating everyone: 1.93x, 95.7% of the oracle

## Refutations of the AIPW estimate (mean over reps)

Estimate 4.611
Placebo treatment (should be near 0): -0.120
Random common cause added (should match estimate): 4.629
80% subsets (should match estimate): 4.632
