"""How strong would an unobserved confounder have to be to explain the effect away?

Omitted-variable-bias sensitivity analysis (Cinelli and Hazlett, 2020) on a linear
regression-adjusted estimate: regress the outcome on treatment and all 25 covariates.

Robustness value RV: a confounder that explains RV of the residual variance of both the
treatment and the outcome would bring the estimate to zero. RV_alpha: the same, but only
enough to make the effect no longer significant at the 5% level. Both are compared with the
strongest OBSERVED covariate's partial R2 with the treatment and with the outcome.

Usage: python sensitivity.py [--source cevae10|ihdp100]
Writes sensitivity_results.json and sensitivity_summary.md next to run.py's results.
"""
import argparse
import json

import numpy as np
from scipy import stats

from data import SOURCES, arrays, load
from run import out_dir


def ols_t(y, D, cols):
    """t statistic and residual degrees of freedom for column `cols` of design D (with intercept)."""
    A = np.column_stack([np.ones(len(y)), D])
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - A @ beta
    dof = len(y) - A.shape[1]
    sigma2 = resid @ resid / dof
    cov = sigma2 * np.linalg.pinv(A.T @ A)
    j = cols + 1
    return beta[j], beta[j] / np.sqrt(cov[j, j]), dof


def robustness_value(f):
    f = max(f, 0.0)
    return 0.5 * (np.sqrt(f ** 4 + 4 * f ** 2) - f ** 2)


def partial_r2(t_stat, dof):
    return t_stat ** 2 / (t_stat ** 2 + dof)


def main(source="cevae10"):
    OUT = out_dir(source)
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for rep in range(1, SOURCES[source] + 1):
        X, t, y, tau = arrays(load(rep, source))
        D = np.column_stack([t, X])
        est, t_stat, dof = ols_t(y, D, 0)
        f = abs(t_stat) / np.sqrt(dof)
        f_crit = stats.t.ppf(0.975, dof - 1) / np.sqrt(dof - 1)
        rv = robustness_value(f)
        rv_alpha = robustness_value(f - f_crit)

        # strongest observed covariate: partial R2 with the outcome (given treatment and the rest)
        # and with the treatment (given the rest)
        r2_y = max(partial_r2(ols_t(y, D, k)[1], dof) for k in range(1, D.shape[1]))
        r2_d = max(partial_r2(ols_t(t.astype(float), X, k)[1], len(y) - X.shape[1] - 1) for k in range(X.shape[1]))

        rows.append({"rep": rep, "truth": float(tau.mean()), "ols_estimate": float(est), "t": float(t_stat),
                     "rv": float(rv), "rv_alpha_05": float(rv_alpha),
                     "max_observed_partial_r2_outcome": float(r2_y), "max_observed_partial_r2_treatment": float(r2_d)})

    def med(k):
        return float(np.median([r[k] for r in rows]))

    summary = {k: med(k) for k in ["ols_estimate", "rv", "rv_alpha_05",
                                   "max_observed_partial_r2_outcome", "max_observed_partial_r2_treatment"]}
    summary["ols_abs_error_mean"] = float(np.mean([abs(r["ols_estimate"] - r["truth"]) for r in rows]))
    (OUT / "sensitivity_results.json").write_text(json.dumps({"summary": summary, "per_rep": rows}, indent=2))

    n = len(rows)
    lines = [f"# Sensitivity to an unobserved confounder ({source}: {n} replications, medians)", "",
             "| Quantity | Value |", "|---|---|",
             f"| Regression-adjusted estimate | {summary['ols_estimate']:.3f} |",
             f"| Robustness value (confounder needed to reach zero) | {summary['rv']:.1%} |",
             f"| Robustness value, 5% significance | {summary['rv_alpha_05']:.1%} |",
             f"| Strongest observed covariate, partial R2 with outcome | {summary['max_observed_partial_r2_outcome']:.1%} |",
             f"| Strongest observed covariate, partial R2 with treatment | {summary['max_observed_partial_r2_treatment']:.1%} |"]
    (OUT / "sensitivity_summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=list(SOURCES), default="cevae10")
    main(ap.parse_args().source)
