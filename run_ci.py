"""Confidence intervals for the average effect, and how often they contain the truth.

AIPW: 95% interval from the influence function (standard error of the cross-fitted scores).
IPW:  95% percentile interval from B_BOOT bootstrap resamples, refitting the propensity each time.
Coverage is the share of replications whose interval contains that replication's true effect.

Usage: python run_ci.py [--source cevae10|ihdp100]
Writes ci_results.json and ci_summary.md next to run.py's results.
"""
import argparse
import json

import numpy as np

from data import SOURCES, arrays, load
from run import aipw_scores, ipw, out_dir

B_BOOT = 200
Z = 1.959964


def main(source="cevae10"):
    OUT = out_dir(source)
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for rep in range(1, SOURCES[source] + 1):
        X, t, y, tau = arrays(load(rep, source))
        truth = tau.mean()

        psi = aipw_scores(X, t, y)
        est, se = psi.mean(), psi.std(ddof=1) / np.sqrt(len(psi))
        aipw_ci = (est - Z * se, est + Z * se)

        rng = np.random.default_rng(rep)
        boots = []
        for _ in range(B_BOOT):
            i = rng.integers(0, len(y), len(y))
            boots.append(ipw(X[i], t[i], y[i]))
        ipw_ci = tuple(np.percentile(boots, [2.5, 97.5]))

        rows.append({"rep": rep, "truth": float(truth),
                     "aipw": {"est": float(est), "lo": float(aipw_ci[0]), "hi": float(aipw_ci[1])},
                     "ipw": {"est": float(ipw(X, t, y)), "lo": float(ipw_ci[0]), "hi": float(ipw_ci[1])}})
        if rep % 10 == 0:
            print(f"rep {rep} done")

    summary = {}
    for m in ["ipw", "aipw"]:
        cover = [r[m]["lo"] <= r["truth"] <= r[m]["hi"] for r in rows]
        width = [r[m]["hi"] - r[m]["lo"] for r in rows]
        summary[m] = {"coverage": float(np.mean(cover)), "mean_width": float(np.mean(width)),
                      "median_width": float(np.median(width))}
    (OUT / "ci_results.json").write_text(json.dumps({"summary": summary, "per_rep": rows}, indent=2))

    n = len(rows)
    lines = [f"# 95% confidence intervals for the average effect ({source}: {n} replications)", "",
             "| Method | Interval | Coverage of the true effect | Mean width | Median width |", "|---|---|---|---|---|",
             f"| IPW | bootstrap percentile, {B_BOOT} resamples | {summary['ipw']['coverage']:.0%} | "
             f"{summary['ipw']['mean_width']:.3f} | {summary['ipw']['median_width']:.3f} |",
             f"| AIPW | influence function | {summary['aipw']['coverage']:.0%} | "
             f"{summary['aipw']['mean_width']:.3f} | {summary['aipw']['median_width']:.3f} |"]
    (OUT / "ci_summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=list(SOURCES), default="cevae10")
    main(ap.parse_args().source)
