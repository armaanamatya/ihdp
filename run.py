"""Estimate the IHDP treatment effect five ways and score each against the known truth.

Every model choice below was fixed before any error against ground truth was looked at:
  propensity  logistic regression on standardized covariates, clipped to [CLIP, 1 - CLIP]
  outcome     gradient-boosted trees (sklearn defaults), cross-fitted over K_FOLDS folds
  split       70/30 train/test per replication, seed = replication number (cevae10), or the
              published 672/75 split (ihdp100)

Usage: python run.py [--source cevae10|ihdp100]
Outputs: results/ (cevae10) or results/ihdp100/: results.json, summary.md, targeting_curve.png
"""
import argparse
import json
from pathlib import Path

import numpy as np
from econml.dml import CausalForestDML, LinearDML
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from data import SOURCES, arrays, load, split_indices

CLIP = 0.01
K_FOLDS = 5
RESULTS = Path(__file__).parent / "results"


def out_dir(source):
    return RESULTS if source == "cevae10" else RESULTS / source


def propensity_model():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))


def outcome_model():
    return GradientBoostingRegressor(random_state=0)


# ---------- ATE estimators ----------

def naive(X, t, y):
    return y[t == 1].mean() - y[t == 0].mean()


def fit_propensity(X, t):
    e = propensity_model().fit(X, t).predict_proba(X)[:, 1]
    return np.clip(e, CLIP, 1 - CLIP)


def ipw(X, t, y):
    """Normalized (Hajek) inverse-propensity weighting."""
    e = fit_propensity(X, t)
    w1, w0 = t / e, (1 - t) / (1 - e)
    return (w1 * y).sum() / w1.sum() - (w0 * y).sum() / w0.sum()


def aipw(X, t, y, seed=0):
    """Doubly robust AIPW with cross-fitted propensity and outcome models."""
    return aipw_scores(X, t, y, seed).mean()


def aipw_scores(X, t, y, seed=0):
    """Per-row AIPW scores; their mean is the estimate and their spread gives its standard error."""
    psi = np.zeros(len(y))
    for tr, te in KFold(K_FOLDS, shuffle=True, random_state=seed).split(X):
        e = np.clip(propensity_model().fit(X[tr], t[tr]).predict_proba(X[te])[:, 1], CLIP, 1 - CLIP)
        m1 = outcome_model().fit(X[tr][t[tr] == 1], y[tr][t[tr] == 1]).predict(X[te])
        m0 = outcome_model().fit(X[tr][t[tr] == 0], y[tr][t[tr] == 0]).predict(X[te])
        tt, yy = t[te], y[te]
        psi[te] = m1 - m0 + tt * (yy - m1) / e - (1 - tt) * (yy - m0) / (1 - e)
    return psi


def linear_dml():
    return LinearDML(model_y=outcome_model(), model_t=propensity_model(),
                     discrete_treatment=True, cv=K_FOLDS, random_state=0)


def causal_forest():
    return CausalForestDML(model_y=outcome_model(), model_t=propensity_model(),
                           discrete_treatment=True, cv=K_FOLDS, n_estimators=1000,
                           min_samples_leaf=5, random_state=0)


# ---------- CATE baselines ----------

def t_learner(Xtr, ttr, ytr, Xte):
    m1 = outcome_model().fit(Xtr[ttr == 1], ytr[ttr == 1])
    m0 = outcome_model().fit(Xtr[ttr == 0], ytr[ttr == 0])
    return m1.predict(Xte) - m0.predict(Xte)


def pehe(pred, true):
    return float(np.sqrt(np.mean((pred - true) ** 2)))


# ---------- refutations (DoWhy-style, implemented by hand) ----------

def refute(X, t, y, est, rng):
    placebo = aipw(X, rng.permutation(t), y)
    extra = np.column_stack([X, rng.normal(size=len(y))])
    random_cause = aipw(extra, t, y)
    subsets = []
    for _ in range(5):
        idx = rng.choice(len(y), int(0.8 * len(y)), replace=False)
        subsets.append(aipw(X[idx], t[idx], y[idx]))
    return {"estimate": est, "placebo_treatment": placebo,
            "random_common_cause": random_cause, "subset_80pct_mean": float(np.mean(subsets))}


# ---------- targeting curve (scored on TRUE effects, not a Qini curve) ----------

def targeting_curve(pred, true, grid):
    """Mean true effect among the top-q fraction ranked by predicted effect."""
    order = np.argsort(-pred)
    return [float(true[order[: max(1, int(round(q * len(true))))]].mean()) for q in grid]


def main(source="cevae10"):
    OUT = out_dir(source)
    OUT.mkdir(parents=True, exist_ok=True)
    n_reps = SOURCES[source]
    grid = np.linspace(0.05, 1.0, 20)
    true_ates = []
    ate_err = {k: [] for k in ["Naive difference", "IPW", "AIPW (doubly robust)", "Double ML (linear)", "Causal forest"]}
    pehe_te = {k: [] for k in ["Constant effect (AIPW ATE)", "T-learner", "Causal forest"]}
    curves = {"model": [], "oracle": [], "random": []}
    refutes = []

    for rep in range(1, n_reps + 1):
        df = load(rep, source)
        X, t, y, tau = arrays(df)
        ate_true = tau.mean()
        true_ates.append(float(ate_true))

        cf = causal_forest().fit(y, t, X=X)
        ests = {
            "Naive difference": naive(X, t, y),
            "IPW": ipw(X, t, y),
            "AIPW (doubly robust)": aipw(X, t, y),
            "Double ML (linear)": float(linear_dml().fit(y, t, X=None, W=X).ate()),
            "Causal forest": float(cf.ate(X)),
        }
        for k, v in ests.items():
            ate_err[k].append(abs(v - ate_true))

        idx_tr, idx_te = split_indices(df, rep)
        Xtr, ttr, ytr = X[idx_tr], t[idx_tr], y[idx_tr]
        Xte, tau_te = X[idx_te], tau[idx_te]
        cf_tr = causal_forest().fit(ytr, ttr, X=Xtr)
        cf_pred = cf_tr.effect(Xte)
        pehe_te["Constant effect (AIPW ATE)"].append(pehe(np.full(len(idx_te), aipw(Xtr, ttr, ytr)), tau_te))
        pehe_te["T-learner"].append(pehe(t_learner(Xtr, ttr, ytr, Xte), tau_te))
        pehe_te["Causal forest"].append(pehe(cf_pred, tau_te))

        curves["model"].append(targeting_curve(cf_pred, tau_te, grid))
        curves["oracle"].append(targeting_curve(tau_te, tau_te, grid))
        curves["random"].append([float(tau_te.mean())] * len(grid))

        refutes.append(refute(X, t, y, ests["AIPW (doubly robust)"], np.random.default_rng(rep)))
        print(f"rep {rep}: true ATE {ate_true:.3f} | " + ", ".join(f"{k} {v:.3f}" for k, v in ests.items()))

    def ms(v):
        v = np.asarray(v)
        return {"mean": float(v.mean()), "se": float(v.std(ddof=1) / np.sqrt(len(v)))}

    results = {
        "config": {"source": source, "reps": n_reps, "clip": CLIP, "k_folds": K_FOLDS},
        "ate_abs_error": {k: ms(v) for k, v in ate_err.items()},
        "ate_abs_error_per_rep": ate_err,
        "true_ate_per_rep": true_ates,
        "ate_rel_error_pct": {k: {"mean": float(np.mean(np.array(v) / true_ates) * 100),
                                  "max": float(np.max(np.array(v) / true_ates) * 100)} for k, v in ate_err.items()},
        "sqrt_pehe_test": {k: ms(v) for k, v in pehe_te.items()},
        "targeting_curve": {"grid": grid.tolist(), **{k: np.mean(v, axis=0).tolist() for k, v in curves.items()}},
        "refutations_per_rep": refutes,
    }
    (OUT / "results.json").write_text(json.dumps(results, indent=2))

    lines = [f"# IHDP results ({source}: {n_reps} replications, mean and standard error)", "",
             "## Absolute error of the average treatment effect (all 747 children)", "",
             "| Method | Error | SE | Relative error (mean) | Relative error (max) |", "|---|---|---|---|---|"]
    lines += [f"| {k} | {v['mean']:.3f} | {v['se']:.3f} | {results['ate_rel_error_pct'][k]['mean']:.1f}% | "
              f"{results['ate_rel_error_pct'][k]['max']:.1f}% |" for k, v in results["ate_abs_error"].items()]
    lines += ["", f"## Root PEHE of per-child effects ({'held-out 30%' if source == 'cevae10' else 'published 75-child test split'})", "", "| Method | sqrt PEHE | SE |", "|---|---|---|"]
    lines += [f"| {k} | {v['mean']:.3f} | {v['se']:.3f} |" for k, v in results["sqrt_pehe_test"].items()]
    tc = results["targeting_curve"]
    i20 = int(np.argmin(np.abs(grid - 0.2)))
    lines += ["", "## Targeting (held-out, scored on true effects)", "",
              f"Mean true effect of the top 20% ranked by the causal forest: {tc['model'][i20]:.3f}",
              f"Oracle top 20%: {tc['oracle'][i20]:.3f}. Everyone (random targeting): {tc['random'][i20]:.3f}",
              f"Lift over treating everyone: {tc['model'][i20] / tc['random'][i20]:.2f}x, "
              f"{tc['model'][i20] / tc['oracle'][i20] * 100:.1f}% of the oracle"]
    r = refutes
    lines += ["", "## Refutations of the AIPW estimate (mean over reps)", "",
              f"Estimate {np.mean([x['estimate'] for x in r]):.3f}",
              f"Placebo treatment (should be near 0): {np.mean([x['placebo_treatment'] for x in r]):.3f}",
              f"Random common cause added (should match estimate): {np.mean([x['random_common_cause'] for x in r]):.3f}",
              f"80% subsets (should match estimate): {np.mean([x['subset_80pct_mean'] for x in r]):.3f}"]
    (OUT / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(grid * 100, tc["oracle"], label="Oracle (true effect ranking)")
    ax.plot(grid * 100, tc["model"], label="Causal forest ranking")
    ax.plot(grid * 100, tc["random"], "--", label="Random targeting")
    ax.set_xlabel("Share of children targeted (%)")
    ax.set_ylabel("Mean true effect in targeted group")
    ax.set_title(f"IHDP targeting curve (held-out, {n_reps} reps)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "targeting_curve.png", dpi=150)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=list(SOURCES), default="cevae10")
    main(ap.parse_args().source)
