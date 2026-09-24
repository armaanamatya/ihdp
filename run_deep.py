"""Score TARNet and DragonNet with the same protocol as run.py.

ATE: fit on all 747 children, average predicted effect, absolute error vs truth.
PEHE: fit on the same 70% split run.py uses (seed = replication), score the held-out 30%.
Writes results/deep_results.json and results/deep_summary.md.
"""
import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

from data import N_REPS, arrays, load
from deep import predict_cate, train
from run import TEST_SIZE, pehe

OUT = Path(__file__).parent / "results"


def main():
    ate_err = {"TARNet": [], "DragonNet": []}
    pehe_te = {"TARNet": [], "DragonNet": []}
    for rep in range(1, N_REPS + 1):
        X, t, y, tau = arrays(load(rep))
        idx_tr, idx_te = train_test_split(np.arange(len(y)), test_size=TEST_SIZE, random_state=rep, stratify=t)
        row = []
        for name, dragon in [("TARNet", False), ("DragonNet", True)]:
            full = train(X, t, y, dragon, seed=rep)
            ate_err[name].append(abs(predict_cate(full, X).mean() - tau.mean()))
            part = train(X[idx_tr], t[idx_tr], y[idx_tr], dragon, seed=rep)
            pehe_te[name].append(pehe(predict_cate(part, X[idx_te]), tau[idx_te]))
            row.append(f"{name} ATE err {ate_err[name][-1]:.3f} PEHE {pehe_te[name][-1]:.3f}")
        print(f"rep {rep}: " + " | ".join(row))

    def ms(v):
        v = np.asarray(v)
        return {"mean": float(v.mean()), "se": float(v.std(ddof=1) / np.sqrt(len(v)))}

    res = {"ate_abs_error": {k: ms(v) for k, v in ate_err.items()},
           "sqrt_pehe_test": {k: ms(v) for k, v in pehe_te.items()},
           "ate_abs_error_per_rep": ate_err, "sqrt_pehe_test_per_rep": pehe_te}
    (OUT / "deep_results.json").write_text(json.dumps(res, indent=2))
    lines = ["# Deep models (10 replications, mean and standard error)", "", "| Method | ATE error | SE | sqrt PEHE | SE |", "|---|---|---|---|---|"]
    lines += [f"| {k} | {res['ate_abs_error'][k]['mean']:.3f} | {res['ate_abs_error'][k]['se']:.3f} | "
              f"{res['sqrt_pehe_test'][k]['mean']:.3f} | {res['sqrt_pehe_test'][k]['se']:.3f} |" for k in ate_err]
    print("\n".join(lines))
    (OUT / "deep_summary.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
