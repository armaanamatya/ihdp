"""Download and load the IHDP benchmark (10 replications, CEVAE copy of Hill 2011 setting A).

Columns: treatment, y_factual, y_cfactual, mu0, mu1, x1..x25.
The true individual effect is mu1 - mu0 (noiseless potential outcomes), not y_cfactual - y_factual.
"""
from pathlib import Path
import urllib.request

import numpy as np
import pandas as pd

URL = "https://raw.githubusercontent.com/AMLab-Amsterdam/CEVAE/master/datasets/IHDP/csv/ihdp_npci_{}.csv"
DATA_DIR = Path(__file__).parent / "data"
N_REPS = 10
COLS = ["t", "yf", "ycf", "mu0", "mu1"] + [f"x{i}" for i in range(1, 26)]


def load(rep: int) -> pd.DataFrame:
    DATA_DIR.mkdir(exist_ok=True)
    path = DATA_DIR / f"ihdp_npci_{rep}.csv"
    if not path.exists():
        urllib.request.urlretrieve(URL.format(rep), path)
    df = pd.read_csv(path, header=None, names=COLS)
    df["t"] = df["t"].astype(int)
    df["x14"] = df["x14"] - 1  # this copy codes x14 as {1, 2}; every other binary is {0, 1}
    return df


def arrays(df: pd.DataFrame):
    X = df[[f"x{i}" for i in range(1, 26)]].to_numpy(dtype=float)
    t = df["t"].to_numpy()
    y = df["yf"].to_numpy()
    tau = (df["mu1"] - df["mu0"]).to_numpy()
    return X, t, y, tau


if __name__ == "__main__":
    for r in range(1, N_REPS + 1):
        df = load(r)
        X, t, y, tau = arrays(df)
        print(f"rep {r}: n={len(df)} treated={t.mean():.3f} true ATE={tau.mean():.3f}")
    binary = {c: sorted(df[c].unique()) for c in COLS[11:]}
    print("binary covariate levels (rep 10):", binary)
