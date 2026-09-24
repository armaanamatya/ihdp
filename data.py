"""Download and load the IHDP benchmark (Hill 2011, setting A, the NPCI simulation).

Two distributions of the same simulation:
  cevae10  10 replications as CSVs, from the CEVAE repo.
  ihdp100  100 replications with the fixed 672/75 train/test split used by Shalit et al. (2017)
           and most later papers. Its first 10 replications are the same children and outcomes
           as cevae10, in a different row order.

The true individual effect is mu1 - mu0 (noiseless potential outcomes), not y_cfactual - y_factual.
"""
from pathlib import Path
import urllib.request

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

URL = "https://raw.githubusercontent.com/AMLab-Amsterdam/CEVAE/master/datasets/IHDP/csv/ihdp_npci_{}.csv"
URL_100 = "http://www.fredjo.com/files/ihdp_npci_1-100.{}.npz"
DATA_DIR = Path(__file__).parent / "data"
SOURCES = {"cevae10": 10, "ihdp100": 100}
N_REPS = SOURCES["cevae10"]
TEST_SIZE = 0.3
COLS = ["t", "yf", "ycf", "mu0", "mu1"] + [f"x{i}" for i in range(1, 26)]


def load(rep: int, source: str = "cevae10") -> pd.DataFrame:
    if source == "ihdp100":
        return _load_100(rep)
    DATA_DIR.mkdir(exist_ok=True)
    path = DATA_DIR / f"ihdp_npci_{rep}.csv"
    if not path.exists():
        urllib.request.urlretrieve(URL.format(rep), path)
    df = pd.read_csv(path, header=None, names=COLS)
    df["t"] = df["t"].astype(int)
    df["x14"] = df["x14"] - 1  # this copy codes x14 as {1, 2}; every other binary is {0, 1}
    return df


_NPZ = {}


def _load_100(rep: int) -> pd.DataFrame:
    DATA_DIR.mkdir(exist_ok=True)
    parts = []
    for split in ["train", "test"]:
        if split not in _NPZ:
            path = DATA_DIR / f"ihdp_npci_1-100.{split}.npz"
            if not path.exists():
                urllib.request.urlretrieve(URL_100.format(split), path)
            _NPZ[split] = dict(np.load(path))
        z, r = _NPZ[split], rep - 1
        df = pd.DataFrame(z["x"][:, :, r], columns=[f"x{i}" for i in range(1, 26)])
        for col in ["t", "yf", "ycf", "mu0", "mu1"]:
            df[col] = z[col][:, r]
        df["split"] = split
        parts.append(df)
    df = pd.concat(parts, ignore_index=True)[COLS + ["split"]]
    df["t"] = df["t"].astype(int)
    df["x14"] = df["x14"] - 1  # coded {1, 2} here too
    return df


def split_indices(df: pd.DataFrame, rep: int):
    """Train/test indices: the published split when the source has one, else a seeded 70/30 split."""
    if "split" in df:
        return np.flatnonzero(df["split"] == "train"), np.flatnonzero(df["split"] == "test")
    idx = np.arange(len(df))
    return train_test_split(idx, test_size=TEST_SIZE, random_state=rep, stratify=df["t"].to_numpy())


def arrays(df: pd.DataFrame):
    X = df[[f"x{i}" for i in range(1, 26)]].to_numpy(dtype=float)
    t = df["t"].to_numpy()
    y = df["yf"].to_numpy()
    tau = (df["mu1"] - df["mu0"]).to_numpy()
    return X, t, y, tau


if __name__ == "__main__":
    import sys
    source = sys.argv[1] if len(sys.argv) > 1 else "cevae10"
    for r in range(1, SOURCES[source] + 1):
        df = load(r, source)
        X, t, y, tau = arrays(df)
        print(f"rep {r}: n={len(df)} treated={t.mean():.3f} true ATE={tau.mean():.3f}")
    binary = {c: sorted(df[c].unique()) for c in COLS[11:]}
    print("binary covariate levels (rep 10):", binary)
