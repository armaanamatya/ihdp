"""TARNet (Shalit et al. 2017) and DragonNet (Shi et al. 2019) in PyTorch.

Hyperparameters follow the papers and were fixed before any error against ground truth was
looked at: shared trunk 3 x 200 ELU, outcome heads 2 x 100 ELU, Adam lr 1e-3, batch 64,
L2 1e-4, early stopping on FACTUAL validation loss (20% of the training rows, patience 30).
DragonNet adds a propensity head (alpha = 1) and targeted regularization (beta = 1).
The true effects are never used in training or model selection.
"""
import numpy as np
import torch
from torch import nn

N_CONT = 6  # x1..x6 are continuous; x7..x25 are binary


class Normalizer(nn.Module):
    """Standardizes the continuous covariates inside the graph so the exported model takes raw features."""

    def __init__(self, mean, std):
        super().__init__()
        self.register_buffer("mean", torch.as_tensor(mean, dtype=torch.float32))
        self.register_buffer("std", torch.as_tensor(std, dtype=torch.float32))

    def forward(self, x):
        return torch.cat([(x[:, :N_CONT] - self.mean) / self.std, x[:, N_CONT:]], dim=1)


def mlp(d_in, width, depth):
    layers = []
    for i in range(depth):
        layers += [nn.Linear(d_in if i == 0 else width, width), nn.ELU()]
    return nn.Sequential(*layers)


class CateNet(nn.Module):
    def __init__(self, d_in, x_mean, x_std, y_mean, y_std, dragon: bool):
        super().__init__()
        self.norm = Normalizer(x_mean, x_std)
        self.trunk = mlp(d_in, 200, 3)
        self.h0 = nn.Sequential(mlp(200, 100, 2), nn.Linear(100, 1))
        self.h1 = nn.Sequential(mlp(200, 100, 2), nn.Linear(100, 1))
        self.dragon = dragon
        self.g = nn.Linear(200, 1)  # propensity head (trained only for DragonNet)
        self.eps = nn.Parameter(torch.zeros(1))  # targeted-regularization epsilon
        self.register_buffer("y_mean", torch.tensor(float(y_mean)))
        self.register_buffer("y_std", torch.tensor(float(y_std)))

    def heads(self, x):
        z = self.trunk(self.norm(x))
        return self.h0(z).squeeze(1), self.h1(z).squeeze(1), torch.sigmoid(self.g(z)).squeeze(1)

    def forward(self, x):
        """Serving output in outcome units: columns [cate, propensity]."""
        q0, q1, e = self.heads(x)
        return torch.stack([(q1 - q0) * self.y_std, e], dim=1)


def _loss(model, x, t, y):
    q0, q1, e = model.heads(x)
    q = torch.where(t == 1, q1, q0)
    loss = ((q - y) ** 2).mean()
    if model.dragon:
        e = e.clamp(0.01, 0.99)
        loss = loss + nn.functional.binary_cross_entropy(e, t)
        h = t / e - (1 - t) / (1 - e)
        loss = loss + ((y - (q + model.eps * h)) ** 2).mean()
    return loss


def train(X, t, y, dragon: bool, seed: int, max_epochs=300, patience=30):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y))
    n_val = int(0.2 * len(y))
    va, tr = idx[:n_val], idx[n_val:]

    xm, xs = X[tr, :N_CONT].mean(0), X[tr, :N_CONT].std(0) + 1e-8
    ym, ys = y[tr].mean(), y[tr].std()
    model = CateNet(X.shape[1], xm, xs, ym, ys, dragon)
    Xt = torch.tensor(X, dtype=torch.float32)
    tt = torch.tensor(t, dtype=torch.float32)
    yt = torch.tensor((y - ym) / ys, dtype=torch.float32)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    best, best_state, bad = float("inf"), None, 0
    for _ in range(max_epochs):
        model.train()
        for b in np.array_split(rng.permutation(tr), max(1, len(tr) // 64)):
            opt.zero_grad()
            _loss(model, Xt[b], tt[b], yt[b]).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            v = _loss(model, Xt[va], tt[va], yt[va]).item()
        if v < best - 1e-5:
            best, bad = v, 0
            best_state = {k: p.clone() for k, p in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    return model


def predict_cate(model, X):
    with torch.no_grad():
        return model(torch.tensor(X, dtype=torch.float32))[:, 0].numpy()
