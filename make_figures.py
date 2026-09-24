"""Draw the README figures from the saved results (100-replication release).

Every figure is written twice, figures/<name>.png (light) and figures/<name>-dark.png (dark),
so the README can switch with the reader's GitHub theme.
Run after run.py, run_deep.py, run_ci.py and sensitivity.py with --source ihdp100.
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from data import arrays, load
from run import fit_propensity

ROOT = Path(__file__).parent
RES = ROOT / "results" / "ihdp100"
FIG = ROOT / "figures"

THEMES = {
    "light": {"surface": "#fcfcfb", "ink": "#0b0b0b", "ink2": "#52514e", "muted": "#898781",
              "grid": "#e1e0d9", "axis": "#c3c2b7", "blue": "#2a78d6", "orange": "#eb6834",
              "neutral": "#b5b3aa", "bad": "#d03b3b"},
    "dark": {"surface": "#1a1a19", "ink": "#ffffff", "ink2": "#c3c2b7", "muted": "#898781",
             "grid": "#2c2c2a", "axis": "#383835", "blue": "#3987e5", "orange": "#d95926",
             "neutral": "#5f5e59", "bad": "#d03b3b"},
}


def load_json(name):
    return json.loads((RES / name).read_text())


def style(ax, th, xgrid=True):
    ax.set_facecolor(th["surface"])
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    for side in ["left", "bottom"]:
        ax.spines[side].set_color(th["axis"])
    ax.tick_params(colors=th["muted"], labelcolor=th["ink2"], length=0, labelsize=9)
    ax.grid(axis="x" if xgrid else "y", color=th["grid"], linewidth=0.8)
    ax.set_axisbelow(True)
    ax.xaxis.label.set_color(th["ink2"])
    ax.yaxis.label.set_color(th["ink2"])


def new_fig(th, w=7.0, h=3.6):
    fig, ax = plt.subplots(figsize=(w, h))
    fig.patch.set_facecolor(th["surface"])
    return fig, ax


def title(fig, th, text, sub):
    fig.text(0.02, 0.965, text, color=th["ink"], fontsize=12, fontweight="bold", va="top")
    fig.text(0.02, 0.895, sub, color=th["ink2"], fontsize=9, va="top")


def save(fig, name, mode):
    FIG.mkdir(exist_ok=True)
    fig.savefig(FIG / (f"{name}.png" if mode == "light" else f"{name}-dark.png"), dpi=160,
                facecolor=fig.get_facecolor())
    plt.close(fig)


def hbar(ax, th, labels, values, fmt, highlight=None, xmax=None):
    y = np.arange(len(labels))[::-1]
    colors = [th["blue"] if (highlight is None or l in highlight) else th["neutral"] for l in labels]
    ax.barh(y, values, height=0.56, color=colors, edgecolor=th["surface"], linewidth=2)
    ax.set_yticks(y, labels)
    top = xmax or max(values) * 1.18
    ax.set_xlim(0, top)
    for yi, v in zip(y, values):
        ax.text(v + top * 0.01, yi, fmt(v), va="center", ha="left", color=th["ink"], fontsize=9)


# ---------- figures ----------

def fig_overlap(th, mode):
    X, t, y, tau = arrays(load(1, "ihdp100"))
    e = fit_propensity(X, t)
    fig, ax = new_fig(th)
    bins = np.linspace(0, 1, 26)
    for grp, col, lab in [(0, th["orange"], "Did not get the program"), (1, th["blue"], "Got the program")]:
        w = np.full((t == grp).sum(), 100 / (t == grp).sum())
        ax.hist(e[t == grp], bins=bins, weights=w, histtype="step", linewidth=2, color=col,
                label=f"{lab} ({(t == grp).sum()} children)")
    style(ax, th, xgrid=False)
    ax.set_xlim(0, 0.7)
    ax.set_xlabel("Estimated chance of getting the program, from the child's background")
    ax.set_ylabel("Share of the group (%)")
    leg = ax.legend(frameon=False, fontsize=9, labelcolor=th["ink2"], loc="upper right")
    title(fig, th, "The two groups are not alike",
          "Children who got the program had different backgrounds, so a simple comparison is biased (replication 1)")
    fig.subplots_adjust(top=0.8, bottom=0.16, left=0.09, right=0.98)
    save(fig, "overlap", mode)


def fig_ate(th, mode):
    r, d = load_json("results.json"), load_json("deep_results.json")
    truth = np.array(r["true_ate_per_rep"])
    per_rep = {**r["ate_abs_error_per_rep"], **{k: v for k, v in d["ate_abs_error_per_rep"].items()}}
    names = {"Naive difference": "Naive difference", "IPW": "IPW (propensity weighting)",
             "AIPW (doubly robust)": "AIPW (doubly robust)", "Double ML (linear)": "Double ML",
             "Causal forest": "Causal forest", "TARNet": "TARNet (neural net)", "DragonNet": "DragonNet (neural net)"}
    rel = {names[k]: float(np.mean(np.array(v) / truth) * 100) for k, v in per_rep.items() if k in names}
    order = sorted(rel, key=rel.get)
    fig, ax = new_fig(th, h=3.8)
    hbar(ax, th, order, [rel[k] for k in order], lambda v: f"{v:.1f}%", highlight={order[0]})
    style(ax, th)
    ax.set_xlabel("Average error in the estimated effect, % of the true effect (lower is better)")
    title(fig, th, "How big is the effect? Propensity weighting gets closest",
          "Error of each method's average-effect estimate across 100 replications")
    fig.subplots_adjust(top=0.82, bottom=0.14, left=0.27, right=0.97)
    save(fig, "average_effect_error", mode)


def fig_coverage(th, mode):
    rows = load_json("ci_results.json")["per_rep"]
    err = np.array([r["ipw"]["est"] - r["truth"] for r in rows])
    lo = np.array([r["ipw"]["lo"] - r["truth"] for r in rows])
    hi = np.array([r["ipw"]["hi"] - r["truth"] for r in rows])
    order = np.argsort(err)
    hit = (lo <= 0) & (hi >= 0)
    fig, ax = new_fig(th, h=3.9)
    for i, k in enumerate(order):
        c = th["blue"] if hit[k] else th["bad"]
        ax.plot([lo[k], hi[k]], [i, i], color=c, linewidth=0.9 if hit[k] else 1.6, solid_capstyle="round")
        ax.plot(err[k], i, "o", color=c, markersize=2)
    ax.axvline(0, color=th["ink2"], linewidth=1)
    ax.text(0.02, len(order) + 1, "true effect", color=th["ink2"], fontsize=8, va="bottom")
    style(ax, th)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("Estimate minus the true effect, with its 95% interval (one line per replication)")
    miss = int((~hit).sum())
    ax.plot([], [], color=th["blue"], linewidth=2, label=f"Interval contains the truth ({len(rows) - miss})")
    ax.plot([], [], color=th["bad"], linewidth=2, label=f"Interval misses it ({miss})")
    ax.legend(frameon=False, fontsize=9, labelcolor=th["ink2"], loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2)
    title(fig, th, f"How sure are we? {len(rows) - miss} of {len(rows)} intervals contain the true effect",
          "95% bootstrap intervals for the propensity-weighting estimate; a well-calibrated interval misses about 5 in 100")
    fig.subplots_adjust(top=0.74, bottom=0.14, left=0.04, right=0.98)
    save(fig, "confidence_intervals", mode)


def fig_robustness(th, mode):
    r = load_json("results.json")["refutations_per_rep"]
    s = load_json("sensitivity_results.json")["summary"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(8.2, 3.6), gridspec_kw={"width_ratios": [1, 1.15], "wspace": 0.75})
    fig.patch.set_facecolor(th["surface"])
    checks = {"Real estimate": np.mean([x["estimate"] for x in r]),
              "Treatment shuffled (placebo)": np.mean([x["placebo_treatment"] for x in r]),
              "Random extra confounder": np.mean([x["random_common_cause"] for x in r]),
              "80% of the data": np.mean([x["subset_80pct_mean"] for x in r])}
    labels = list(checks)
    y = np.arange(len(labels))[::-1]
    vals = [checks[k] for k in labels]
    a1.barh(y, vals, height=0.56, color=[th["neutral"] if "placebo" in k else th["blue"] for k in labels],
            edgecolor=th["surface"], linewidth=2)
    a1.set_yticks(y, labels)
    a1.set_xlim(-0.6, 5.6)
    a1.axvline(0, color=th["axis"], linewidth=1)
    for yi, v in zip(y, vals):
        a1.text(max(v, 0) + 0.1, yi, f"{v:.2f}", va="center", color=th["ink"], fontsize=9)
    style(a1, th)
    a1.set_xlabel("Estimated effect (AIPW)")
    a1.set_title("Placebo and robustness checks", color=th["ink2"], fontsize=9.5, loc="left")

    items = {"Hidden confounder needed\nto erase the effect": s["rv"] * 100,
             "Strongest measured factor,\nlink to the outcome": s["max_observed_partial_r2_outcome"] * 100,
             "Strongest measured factor,\nlink to the treatment": s["max_observed_partial_r2_treatment"] * 100}
    hbar(a2, th, list(items), list(items.values()), lambda v: f"{v:.0f}%",
         highlight={list(items)[0]}, xmax=100)
    style(a2, th)
    a2.set_xlabel("Share of leftover variation explained")
    a2.set_title("Hidden-confounder sensitivity", color=th["ink2"], fontsize=9.5, loc="left")
    title(fig, th, "Could something else explain it? Not easily",
          "Left: the placebo should drop to zero and the other checks should not move it (means). Right: medians. 100 replications")
    fig.subplots_adjust(top=0.76, bottom=0.16, left=0.22, right=0.97)
    save(fig, "robustness", mode)


def fig_per_child(th, mode):
    r, d = load_json("results.json"), load_json("deep_results.json")
    vals = {**{k: v["mean"] for k, v in r["sqrt_pehe_test"].items()}, **{k: v["mean"] for k, v in d["sqrt_pehe_test"].items()}}
    rename = {"Constant effect (AIPW ATE)": "Same effect for everyone"}
    vals = {rename.get(k, k): v for k, v in vals.items()}
    order = sorted(vals, key=vals.get)
    fig, ax = new_fig(th, h=3.9)
    hbar(ax, th, order, [vals[k] for k in order], lambda v: f"{v:.2f}", highlight={"TARNet", "DragonNet"})
    published = {"Causal forest": 3.8, "TARNet": 0.95}
    y = np.arange(len(order))[::-1]
    for name, p in published.items():
        if name in order:
            yi = y[order.index(name)]
            ax.plot([p, p], [yi - 0.36, yi + 0.36], color=th["ink"], linewidth=1.6)
    ax.plot([], [], color=th["ink"], linewidth=1.6, label="Published result (Shalit et al., 2017)")
    ax.legend(frameon=False, fontsize=8.5, labelcolor=th["ink2"], loc="upper right")
    style(ax, th)
    ax.set_xlabel("Error in each child's estimated effect, root PEHE on held-out children (lower is better)")
    title(fig, th, "Who benefits most? Neural models estimate each child's effect best",
          "Error on the 75 held-out children of the published split, averaged over 100 replications")
    fig.subplots_adjust(top=0.8, bottom=0.14, left=0.24, right=0.97)
    save(fig, "per_child_error", mode)


def fig_targeting(th, mode):
    tc = load_json("results.json")["targeting_curve"]
    g = np.array(tc["grid"]) * 100
    fig, ax = new_fig(th)
    ax.plot(g, tc["oracle"], color=th["blue"], linewidth=2, label="Perfect ranking (true effects)")
    ax.plot(g, tc["model"], color=th["orange"], linewidth=2, label="Ranked by causal forest")
    ax.plot(g, tc["random"], color=th["muted"], linewidth=1.5, linestyle="--", label="Treat everyone")
    ax.legend(frameon=False, fontsize=9, labelcolor=th["ink2"], loc="upper right")
    i = int(np.argmin(np.abs(g - 20)))
    ax.plot(g[i], tc["model"][i], "o", color=th["orange"], markersize=8, markeredgecolor=th["surface"], markeredgewidth=2)
    ax.annotate(f"top 20%: {tc['model'][i] / tc['random'][i]:.1f}x the average effect", (g[i], tc["model"][i]),
                xytext=(g[i] + 4, tc["model"][i] - 2.4), color=th["ink"], fontsize=9,
                arrowprops={"arrowstyle": "-", "color": th["muted"], "linewidth": 1})
    style(ax, th, xgrid=False)
    ax.set_xlabel("Share of children targeted, highest predicted benefit first (%)")
    ax.set_ylabel("Average true effect in the group")
    title(fig, th, "Targeting the children who benefit most",
          "Held-out children, 100 replications, scored on the known true effects")
    fig.subplots_adjust(top=0.8, bottom=0.15, left=0.09, right=0.97)
    save(fig, "targeting", mode)


if __name__ == "__main__":
    plt.rcParams["font.family"] = ["Segoe UI", "Helvetica", "Arial", "DejaVu Sans"]
    for mode, th in THEMES.items():
        for f in [fig_overlap, fig_ate, fig_coverage, fig_robustness, fig_per_child, fig_targeting]:
            f(th, mode)
    print("wrote", sorted(p.name for p in FIG.glob("*.png")))
