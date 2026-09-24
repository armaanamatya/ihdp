# IHDP: causal effect estimation, from benchmark to production

How well do causal inference methods recover a treatment effect from observational data, when the true answer is known? And what does it take to serve per-person effect estimates in production?

This repo compares classical and deep causal estimators on the IHDP benchmark, then exports the served model to ONNX behind a FastAPI scoring service.

## Dataset

IHDP is a **semi-synthetic** benchmark (Hill, 2011). The covariates come from a real randomized trial of an early-childhood health and development intervention for low-birth-weight infants. Hill made it observational by removing a non-random subset of treated children (18.6% treated remain) and simulated the outcomes, so each child's true effect is known.

This repo uses the 10 replications distributed with CEVAE: 747 children, 6 continuous and 19 binary covariates. `data.py` downloads them into `data/` on first run.

## Results

Added as each part lands.

## Setup

```
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt
.venv/Scripts/python data.py
```

## References

- J. L. Hill. Bayesian nonparametric modeling for causal inference. JCGS, 2011.
- C. Louizos et al. Causal effect inference with deep latent-variable models (CEVAE). NeurIPS, 2017.
