"""Benchmark the served model: PyTorch eager vs ONNX Runtime in-process, plus end-to-end HTTP.

Usage:
  python bench.py                             # in-process only
  python bench.py --url http://localhost:8000 # also time a running service
Writes results/bench.json and results/bench.md.
"""
import argparse
import json
import platform
import statistics
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from deep import CateNet

ROOT = Path(__file__).parent
ART, OUT = ROOT / "artifacts", ROOT / "results"
BATCHES = [1, 64, 1024, 16384]


def timeit(fn, repeats):
    for _ in range(5):
        fn()
    ts = []
    for _ in range(repeats):
        s = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - s) * 1000)
    ts.sort()
    return {"p50_ms": statistics.median(ts), "p99_ms": ts[min(len(ts) - 1, int(0.99 * len(ts)))]}


def in_process():
    model = CateNet(25, np.zeros(6), np.ones(6), 0.0, 1.0, dragon=True)
    model.load_state_dict(torch.load(ART / "dragonnet.pt"))
    model.eval()
    sess = ort.InferenceSession(str(ART / "dragonnet.onnx"), providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(0)
    rows = []
    for b in BATCHES:
        x = rng.normal(size=(b, 25)).astype(np.float32)
        xt = torch.from_numpy(x)
        reps = 200 if b <= 1024 else 30
        with torch.inference_mode():
            pt = timeit(lambda: model(xt), reps)
        ox = timeit(lambda: sess.run(None, {"features": x}), reps)
        rows.append({"batch": b, "pytorch": pt, "onnxruntime": ox,
                     "pytorch_rows_per_s": b / pt["p50_ms"] * 1000, "onnxruntime_rows_per_s": b / ox["p50_ms"] * 1000,
                     "speedup_p50": pt["p50_ms"] / ox["p50_ms"]})
    return rows


def http(url):
    import httpx
    rng = np.random.default_rng(1)
    rows = []
    with httpx.Client(base_url=url, timeout=60) as c:
        for b, reps in [(1, 300), (1024, 50)]:
            payload = {"rows": rng.normal(size=(b, 25)).round(4).tolist()}
            res = timeit(lambda: c.post("/score", json=payload).raise_for_status(), reps)
            rows.append({"batch": b, **res, "rows_per_s": b / res["p50_ms"] * 1000})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url")
    args = ap.parse_args()
    res = {"machine": {"cpu": platform.processor(), "torch": torch.__version__, "onnxruntime": ort.__version__,
                       "torch_threads": torch.get_num_threads()},
           "in_process": in_process()}
    if args.url:
        res["http"] = http(args.url)
    OUT.mkdir(exist_ok=True)
    (OUT / "bench.json").write_text(json.dumps(res, indent=2))

    lines = ["# Serving benchmark", "", f"CPU: {res['machine']['cpu']}", "",
             "| Batch | PyTorch p50 (ms) | ONNX Runtime p50 (ms) | Speedup | ONNX rows/s |", "|---|---|---|---|---|"]
    lines += [f"| {r['batch']} | {r['pytorch']['p50_ms']:.3f} | {r['onnxruntime']['p50_ms']:.3f} | "
              f"{r['speedup_p50']:.1f}x | {r['onnxruntime_rows_per_s']:,.0f} |" for r in res["in_process"]]
    if "http" in res:
        lines += ["", "End to end over HTTP", "", "| Batch | p50 (ms) | p99 (ms) | rows/s |", "|---|---|---|---|"]
        lines += [f"| {r['batch']} | {r['p50_ms']:.2f} | {r['p99_ms']:.2f} | {r['rows_per_s']:,.0f} |" for r in res["http"]]
    (OUT / "bench.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
