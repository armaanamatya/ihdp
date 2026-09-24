"""Train the serving model and export it to ONNX.

The service uses DragonNet rather than TARNet: its propensity head lets the API flag inputs with
poor overlap, where any effect estimate is unreliable. (TARNet scored slightly better on PEHE,
but PEHE needs ground truth that a production system never has, so it is not a selection rule.)
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from data import arrays, load
from deep import train

ART = Path(__file__).parent / "artifacts"
FEATURES = [f"x{i}" for i in range(1, 26)]
VERSION = "dragonnet-ihdp-rep1-v1"


def main():
    ART.mkdir(exist_ok=True)
    X, t, y, _ = arrays(load(1))
    model = train(X, t, y, dragon=True, seed=1)
    torch.save(model.state_dict(), ART / "dragonnet.pt")

    onnx_path = ART / "dragonnet.onnx"
    torch.onnx.export(model, torch.zeros(1, 25), onnx_path, input_names=["features"],
                      output_names=["cate_propensity"], dynamic_axes={"features": {0: "batch"}, "cate_propensity": {0: "batch"}},
                      opset_version=17, dynamo=False)

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    with torch.no_grad():
        ref = model(torch.tensor(X, dtype=torch.float32)).numpy()
    got = sess.run(None, {"features": X.astype(np.float32)})[0]
    diff = float(np.abs(ref - got).max())
    assert diff < 1e-4, diff

    meta = {"version": VERSION, "features": FEATURES, "outputs": ["cate", "propensity"],
            "feature_notes": "x1..x6 continuous, unnormalized (the model normalizes them). x7..x25 binary 0/1; "
                             "x14 is 1/2 in the CEVAE CSV and must be sent as 0/1 (value - 1).",
            "trained_on": "IHDP replication 1, all 747 rows", "parity_max_abs_diff": diff,
            "sha256": hashlib.sha256(onnx_path.read_bytes()).hexdigest(),
            "params": sum(p.numel() for p in model.parameters())}
    (ART / "meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
