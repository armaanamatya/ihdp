import numpy as np
import onnxruntime as ort
import pytest
import torch
from fastapi.testclient import TestClient

from data import arrays, load
from deep import CateNet
from serve.app import ART, app

client = TestClient(app)


@pytest.fixture(scope="module")
def X():
    return arrays(load(1))[0]


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_score_shapes_and_flags(X):
    r = client.post("/score", json={"rows": X[:50].tolist(), "threshold": 4.0})
    assert r.status_code == 200
    body = r.json()
    assert len(body["cate"]) == len(body["propensity"]) == len(body["treat"]) == 50
    assert all(0.0 <= p <= 1.0 for p in body["propensity"])
    assert body["treat"] == [c > 4.0 for c in body["cate"]]


@pytest.mark.parametrize("rows", [[], [[0.0] * 24], [[0.0] * 26]])
def test_rejects_bad_input(rows):
    assert client.post("/score", json={"rows": rows}).status_code == 422


def test_onnx_matches_pytorch(X):
    model = CateNet(25, np.zeros(6), np.ones(6), 0.0, 1.0, dragon=True)
    model.load_state_dict(torch.load(ART / "dragonnet.pt"))
    model.eval()
    with torch.no_grad():
        ref = model(torch.tensor(X, dtype=torch.float32)).numpy()
    sess = ort.InferenceSession(str(ART / "dragonnet.onnx"), providers=["CPUExecutionProvider"])
    got = sess.run(None, {"features": X.astype(np.float32)})[0]
    assert np.abs(ref - got).max() < 1e-4
