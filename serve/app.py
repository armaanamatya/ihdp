"""FastAPI service that scores per-child treatment effects with the exported ONNX DragonNet."""
import json
import os
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI
from pydantic import BaseModel, Field, conlist

ART = Path(os.environ.get("ARTIFACT_DIR", Path(__file__).resolve().parent.parent / "artifacts"))
META = json.loads((ART / "meta.json").read_text())
N_FEATURES = len(META["features"])
MAX_ROWS = 50_000
OVERLAP = (0.05, 0.95)

opts = ort.SessionOptions()
opts.intra_op_num_threads = int(os.environ.get("ORT_THREADS", "0"))
SESSION = ort.InferenceSession(str(ART / "dragonnet.onnx"), opts, providers=["CPUExecutionProvider"])

app = FastAPI(title="IHDP treatment-effect scoring", version=META["version"])


class ScoreRequest(BaseModel):
    rows: conlist(conlist(float, min_length=N_FEATURES, max_length=N_FEATURES), min_length=1, max_length=MAX_ROWS)
    threshold: float = Field(0.0, description="Recommend treatment when the predicted effect exceeds this")


class ScoreResponse(BaseModel):
    model_version: str
    cate: list[float]
    propensity: list[float]
    treat: list[bool]
    low_overlap: list[bool]
    latency_ms: float


@app.get("/health")
def health():
    return {"status": "ok", "model_version": META["version"]}


@app.get("/metadata")
def metadata():
    return META


@app.post("/score", response_model=ScoreResponse)
def score(req: ScoreRequest):
    start = time.perf_counter()
    out = SESSION.run(None, {"features": np.asarray(req.rows, dtype=np.float32)})[0]
    cate, prop = out[:, 0], out[:, 1]
    return ScoreResponse(
        model_version=META["version"],
        cate=cate.tolist(),
        propensity=prop.tolist(),
        treat=(cate > req.threshold).tolist(),
        low_overlap=((prop < OVERLAP[0]) | (prop > OVERLAP[1])).tolist(),
        latency_ms=(time.perf_counter() - start) * 1000,
    )
