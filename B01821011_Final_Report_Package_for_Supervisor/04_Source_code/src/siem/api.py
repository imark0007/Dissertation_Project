"""
FastAPI endpoint for scoring and SIEM alert generation.

Run: uvicorn src.siem.api:app --reload
"""
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from fastapi import FastAPI
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.preprocess import load_config
from src.models.dynamic_gnn import DynamicGNN
from src.explain.explainer import explain_sequence
from src.siem.alert_formatter import format_ecs_alert, alert_to_json

app = FastAPI(title="IoT GNN SIEM API", version="1.0")

_MODEL: Optional[DynamicGNN] = None
_CFG: Optional[dict] = None
_DEVICE = torch.device("cpu")


def _load():
    global _MODEL, _CFG
    _CFG = load_config("config/experiment.yaml")
    ckpt = Path("results/checkpoints/dynamic_gnn_best.pt")
    _MODEL = DynamicGNN.from_config(_CFG)
    if ckpt.exists():
        _MODEL.load_state_dict(torch.load(ckpt, map_location=_DEVICE))
    _MODEL.eval()


class FlowWindow(BaseModel):
    features: List[List[float]] = Field(..., description="(N, 46) flow features for one window")


class ScoreRequest(BaseModel):
    windows: List[FlowWindow] = Field(..., description="Sequence of flow windows")


class ScoreResponse(BaseModel):
    prediction: str
    score: float
    inference_ms: float
    alert: Dict[str, Any]


@app.on_event("startup")
def startup():
    _load()


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _MODEL is not None}


@app.post("/score", response_model=ScoreResponse)
def score(req: ScoreRequest):
    from src.data.graph_builder import flows_to_knn_graph
    t0 = time.perf_counter()
    graphs = []
    for w in req.windows:
        feats = np.array(w.features, dtype=np.float32)
        g = flows_to_knn_graph(feats, 0, k=8)
        graphs.append(g)

    bundle = explain_sequence(_MODEL, graphs, _DEVICE, top_k_nodes=5, top_k_features=5)
    elapsed = (time.perf_counter() - t0) * 1000

    feat_names = _CFG["data"]["flow_feature_columns"] if _CFG else None
    alert = format_ecs_alert(bundle, inference_ms=elapsed, flow_feature_names=feat_names)

    return ScoreResponse(
        prediction="malicious" if bundle.prediction == 1 else "benign",
        score=bundle.score,
        inference_ms=elapsed,
        alert=alert,
    )
