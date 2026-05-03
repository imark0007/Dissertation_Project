"""
SIEM alert formatter: converts model predictions + ExplanationBundle into
Elastic Common Schema (ECS) compatible JSON alerts.
"""
import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from src.explain.explainer import ExplanationBundle


def format_ecs_alert(
    bundle: ExplanationBundle,
    inference_ms: float = 0.0,
    flow_feature_names: Optional[List[str]] = None,
    rule_name: str = "IoT Dynamic GNN Detector",
) -> Dict[str, Any]:
    """
    Build an ECS-formatted alert dictionary from an ExplanationBundle.

    ECS fields: https://www.elastic.co/guide/en/ecs/current/ecs-reference.html
    """
    severity = "high" if bundle.score >= 0.8 else ("medium" if bundle.score >= 0.5 else "low")
    is_alert = bundle.prediction == 1

    # Resolve feature names
    top_features_named = []
    for f in bundle.top_features:
        idx = f["feature_idx"]
        name = flow_feature_names[idx] if flow_feature_names and idx < len(flow_feature_names) else f"feature_{idx}"
        top_features_named.append({"name": name, "importance": f["importance"]})

    return {
        "@timestamp": datetime.now(timezone.utc).isoformat(),
        "event": {
            "kind": "alert" if is_alert else "event",
            "category": ["threat"] if is_alert else ["network"],
            "type": ["indicator"] if is_alert else ["info"],
            "severity": 75 if is_alert else 25,
            "risk_score": round(bundle.score * 100, 1),
        },
        "rule": {
            "name": rule_name,
            "description": "Dynamic GNN-based IoT traffic anomaly detection",
        },
        "threat": {
            "indicator": {
                "confidence": severity,
            }
        },
        "ml": {
            "prediction": "malicious" if is_alert else "benign",
            "score": round(bundle.score, 4),
            "inference_ms": round(inference_ms, 2),
            "model": "DynamicGNN-GAT-GRU",
        },
        "explanation": {
            "top_nodes": bundle.top_nodes[:5],
            "top_features": top_features_named[:5],
        },
    }


def alert_to_json(alert: Dict[str, Any], indent: int = 2) -> str:
    return json.dumps(alert, indent=indent, default=str)
