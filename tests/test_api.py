"""
API tests: measure inference latency and verify response schema.

Run: python -m pytest tests/test_api.py -v
  or: python tests/test_api.py
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def test_health():
    from fastapi.testclient import TestClient
    from src.siem.api import app
    client = TestClient(app)
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert "status" in data
    assert data["status"] == "ok"
    print("PASS: /health")


def test_score_response_schema():
    from fastapi.testclient import TestClient
    from src.siem.api import app
    client = TestClient(app)

    dummy_window = {"features": np.random.randn(50, 46).tolist()}
    payload = {"windows": [dummy_window] * 3}
    resp = client.post("/score", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert "prediction" in data
    assert "score" in data
    assert "inference_ms" in data
    assert "alert" in data
    alert = data["alert"]
    assert "event" in alert
    assert "ml" in alert
    assert "explanation" in alert
    print(f"PASS: /score (prediction={data['prediction']}, score={data['score']:.4f}, "
          f"inference_ms={data['inference_ms']:.1f})")


def test_inference_latency():
    from fastapi.testclient import TestClient
    from src.siem.api import app
    client = TestClient(app)

    dummy_window = {"features": np.random.randn(50, 46).tolist()}
    payload = {"windows": [dummy_window] * 5}

    times = []
    for _ in range(10):
        t0 = time.perf_counter()
        resp = client.post("/score", json=payload)
        times.append((time.perf_counter() - t0) * 1000)
        assert resp.status_code == 200

    avg = np.mean(times)
    p95 = np.percentile(times, 95)
    print(f"PASS: Latency over 10 calls: avg={avg:.1f}ms, p95={p95:.1f}ms")


def test_memory_usage():
    import psutil
    import os
    process = psutil.Process(os.getpid())
    mem_mb = process.memory_info().rss / 1e6
    print(f"PASS: Current memory usage: {mem_mb:.1f} MB")


if __name__ == "__main__":
    print("=== API Tests ===")
    test_health()
    test_score_response_schema()
    test_inference_latency()
    try:
        test_memory_usage()
    except ImportError:
        print("SKIP: psutil not installed (pip install psutil)")
    print("\nAll tests passed.")
