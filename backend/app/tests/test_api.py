from app.settings import settings


settings.data_dir = r"C:\Users\Acer\Downloads\ai-policy-helper-starter-pack\data"
import requests
import time

BASE_URL = "http://127.0.0.1:8000"

def wait_for_backend():
    """Wait for backend to be ready before running tests."""
    for _ in range(10):
        try:
            r = requests.get(f"{BASE_URL}/api/health")
            if r.status_code == 200 and r.json().get("status") == "ok":
                return True
        except Exception:
            time.sleep(2)
    raise RuntimeError("Backend not ready")

def test_health():
    wait_for_backend()
    r = requests.get(f"{BASE_URL}/api/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"
def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"

def test_ingest_and_ask(client):
    r = client.post("/api/ingest")
    assert r.status_code == 200
    # Ask a deterministic question
    r2 = client.post("/api/ask", json={"query":"What is the refund window for small appliances?"})
    assert r2.status_code == 200
    data = r2.json()
    assert "citations" in data and len(data["citations"]) > 0
    assert "answer" in data and isinstance(data["answer"], str)
