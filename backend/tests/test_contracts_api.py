"""邊界契約測試 — API 路由冒煙（v154）

⚠️ TestClient(app) 刻意「不進 context manager」：FastAPI lifespan 只在
`with TestClient(...)` 時觸發，main.py 的重副作用（背景 thread、事件日曆
網路同步、asyncio 任務）全在 lifespan 裡 — 不進 with 就完全不跑。
若日後把這裡改成 with 寫法，測試會開始打網路、變慢且不穩定。
"""

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


class TestHealthEndpoint:
    def test_health(self):
        r = client.get("/api/health")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "healthy"
        assert isinstance(body.get("semantic_cache"), dict)
        assert isinstance(body.get("knowledge_fragments"), dict)


class TestGoogleSheetsEndpoints:
    def test_setup_status_shape(self):
        r = client.get("/api/google-sheets/setup")
        assert r.status_code == 200
        body = r.json()
        assert isinstance(body["configured"], bool)
        assert isinstance(body["password_loaded"], bool)

    def test_password_too_short(self):
        r = client.post("/api/google-sheets/setup", json={"password": "short"})
        assert r.status_code == 400


class TestScannerRangeValidation:
    def test_bad_scope_rejected(self):
        r = client.get(
            "/api/scanner/tw-bb-width/range/export",
            params={"start_date": "2026-01-01", "end_date": "2026-01-10", "scope": "bogus"},
        )
        assert r.status_code == 400

    def test_bad_scope_rejected_stream(self):
        r = client.get(
            "/api/scanner/tw-bb-width/range",
            params={"start_date": "2026-01-01", "end_date": "2026-01-10", "scope": "bogus"},
        )
        assert r.status_code == 400
