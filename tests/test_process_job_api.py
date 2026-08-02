"""Contract tests for polling-based image processing jobs."""

from fastapi.testclient import TestClient

from test_main_claude_keywords import FakeExtractor, image_upload, load_main


def test_job_endpoint_returns_accepted_snapshot_then_completed_result(monkeypatch):
    main = load_main(monkeypatch)
    extractor = FakeExtractor(
        result=type("Keywords", (), {"keywords": ["dog"], "occluders": []})()
    )
    monkeypatch.setattr(main, "get_keyword_extractor", lambda: extractor)
    client = TestClient(main.app)

    accepted = client.post(
        "/api/process-image/jobs",
        files={"file": image_upload()},
    )

    assert accepted.status_code == 202
    job_id = accepted.json()["job_id"]
    status = client.get(f"/api/process-image/jobs/{job_id}")
    assert status.status_code == 200
    assert status.json()["status"] == "completed"
    assert status.json()["progress"] == 100
    assert status.json()["result"]["background_base64"] == "background"


def test_unknown_job_returns_404(monkeypatch):
    main = load_main(monkeypatch)

    response = TestClient(main.app).get("/api/process-image/jobs/missing")

    assert response.status_code == 404
    assert response.json() == {"detail": "Processing job not found."}


def test_job_rejects_unsupported_upload_before_creation(monkeypatch):
    main = load_main(monkeypatch)

    response = TestClient(main.app).post(
        "/api/process-image/jobs",
        files={"file": ("notes.txt", b"hello", "text/plain")},
    )

    assert response.status_code == 400
