from fastapi.middleware.cors import CORSMiddleware

from test_main_claude_keywords import load_main


def test_backend_cors_allows_localhost_and_lan_frontend(monkeypatch):
    main = load_main(monkeypatch)
    cors_middleware = next(
        middleware
        for middleware in main.app.user_middleware
        if middleware.cls is CORSMiddleware
    )

    assert cors_middleware.kwargs["allow_origins"] == [
        "http://localhost:8009",
        "http://localhost:5173",
        "http://192.168.1.20:5173",
    ]
