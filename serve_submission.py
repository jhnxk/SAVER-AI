"""Render entrypoint for the submitted Flask application."""
import hmac
import os

from flask import jsonify, request

from ai_server import app


@app.before_request
def protect_db_sync():
    if request.method != "POST" or request.path.rstrip("/") != "/api/sync-hospitals":
        return None

    token = os.getenv("IEUM_SYNC_TOKEN", "").strip()
    if not token:
        return jsonify(success=False, error="IEUM_SYNC_TOKEN이 설정되지 않았습니다."), 403

    supplied = request.headers.get("Authorization", "").encode()
    if not hmac.compare_digest(supplied, f"Bearer {token}".encode()):
        return jsonify(success=False, error="DB 동기화 토큰이 일치하지 않습니다."), 401
    return None


@app.get("/healthz")
def health_check():
    return jsonify(ok=True)
