from __future__ import annotations

import base64
import hashlib
import json
import secrets
import threading
import time
import urllib.parse
import webbrowser
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any

import requests

from .settings import Settings, require_client_id
from .utils import ensure_parent, read_json, utc_now_epoch, write_json

AUTH_URL = "https://x.com/i/oauth2/authorize"
TOKEN_URL = "https://api.x.com/2/oauth2/token"


def _base64_url_no_padding(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).decode("ascii").rstrip("=")


def make_code_verifier() -> str:
    return _base64_url_no_padding(secrets.token_bytes(64))


def make_code_challenge(verifier: str) -> str:
    digest = hashlib.sha256(verifier.encode("ascii")).digest()
    return _base64_url_no_padding(digest)


@dataclass
class CallbackResult:
    code: str = ""
    state: str = ""
    error: str = ""
    error_description: str = ""


class OAuthCallbackServer(HTTPServer):
    def __init__(self, server_address: tuple[str, int], expected_state: str):
        super().__init__(server_address, OAuthCallbackHandler)
        self.expected_state = expected_state
        self.result = CallbackResult()
        self.finished = threading.Event()


class OAuthCallbackHandler(BaseHTTPRequestHandler):
    server: OAuthCallbackServer

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
        return

    def do_GET(self) -> None:  # noqa: N802
        parsed = urllib.parse.urlparse(self.path)
        params = urllib.parse.parse_qs(parsed.query)

        result = CallbackResult(
            code=(params.get("code") or [""])[0],
            state=(params.get("state") or [""])[0],
            error=(params.get("error") or [""])[0],
            error_description=(params.get("error_description") or [""])[0],
        )

        if result.error:
            message = f"X authorization failed: {result.error} {result.error_description}".strip()
            status = 400
        elif result.state != self.server.expected_state:
            result.error = "state_mismatch"
            message = "Authorization failed: state mismatch. Close this tab and run auth again."
            status = 400
        elif not result.code:
            result.error = "missing_code"
            message = "Authorization failed: callback did not include a code. Close this tab and run auth again."
            status = 400
        else:
            message = "Authorization complete. You can close this browser tab and return to the terminal."
            status = 200

        self.server.result = result
        body = f"""<!doctype html>
<html lang=\"en\">
  <head><meta charset=\"utf-8\"><title>X Reposts Obsidian Auth</title></head>
  <body style=\"font-family: system-ui, sans-serif; max-width: 680px; margin: 48px auto; line-height: 1.5;\">
    <h1>X Reposts Obsidian</h1>
    <p>{message}</p>
  </body>
</html>""".encode("utf-8")

        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
        self.server.finished.set()


def _callback_bind_address(redirect_uri: str) -> tuple[str, int, str]:
    parsed = urllib.parse.urlparse(redirect_uri)
    if parsed.scheme != "http":
        raise SystemExit("This local CLI expects an http:// redirect URI such as http://127.0.0.1:8765/callback")
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or 80
    path = parsed.path or "/callback"
    return host, port, path


def _basic_auth_header(client_id: str, client_secret: str) -> str:
    raw = f"{client_id}:{client_secret}".encode("utf-8")
    encoded = base64.b64encode(raw).decode("ascii")
    return f"Basic {encoded}"


class TokenStore:
    def __init__(self, path: Path):
        self.path = path

    def load(self) -> dict[str, Any]:
        if not self.path.exists():
            return {}
        return read_json(self.path)

    def save(self, token_payload: dict[str, Any]) -> dict[str, Any]:
        payload = dict(token_payload)
        expires_in = int(payload.get("expires_in") or 0)
        if expires_in:
            payload["expires_at"] = utc_now_epoch() + max(1, expires_in - 60)
        payload["saved_at"] = utc_now_epoch()
        write_json(self.path, payload)
        try:
            self.path.chmod(0o600)
        except OSError:
            pass
        return payload

    def has_valid_access_token(self) -> bool:
        token = self.load()
        return bool(token.get("access_token") and int(token.get("expires_at") or 0) > utc_now_epoch())


class OAuth2PKCEAuthenticator:
    def __init__(self, settings: Settings):
        require_client_id(settings)
        self.settings = settings
        self.token_store = TokenStore(settings.token_file)

    def build_authorize_url(self, state: str, code_challenge: str) -> str:
        params = {
            "response_type": "code",
            "client_id": self.settings.client_id,
            "redirect_uri": self.settings.redirect_uri,
            "scope": self.settings.scopes,
            "state": state,
            "code_challenge": code_challenge,
            "code_challenge_method": "S256",
        }
        return AUTH_URL + "?" + urllib.parse.urlencode(params)

    def exchange_code_for_token(self, code: str, code_verifier: str) -> dict[str, Any]:
        headers = {"Content-Type": "application/x-www-form-urlencoded"}
        data = {
            "code": code,
            "grant_type": "authorization_code",
            "redirect_uri": self.settings.redirect_uri,
            "code_verifier": code_verifier,
        }
        if self.settings.client_secret:
            headers["Authorization"] = _basic_auth_header(self.settings.client_id, self.settings.client_secret)
        else:
            data["client_id"] = self.settings.client_id

        response = requests.post(TOKEN_URL, headers=headers, data=data, timeout=self.settings.request_timeout_seconds)
        if response.status_code >= 400:
            raise RuntimeError(f"Token exchange failed: HTTP {response.status_code}: {response.text}")
        return response.json()

    def refresh_access_token(self, refresh_token: str) -> dict[str, Any]:
        headers = {"Content-Type": "application/x-www-form-urlencoded"}
        data = {
            "refresh_token": refresh_token,
            "grant_type": "refresh_token",
        }
        if self.settings.client_secret:
            headers["Authorization"] = _basic_auth_header(self.settings.client_id, self.settings.client_secret)
        else:
            data["client_id"] = self.settings.client_id

        response = requests.post(TOKEN_URL, headers=headers, data=data, timeout=self.settings.request_timeout_seconds)
        if response.status_code >= 400:
            raise RuntimeError(f"Token refresh failed: HTTP {response.status_code}: {response.text}")
        return self.token_store.save(response.json())

    def get_access_token(self) -> str:
        token = self.token_store.load()
        access_token = token.get("access_token")
        expires_at = int(token.get("expires_at") or 0)
        if access_token and expires_at > utc_now_epoch():
            return str(access_token)
        refresh_token = token.get("refresh_token")
        if refresh_token:
            refreshed = self.refresh_access_token(str(refresh_token))
            return str(refreshed["access_token"])
        raise SystemExit("No valid X access token. Run: xro auth")

    def login(self, open_browser: bool = True, timeout_seconds: int = 180) -> dict[str, Any]:
        host, port, callback_path = _callback_bind_address(self.settings.redirect_uri)
        state = secrets.token_urlsafe(32)
        verifier = make_code_verifier()
        challenge = make_code_challenge(verifier)
        authorize_url = self.build_authorize_url(state, challenge)

        server = OAuthCallbackServer((host, port), expected_state=state)

        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()

        print(f"Listening for OAuth callback at {self.settings.redirect_uri}")
        print("Open this URL in your browser if it does not open automatically:")
        print(authorize_url)
        if open_browser:
            webbrowser.open(authorize_url)

        if not server.finished.wait(timeout_seconds):
            server.shutdown()
            raise SystemExit(f"Timed out after {timeout_seconds} seconds waiting for OAuth callback.")

        server.shutdown()
        result = server.result
        if result.error:
            raise SystemExit(result.error_description or result.error)

        token_payload = self.exchange_code_for_token(result.code, verifier)
        saved = self.token_store.save(token_payload)
        ensure_parent(self.settings.token_file)
        print(f"Saved OAuth tokens to {self.settings.token_file}")
        return saved
