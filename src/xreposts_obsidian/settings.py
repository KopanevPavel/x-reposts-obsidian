from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


DEFAULT_REDIRECT_URI = "http://127.0.0.1:8765/callback"
DEFAULT_SCOPES = "tweet.read users.read offline.access"
DEFAULT_TOKEN_FILE = ".xro/tokens.json"


@dataclass(frozen=True)
class Settings:
    client_id: str
    client_secret: str
    redirect_uri: str
    scopes: str
    token_file: Path
    request_timeout_seconds: float = 30.0


def load_settings(env_file: str | None = None) -> Settings:
    if env_file:
        load_dotenv(env_file)
    else:
        load_dotenv()

    client_id = os.getenv("X_CLIENT_ID", "").strip()
    client_secret = os.getenv("X_CLIENT_SECRET", "").strip()
    redirect_uri = os.getenv("X_REDIRECT_URI", DEFAULT_REDIRECT_URI).strip()
    scopes = os.getenv("X_SCOPES", DEFAULT_SCOPES).strip()
    token_file = Path(os.getenv("XRO_TOKEN_FILE", DEFAULT_TOKEN_FILE)).expanduser()
    request_timeout_seconds = float(os.getenv("XRO_REQUEST_TIMEOUT_SECONDS", "30.0"))

    return Settings(
        client_id=client_id,
        client_secret=client_secret,
        redirect_uri=redirect_uri,
        scopes=scopes,
        token_file=token_file,
        request_timeout_seconds=request_timeout_seconds,
    )


def require_client_id(settings: Settings) -> None:
    if not settings.client_id:
        raise SystemExit(
            "X_CLIENT_ID is empty. Copy .env.example to .env, paste your OAuth 2.0 Client ID, "
            "and make sure X_REDIRECT_URI matches your X app callback URL exactly."
        )
